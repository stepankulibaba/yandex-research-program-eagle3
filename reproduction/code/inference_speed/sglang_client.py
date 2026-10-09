"""Ask a running SGLang server the EAGLE benchmark questions one at a time (batch 1) and time every answer.

    python sglang_client.py <server url> <model dir> <bench> <out.json> [temperature, default 0]
    env: MAX_Q (questions, default 80), SPECULATIVE=1 for an EAGLE-3 server, SAMPLING_SEED

Mirrors the authors' eval script: the same system prompt and chat template, every turn of the question (the
answer to turn 1 goes into the context of turn 2), at most 512 new tokens per turn, stop on eos / <|eot_id|>,
warm-up = the first question with all its turns, three times. (SGLang additionally always stops on the eos list of
the model's generation_config; for this model those extra ids are practically never generated.)
Per turn it stores the time of the HTTP round trip, the generated tokens and, for EAGLE-3, the number of
verification rounds and of accepted draft tokens.
"""
import os
from pathlib import Path
import sys
import time

from common import SYSTEM, atomic_json, digest, questions, validate_answers

MAX_NEW_TOKENS = 512
MAX_CONTEXT = 1979      # prompt + answer budget per turn; a longer prompt is an error, never silently truncated


def stop_ids(tok, model_dir=None):
    """The authors' stop tokens: the tokenizer's eos and <|eot_id|> (the same id for this model)."""
    ids = {tok.eos_token_id, tok.convert_tokens_to_ids('<|eot_id|>')}
    return sorted(i for i in ids if isinstance(i, int) and i >= 0 and i != tok.unk_token_id)


def clean_text(text, tok):
    """Answer text without special tokens, as it goes back into the next turn's context."""
    for token in tok.special_tokens_map.values():
        for value in token if isinstance(token, list) else [token]:
            text = text.replace(str(value), '')
    return text.strip()


def generate(server, ids, temperature=0.0, *, stops, seed=0, speculative=False):
    import requests
    limit = min(MAX_NEW_TOKENS, MAX_CONTEXT - len(ids))
    if limit <= 0:
        raise ValueError('Prompt exceeds context budget; no silent truncation')
    params = {'temperature': temperature, 'max_new_tokens': limit, 'stop_token_ids': stops,
              'top_p': 1.0, 'top_k': -1, 'min_p': 0.0, 'sampling_seed': seed,
              'frequency_penalty': 0.0, 'presence_penalty': 0.0, 'repetition_penalty': 1.0}
    t0 = time.perf_counter()
    response = requests.post(f'{server}/generate', json={'input_ids': ids, 'sampling_params': params},
                             timeout=(10, 600))
    response.raise_for_status()
    seconds = time.perf_counter() - t0
    out = response.json()
    meta = out['meta_info']

    # SGLang 0.5.9 reports these only when spec_verify_ct > 0, and calls the accepted count spec_accept_token_num.
    steps, accepted = meta.get('spec_verify_ct'), meta.get('spec_accept_token_num')
    if speculative and (steps is None or accepted is None):
        if meta['completion_tokens'] > 1:
            raise ValueError('Missing speculation counters: wrong server/version')
        steps, accepted = 0, 0      # finished at prefill: no verification round
    if meta['completion_tokens'] > limit or not meta.get('finish_reason'):
        raise ValueError('Invalid completion length/finish reason')
    return {'text': out['text'], 'tokens': meta['completion_tokens'], 'steps': steps,
            'accepted_draft_tokens': accepted, 'seconds': seconds, 'timing': 'http_round_trip',
            'finish_reason': meta['finish_reason'], 'prompt_ids': ids, 'prompt_sha256': digest(ids),
            'output_ids': out.get('output_ids'), 'sampling_params': params, 'meta_info': meta}


def main(server, model_dir, bench, out_path, temperature='0'):
    from transformers import AutoTokenizer
    temperature = float(temperature)
    tok = AutoTokenizer.from_pretrained(model_dir)
    expected = questions(Path(__file__).parent / 'EAGLE', bench, int(os.environ.get('MAX_Q', '80')))
    stops = stop_ids(tok, model_dir)
    speculative = os.environ.get('SPECULATIVE', '0') == '1'
    seed = int(os.environ.get('SAMPLING_SEED', '0'))

    def answer(question):
        """All turns of one question; each answer goes into the context of the next turn."""
        messages = [{'role': 'system', 'content': SYSTEM}]
        turns = []
        for user_turn in question['turns']:
            messages.append({'role': 'user', 'content': user_turn})
            ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            turn = generate(server, ids, temperature, stops=stops, seed=seed, speculative=speculative)
            turn['clean_text'] = clean_text(turn['text'], tok)
            turns.append(turn)
            messages.append({'role': 'assistant', 'content': turn['clean_text']})
        return turns

    for _ in range(3):                  # warm-up, as the authors: the first question, all turns
        answer(expected[0])
    rows = [{'question_id': q['question_id'], 'turns': answer(q)} for q in expected]
    atomic_json(out_path, rows)
    validate_answers(out_path, expected, 'sglang')
    print(f'{bench}: {len(rows)} complete questions; HTTP latency; seed={seed}', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:6])
