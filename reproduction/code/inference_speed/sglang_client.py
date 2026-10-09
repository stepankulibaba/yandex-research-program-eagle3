"""Serial HTTP benchmark with explicit stopping, seeds, prompts and counters.
The bounded SGLang protocol is separate from the unchanged authors' protocol.
"""
import os
from pathlib import Path
import sys
import time
from runtime import SYSTEM, atomic_json, digest, read_json, questions, validate_answers

MAX_NEW_TOKENS = 512
MAX_CONTEXT = 1979


def stop_ids(tok, model_dir):
    ids = {tok.eos_token_id, tok.convert_tokens_to_ids('<|eot_id|>')}
    p = Path(model_dir) / 'generation_config.json'
    if p.exists():
        eos = read_json(p).get('eos_token_id', [])
        ids.update(eos if isinstance(eos, list) else [eos])
    return sorted(i for i in ids if isinstance(i, int) and i >= 0 and i != tok.unk_token_id)


def clean_text(text, tok):
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
    r = requests.post(f'{server}/generate', json={'input_ids': ids, 'sampling_params': params}, timeout=(10, 600))
    r.raise_for_status()
    seconds = time.perf_counter() - t0
    out = r.json()
    meta = out['meta_info']
    # SGLang 0.5.9 (tokenizer_manager._calculate_spec_decoding_metrics) reports these only when
    # spec_verify_ct > 0, and names the accepted count spec_accept_token_num.
    steps, accepted = meta.get('spec_verify_ct'), meta.get('spec_accept_token_num')
    if speculative and (steps is None or accepted is None):
        if meta['completion_tokens'] > 1:
            raise ValueError('Missing speculation counters: wrong server/version')
        steps, accepted = 0, 0    # finished at prefill: no verification round
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
    warm = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': expected[0]['turns'][0]}]
    warm_ids = tok.apply_chat_template(warm, tokenize=True, add_generation_prompt=True)
    for _ in range(3):
        generate(server, warm_ids, temperature, stops=stops, seed=seed, speculative=speculative)
    rows = []
    for q in expected:
        messages = [{'role': 'system', 'content': SYSTEM}]
        turns = []
        for user in q['turns']:
            messages.append({'role': 'user', 'content': user})
            ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            turn = generate(server, ids, temperature, stops=stops, seed=seed, speculative=speculative)
            turn['clean_text'] = clean_text(turn['text'], tok)
            turns.append(turn)
            messages.append({'role': 'assistant', 'content': turn['clean_text']})
        rows.append({'question_id': q['question_id'], 'turns': turns})
    atomic_json(out_path, rows)
    validate_answers(out_path, expected, 'sglang')
    print(f'{bench}: {len(rows)} complete questions; HTTP latency; seed={seed}', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:6])
