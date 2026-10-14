"""Independent fixed-continuation matcher and pinned posterior oracle checks."""
import random
import torch

EOF = -2


def match_prefix(candidates, continuation):
    assert continuation and candidates and all(row[0] == continuation[0] for row in candidates)
    counts = []
    for path in candidates:
        n = 0
        for token, expected in zip(path, continuation):
            if token < 0 or token != expected:
                break
            n += 1
        counts.append(n)
    maximum = max(counts)
    assert maximum >= 1
    return counts.index(maximum), maximum


def posterior_oracle(candidates, continuation, evaluator):
    # A bijection preserves equality and removes the need for 152K-wide test logits.
    vocabulary = sorted({x for row in candidates for x in row if x >= 0} | set(continuation) | {EOF})
    vocab = {token: index for index, token in enumerate(vocabulary)}
    compact = torch.tensor([[vocab[x] if x >= 0 else -1 for x in row] for row in candidates])
    rows, depth = compact.shape
    logits = torch.full((rows, depth, len(vocabulary)), -1.0)
    for j in range(depth):
        expected = continuation[j + 1] if j + 1 < len(continuation) else EOF
        logits[:, j, vocab[expected]] = 1.0
    best, accept, _ = evaluator(logits, compact, None)
    return int(best), int(accept) + 1


def cpu_gate(evaluator):
    cases = [([[0]], [0]), ([[0, 1, -1], [0, 2, 3]], [0, 1, 4]),
             ([[0, 1, 2], [0, 1, 2]], [0, 1, 2, 3]),
             ([[0, 1, 2], [0, 1, 3]], [0]),
             ([[0, 1, -1], [0, 1, 2]], [0, 1]),
             ([[0, 1, 2], [0, 2, 1]], [0, 3, 4])]
    rng = random.Random(20261008)
    for _ in range(58):
        depth = rng.randrange(2, 9)
        continuation = [0] + [rng.randrange(1, 17) for _ in range(rng.randrange(0, depth + 2))]
        paths = []
        for i in range(rng.randrange(1, 20)):
            length = rng.randrange(1, depth + 1)
            path = [0] + [rng.randrange(1, 17) for _ in range(length - 1)]
            shared = rng.randrange(0, min(length, len(continuation)))
            path[:shared + 1] = continuation[:shared + 1]
            paths.append(path + [-1] * (depth - length))
        cases.append((paths, continuation))
    result = []
    for i, (paths, continuation) in enumerate(cases):
        actual, expected = match_prefix(paths, continuation), posterior_oracle(paths, continuation, evaluator)
        result.append({'case': i, 'best': actual[0], 'length': actual[1], 'passed': actual == expected})
    assert len(result) == 64 and all(r['passed'] for r in result)
    return result
