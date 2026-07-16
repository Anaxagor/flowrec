"""CPU smoke tests for analytical catalog flows and stratified sampling."""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.ft_methods.gflow.exact_flows import (
    CatalogPrefixFlowIndex,
    stratified_catalog_sample,
)

torch.manual_seed(0)
EPS = 1e-6

# Small catalog with many shared prefixes; SIDs must be unique.
raw = torch.randint(0, 6, (400, 4))
catalog = torch.unique(raw, dim=0)
N, L = catalog.shape
index = CatalogPrefixFlowIndex(catalog)

B = 3
dense = torch.rand(B, N).clamp_min(EPS)
# Candidates: catalog rows + 3 off-catalog SIDs (token 11 never appears).
cand_rows = torch.randperm(N)[:50]
off = torch.full((3, L), 11, dtype=torch.long)
cands = torch.cat([catalog[cand_rows], off]).unsqueeze(0).expand(B, -1, -1)

flows = index.prefix_log_flows(cands, dense, reward_epsilon=EPS)
assert flows.shape == (B, cands.size(1), L + 1)

# Brute force.
for b in range(B):
    for m in range(cands.size(1)):
        for d in range(L + 1):
            if d == 0:
                ref = dense[b].sum()
            else:
                match = (catalog[:, :d] == cands[b, m, :d].unsqueeze(0)).all(dim=1)
                ref = dense[b][match].sum() if match.any() else torch.tensor(0.0)
            ref = ref.clamp_min(EPS).log()
            got = flows[b, m, d]
            assert abs(float(got - ref)) < 1e-4, (b, m, d, float(got), float(ref))
print("exact prefix flows vs brute force: OK")

# Terminal consistency: flow at depth L for a catalog item == its own reward.
for b in range(B):
    for j, row in enumerate(cand_rows[:10].tolist()):
        assert abs(float(flows[b, j, L]) - float(dense[b, row].log())) < 1e-4
print("terminal flow == item reward: OK")

# dense_rewards scatter.
top_rows = torch.tensor([[0, 5, 7], [1, 2, 3], [4, 4, 9]])
top_rewards = torch.rand(3, 3) + 0.5
d2 = index.dense_rewards(top_rows, top_rewards, reward_epsilon=EPS)
assert d2.shape == (3, N)
assert abs(float(d2[0, 5]) - float(top_rewards[0, 1])) < 1e-6
assert abs(float(d2[0, 100]) - EPS) < 1e-9
print("dense_rewards scatter: OK")

# Stratified sampling: unbiased weighted sums.
f = torch.rand(N)  # arbitrary per-item statistic
dense1 = (torch.rand(1, N) ** 8).clamp_min(EPS)  # peaked rewards
w_full = dense1[0] / dense1.sum()
target_sum = float((w_full * f).sum())
top_rows1 = torch.argsort(dense1[0], descending=True)[:64].unsqueeze(0)
est, trials = 0.0, 400
for _ in range(trials):
    rows, weights = stratified_catalog_sample(dense1, top_rows1, sample_size=128, top_k=64)
    est += float((weights[0] * f[rows[0]]).sum())
est /= trials
rel = abs(est - target_sum) / max(abs(target_sum), 1e-9)
assert rel < 0.05, (est, target_sum, rel)
# Head rows included exactly, collisions zero-weighted.
rows, weights = stratified_catalog_sample(dense1, top_rows1, sample_size=128, top_k=64)
assert torch.equal(rows[0, :64], top_rows1[0])
coll = (rows[0, 64:].unsqueeze(-1) == top_rows1[0].unsqueeze(0)).any(dim=-1)
assert (weights[0, 64:][coll] == 0).all()
print(f"stratified sampling unbiased (rel err {rel:.3f} over {trials} trials): OK")

print("ALL EXACT-FLOW SMOKE TESTS PASSED")
