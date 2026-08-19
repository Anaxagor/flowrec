"""CPU smoke test for exact analytical flows over full-catalog rewards."""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.ft_methods.gflow.exact_flows import CatalogPrefixFlowIndex

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

print("ALL EXACT-FLOW SMOKE TESTS PASSED")
