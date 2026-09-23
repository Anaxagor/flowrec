# MiniOneRec reward runs (Beauty)

Each `beauty_*.json` starts from the same tuned TIGER checkpoint and uses the
same data split, generated group size (8), optimizer and evaluation settings.
The dataset uses the exp2 `verified_purchase == True` criterion. Checkpoints
and reward assets are paths relative to the repository root. The parquet
reviews and metadata files must also be present there on the training host.

Run any variant with:

```bash
python scripts/run_tiger.py --config configs/MiniOneRec/beauty_sasrec.json --device cuda:0
```

Available suffixes: `rule_rank` (existing binary plus rank penalty), `binary`
(GFlow positive/negative reward), `sasrec`, `item_prior` (the exp4 shrunk
per-item rating prior), `sim_embed`, `sim_hier`, `interaction`, `svd`, `mix`
and `multiply`. The last two combine SASRec, item prior and hierarchical
similarity with equal weights. `mix` uses `normalize: none`, so it does not
depend on the generated group. `reward_power` can sharpen or flatten an
external reward; it defaults to 1. The configured external reward replaces
the `rule_rank` reward. The rank penalty is used only in `rule_rank`.

These runs evaluate the same reward-model classes used to build GFlow reward
caches, directly on MiniOneRec rollouts. They do not use GFlow's full-catalog
cache or its full-catalog normalization. The `interaction` signal depends on
past logged evidence and may be close to neutral if the source has little
auxiliary evidence. The `item_prior` uses the full raw reviews frame, matching
the exp4 GFlow rating-prior convention; it is distinct from `binary`.
