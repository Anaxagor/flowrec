"""Run sampled reward CE on a global checkpoint and its prepared GFlow rows."""
from __future__ import annotations

from copy import copy

import ce_distill as amazon_global
import ce_distill_gpt2 as ce
import ce_distill_gpt2_yambda as yambda_ce
import ce_from_sft
import gflow_finetune_yambda as gflow


class GlobalYambdaCEProfile(amazon_global.GlobalProfile):
    """Use the bound global student, catalog, and factor caches for both model families."""

    def __init__(self, session, dm):
        self.session = session
        self.dm = dm
        self.model_name = session.source["model"]
        self.dataset = session.source["dataset"]
        self.dataset_root = session.output_root
        directories = session.cache_dirs()
        self.cache_dirs = {
            spec["name"]: directories[key]
            for key, spec in zip(session.config["method"]["component_keys"],
                                 session.config["method"]["reward"]["components"])
        }

    def matched_args(self, args, dataset, seed, matched):
        return self.session.args

    def build_run_config(self, base, dataset, b):
        config = dict(self.session.config)
        # Exp7 uses sim_hier for the hierarchical factor's internal key on both models.
        config["method"] = {**config["method"], "component_keys": list(ce.INTERNAL_COMPONENTS.values())}
        config["pretrained_model"] = config["model"]
        config["global_init"] = self.session.source_identity
        return config

    def load_data(self, args, base, b, config, config_path):
        return self.dm

    def finetune_indices(self, train_dataset, b):
        # The compact dataset already contains the selected rows in reward-cache order.
        return None

    def component_cache_dir(self, base, dataset, name, b):
        return self.cache_dirs[name]

    def check_component_manifest(self, base, config, path, name, num_examples, b):
        # Global preparation verifies source, subset hash, numeric checks and catalog.
        # The CE loader additionally truncates its input when running a smoke test.
        manifest = ce.read_json(path)
        if manifest["num_examples"] != num_examples:
            raise ValueError(f"CE expects {num_examples} rows for {name}, found {manifest['num_examples']} at {path}")
        return manifest

    verify_caches = staticmethod(yambda_ce.verify_caches)

    def audit_data(self, base, dm, b):
        audit = dict(dm._ce_prepared_audit)
        if "finetune_subset" in self.session.config:
            audit["finetune_subset"] = self.session.config["finetune_subset"]
        return audit

    def load_initial_model(self, base, dataset, dm, config, config_path):
        checkpoint = self.session.checkpoint
        if self.model_name == "gpt2":
            model, _ = self.session.base.build_gpt2_from_tuned_checkpoint(dataset, dm, self.session.args)
        else:
            model = base.tiger_runner.build_model(config, dm, config_path=config_path)
            base.control.load_tiger_checkpoint(model, checkpoint)
            base.gpt2_beauty.check_semantic_row_table(model, dm)
        return model, checkpoint

    def asset_paths_for(self, base, dataset, config, components, component_paths):
        return ce_from_sft.asset_paths(self.session, config, components, component_paths)

    def trainer_overrides(self, args, b):
        return ce_from_sft.trainer_overrides(self.session)

    def build_identity(self, args, entry, config, trainer_config, assets, curriculum_steps):
        identity = ce.shared.build_identity(args, entry, config, trainer_config, assets, curriculum_steps)
        identity.update(model=self.model_name, split_strategy="global_time",
                        init_checkpoint=str(self.session.checkpoint))
        return identity

    def result_metadata(self, args, entry, b, config):
        if self.dataset == "beauty_2023":
            targets = "all_dense_window_labels" if self.model_name == "gpt2" else "all_prefixes"
        else:
            targets = "sampled50" if self.dataset == gflow.Y500 else "last_128"
        return {"split_strategy": "global_time", "init_checkpoint": str(self.session.checkpoint),
                "target_selection": targets,
                "global_init": self.session.source_identity,
                "finetune_subset": config.get("finetune_subset")}


def run_cell(session, dm, cell):
    """Train one CE configuration and return its validation-selected result."""
    import gflow_finetune_gpt2_amazon as base

    profile = GlobalYambdaCEProfile(session, dm)
    if (cell["model"], cell["dataset"], cell["seed"]) != (profile.model_name, profile.dataset, session.args.seed):
        raise ValueError("CE cell differs from its bound global source")
    args = copy(session.args)
    args.protocol_path = gflow.PROTOCOL
    args.output_root = session.output_root
    args.estimator, args.sampling, args.num_samples = cell["estimator"], cell["sampling"], cell["num_samples"]
    args.catalog_chunk_size = 16
    args.proposal_q_mix = 0.1  # Used only by the importance-corrected proposal.
    args.calibration_batches = args.gflow_calibration_batches
    args.calibration_seed = args.gflow_calibration_seed
    interval = getattr(args, "checkpoint_every_updates", None)
    args.checkpoint_every_updates = 100 if interval is None else interval
    entry = {key: cell[key] for key in ("dataset", "seed", "combo", "power", "rho")}
    directory = ce.run_dir_for(args, entry, dataset_root=session.output_root)
    with base.control.file_lock(directory / "run.lock"):
        ce.run_entry(args, entry, None, base, profile=profile)
        result = ce.read_json(directory / "result.json")
        if result["status"] != "completed":
            raise RuntimeError(f"CE training failed at {directory}")
        return result
