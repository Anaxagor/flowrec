"""Post-cold-filter interactions re-keying (remap_interactions_to_catalog).

The cold-item filter compactly renumbers item ids; every datamodule keeps its
interactions frame re-keyed onto the final catalog so positive-history stores
and other consumers never mix id spaces.
"""

import pandas as pd

from flowrec.data.post_split_filtering import remap_interactions_to_catalog


def _frame() -> pd.DataFrame:
    # Pre-filter ids 1..4; the filter kept items A, C, D and renumbered them.
    return pd.DataFrame(
        {
            "raw_user_id": ["u1", "u1", "u1", "u2", "u2"],
            "raw_item_id": ["A", "B", "C", "D", "B"],
            "user_id": [0, 0, 0, 1, 1],
            "item_id": [1, 2, 3, 4, 2],
            "timestamp": [10, 20, 30, 10, 20],
            "position": [0, 1, 2, 0, 1],
        }
    )


POST_FILTER_ITEM2ID = {"A": 1, "C": 2, "D": 3}  # B removed


class TestRemapInteractionsToCatalog:
    def test_catalog_items_get_post_filter_ids(self) -> None:
        out = remap_interactions_to_catalog(_frame(), POST_FILTER_ITEM2ID)
        assert out["item_id"].tolist() == [1, 0, 2, 3, 0]

    def test_removed_items_become_pad_and_rows_survive(self) -> None:
        frame = _frame()
        out = remap_interactions_to_catalog(frame, POST_FILTER_ITEM2ID)
        # No rows dropped: positions and per-user lengths stay exact for
        # position-based train-boundary computations (popularity, ARP).
        assert len(out) == len(frame)
        assert out["position"].tolist() == frame["position"].tolist()
        assert (out.loc[out["raw_item_id"] == "B", "item_id"] == 0).all()

    def test_prefilter_ids_preserved(self) -> None:
        frame = _frame()
        out = remap_interactions_to_catalog(frame, POST_FILTER_ITEM2ID)
        assert out["item_id_prefilter"].tolist() == frame["item_id"].tolist()

    def test_input_frame_untouched(self) -> None:
        frame = _frame()
        remap_interactions_to_catalog(frame, POST_FILTER_ITEM2ID)
        assert frame["item_id"].tolist() == [1, 2, 3, 4, 2]
        assert "item_id_prefilter" not in frame.columns

    def test_ids_stay_inside_final_vocab(self) -> None:
        out = remap_interactions_to_catalog(_frame(), POST_FILTER_ITEM2ID)
        vocab = len(POST_FILTER_ITEM2ID) + 1  # + pad
        assert out["item_id"].between(0, vocab - 1).all()
        assert out["item_id"].dtype == "int64"
