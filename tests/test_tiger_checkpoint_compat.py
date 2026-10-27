import torch
import pytest
import flowrec.models.tiger as tiger_module

from flowrec.models.tiger import TigerModel


@pytest.mark.parametrize('force_tied_weights', [False, True])
def test_tiger_keeps_input_and_output_weights_independent_when_loading_checkpoint(monkeypatch, force_tied_weights):
    if force_tied_weights:
        original = tiger_module.T5ForConditionalGeneration

        def tied_t5(*args, **kwargs):
            t5 = original(*args, **kwargs)
            t5.config.tie_word_embeddings = True
            t5.tie_weights()
            return t5

        # Exercise Transformers 5's forced tying even on older test runtimes.
        monkeypatch.setattr(tiger_module, 'T5ForConditionalGeneration', tied_t5)
    model = TigerModel(
        codebook_size=2, embedding_dim=16, dim_feedforward=32, num_heads=2, d_kv=8,
        num_encoder_layers=1, num_decoder_layers=1, user_ids_count=8,
        semantic_id_mapping={'a': [1, 3], 'b': [2, 4]}, sem_id_len=2,
    )
    t5 = model.model
    assert not t5.config.tie_word_embeddings
    assert t5.shared.weight.data_ptr() != t5.lm_head.weight.data_ptr()

    # Legacy checkpoints contain shared encoder/decoder embeddings and an
    # independent output projection. Strict loading must preserve both values.
    state = {key: value.clone() for key, value in model.state_dict().items()}
    for key in ('model.shared.weight', 'model.encoder.embed_tokens.weight',
                'model.decoder.embed_tokens.weight'):
        state[key].fill_(0.125)
    state['model.lm_head.weight'].fill_(0.75)
    model.load_state_dict(state, strict=True)
    torch.testing.assert_close(t5.shared.weight, state['model.shared.weight'])
    torch.testing.assert_close(t5.lm_head.weight, state['model.lm_head.weight'])
    assert t5.shared.weight.data_ptr() != t5.lm_head.weight.data_ptr()
