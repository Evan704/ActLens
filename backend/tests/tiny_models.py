"""Tiny random-init models, one per registered architecture (no downloads). The parametrized contract tests in
test_archs.py run on every entry, so supporting a new architecture means adding an adapter *and* a line here."""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

VOCAB, D, NH, NKV, DH, INTER, LAYERS = 97, 32, 4, 2, 8, 64, 3
COMMON = dict(vocab_size=VOCAB, hidden_size=D, intermediate_size=INTER, num_hidden_layers=LAYERS,
              num_attention_heads=NH, num_key_value_heads=NKV, head_dim=DH, max_position_embeddings=64,
              attn_implementation="eager")


def llama():
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(**COMMON)).eval()


def qwen3():  # adds per-head q_norm / k_norm
    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config(**COMMON)).eval()


def gpt2():  # LayerNorm, fused QKV, learned positions, non-gated MLP
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=VOCAB, n_embd=D, n_inner=INTER, n_layer=LAYERS, n_head=NH, n_positions=64,
                     bos_token_id=0, eos_token_id=0, attn_implementation="eager")
    return GPT2LMHeadModel(cfg).eval()


def auto(model_type, **kw):
    """Factory for any `model_type` that `AutoModelForCausalLM` knows, built from COMMON plus overrides `kw`."""
    def make():
        torch.manual_seed(0)
        cfg = AutoConfig.for_model(model_type, **{**COMMON, "bos_token_id": 0, "eos_token_id": 0, "pad_token_id": 0, **kw})
        return AutoModelForCausalLM.from_config(cfg).eval()
    return make


# name -> factory
TINY = {
    "llama": llama, "qwen3": qwen3, "gpt2": gpt2,
    # real Gemma has head_dim (256) != hidden/heads and MQA on the 2B model: pin both here
    "gemma": auto("gemma", head_dim=16, num_key_value_heads=1),
    "stablelm": auto("stablelm", partial_rotary_factor=0.5),  # LayerNorm, partial RoPE
    "olmo": auto("olmo", clip_qkv=0.05),  # non-parametric LayerNorm; a tiny clip_qkv makes the clamp bite
}
