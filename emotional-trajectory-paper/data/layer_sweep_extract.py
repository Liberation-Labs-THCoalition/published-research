"""Layer Sweep Extraction — all layers for circumplex trajectory analysis.

Extracts residual, pre-RoPE K, and V at EVERY layer for a subset of prompts.
Enables tracing emotion geometry through the full transformer stack.

Usage:
  python layer_sweep_extract.py --model-dir /path/to/model --stimuli stimuli.json --output sweep_activations.npz
  python layer_sweep_extract.py --model-dir /path/to/model --stimuli stimuli.json --n-prompts 50
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_all_layers(model, tokenizer, prompts, device="cpu"):
    """Extract residual, K, V at every layer for all prompts."""

    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    hidden_dim = model.config.hidden_size
    head_dim = hidden_dim // model.config.num_attention_heads

    all_layers = list(range(n_layers))

    # Pre-allocate results
    results = {"token_counts": []}
    for l in all_layers:
        results[f"residual_L{l}"] = []
        for h in range(n_kv):
            results[f"k_L{l}_H{h}"] = []
            results[f"v_L{l}_H{h}"] = []

    # Register hooks on all layers
    captured_residuals = {}
    captured_k = {}
    captured_v = {}
    hooks = []

    def make_residual_hook(layer_idx):
        def hook_fn(module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is not None:
                captured_residuals[layer_idx] = hidden.detach()
            return None
        return hook_fn

    def make_k_hook(layer_idx):
        def hook_fn(module, input, output):
            captured_k[layer_idx] = output.detach()
        return hook_fn

    def make_v_hook(layer_idx):
        def hook_fn(module, input, output):
            captured_v[layer_idx] = output.detach()
        return hook_fn

    for l in all_layers:
        layer = model.model.layers[l]
        hooks.append(layer.register_forward_pre_hook(make_residual_hook(l), with_kwargs=True))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(l)))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(l)))

    print(f"Hooks on all {n_layers} layers ({n_layers * 3} hooks)")

    for i, prompt in enumerate(prompts):
        if i % 10 == 0:
            print(f"  Processing {i}/{len(prompts)}...")

        inputs = tokenizer(prompt, return_tensors="pt", padding=False).to(device)
        seq_len = inputs["input_ids"].shape[1]
        results["token_counts"].append(seq_len)

        with torch.no_grad():
            model(**inputs)

        for l in all_layers:
            res = captured_residuals[l][0, -1, :].cpu().float().numpy()
            results[f"residual_L{l}"].append(res)

            k_full = captured_k[l][0, -1, :].cpu().float().numpy()
            v_full = captured_v[l][0, -1, :].cpu().float().numpy()

            for h in range(n_kv):
                results[f"k_L{l}_H{h}"].append(k_full[h * head_dim:(h + 1) * head_dim])
                results[f"v_L{l}_H{h}"].append(v_full[h * head_dim:(h + 1) * head_dim])

        captured_residuals.clear()
        captured_k.clear()
        captured_v.clear()

    for h in hooks:
        h.remove()

    for key in results:
        results[key] = np.array(results[key])

    return results, all_layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--output", default="sweep_activations.npz")
    parser.add_argument("--n-prompts", type=int, default=None,
                        help="Use subset of prompts (default: all)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    stimuli = json.loads(Path(args.stimuli).read_text())

    if args.n_prompts:
        # Balanced subset
        by_cat = {}
        for s in stimuli:
            by_cat.setdefault(s["category"], []).append(s)
        per_cat = args.n_prompts // len(by_cat)
        stimuli = []
        for cat, items in sorted(by_cat.items()):
            stimuli.extend(items[:per_cat])

    prompts = [s["prompt"] for s in stimuli]
    categories = [s["category"] for s in stimuli]

    print(f"Loading model from {args.model_dir}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    except (AttributeError, Exception):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
        print("  (used fallback tokenizer)")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.float32, trust_remote_code=True)
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    print(f"Model: {n_layers} layers, {model.config.hidden_size}d, {n_kv} KV heads")
    print(f"Extracting ALL layers for {len(prompts)} prompts")

    results, all_layers = extract_all_layers(model, tokenizer, prompts, args.device)

    # Sanity check
    print(f"\nSanity check:")
    for l in [0, n_layers // 2, n_layers - 1]:
        res = results[f"residual_L{l}"]
        print(f"  L{l}: shape={res.shape}, mean={res.mean():.4f}, std={res.std():.4f}, "
              f"nan={np.isnan(res).sum()}")

    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(outpath),
        categories=np.array(categories),
        all_layers=np.array(all_layers),
        n_kv_heads=np.array(n_kv),
        n_layers=np.array(n_layers),
        **results,
    )
    print(f"\nSaved to {outpath} ({outpath.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
