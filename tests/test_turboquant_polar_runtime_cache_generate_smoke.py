from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.llama_score_patch import (
    patch_llama_model_with_turboquant_scores,
)
from turboquant.runtime_cache import TurboQuantRuntimeCache
from turboquant.polar_runtime_cache import (
    TurboQuantPolarRuntimeCache,
)
from turboquant.polarquant import (
    recursive_polar_encode,
)
from turboquant.polarquant_quant import (
    DEFAULT_POLAR_BITS_BY_LEVEL,
    fit_polar_angle_codebooks_from_encodings,
)
from turboquant.qjl import (
    make_gaussian_sketch,
)

def load_cached_svd_model(path: str, device: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    model = obj["model"]
    tokenizer = obj["tokenizer"]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    model.to(device)
    return model, tokenizer


@torch.no_grad()
def main():
    device = "cuda:0"
    cached_model = "runs/svd_uniform_08/model/svd_uniform_08_cached.pt"

    prompt = "TurboQuant is useful for language model inference because"

    model, tokenizer = load_cached_svd_model(
        cached_model,
        device,
    )

    tq_state = patch_llama_model_with_turboquant_scores(
        model,
        qjl_m=256,
        device=device,
    )

    # ============================================================
    # Build TurboQuant Polar runtime cache
    # ============================================================

    D = model.config.hidden_size // model.config.num_attention_heads
    L = 4
    M = 256

    # Small synthetic calibration set for smoke test only.
    # Later runtime benchmark should use real K-activation calibration.
    x_calib = torch.randn(
        4096,
        D,
        device=model.device,
        dtype=torch.float32,
    )

    enc_calib = recursive_polar_encode(
        x_calib,
        num_levels=L,
    )

    codebooks = fit_polar_angle_codebooks_from_encodings(
        [enc_calib],
        bits_by_level=DEFAULT_POLAR_BITS_BY_LEVEL,
        max_iters=30,
        max_samples_per_level=200_000,
        seed=0,
    )

    sketch = make_gaussian_sketch(
        d=D,
        m=M,
        device=model.device,
        dtype=torch.float32,
        seed=123,
    )

    runtime_cache = TurboQuantPolarRuntimeCache(
        num_layers=model.config.num_hidden_layers,
        codebooks=codebooks,
        sketch=sketch,
        num_levels=L,
    )

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=runtime_cache,
        max_new_tokens=4,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(
        out[0],
        skip_special_tokens=True,
    )

    print("========== TurboQuant Polar runtime-cache generate ==========")
    print(text)
    print()

    print("----- Token IDs -----")
    print(out[0].tolist())
    print()

    report = runtime_cache.report()
    print("----- Runtime cache report -----")
    print(report)
    print()

    layer_lens = [
        runtime_cache.get_seq_length(i)
        for i in range(len(model.model.layers))
    ]

    print("layer cache seq lens:", layer_lens)

    assert len(set(layer_lens)) == 1, "Layer cache sequence lengths disagree."
    assert layer_lens[0] == 14, f"Expected runtime cache seq_len=14, got {layer_lens[0]}"

    print("[PASS] TurboQuant CUDA runtime-cache generation smoke test passed.")


if __name__ == "__main__":
    main()
