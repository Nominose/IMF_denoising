"""
v-head end-to-end sanity check.

Verifies on the actual training environment that:
  1. import resolves to the top-level conditional_diffusion.py (not the inner duplicate)
  2. that file has auxiliary_v_head support
  3. Unet(auxiliary_v_head=True) adds parameters and sets the flag
  4. forward pass with v-head succeeds on GPU
  5. ImprovedMeanFlow.forward returns a valid loss with v-head enabled

Run via run_vhead_check.sh (sbatch).
"""

import sys
import torch


def main() -> int:
    rc = 0

    # ---- 1. Module resolution ----
    print("=== 1. Module resolution ===", flush=True)
    import IMF_denoising.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion as cd
    print("Module path:", cd.__file__, flush=True)

    inner_marker = "Thinslice_experiments"
    if inner_marker in cd.__file__:
        print(f"[FAIL] import resolves to the INNER duplicate ({inner_marker} in path)", flush=True)
        rc = 1
    else:
        print("[OK] import resolves to top-level package.", flush=True)

    with open(cd.__file__) as f:
        content = f.read()
    has_v_head_in_source = "auxiliary_v_head" in content
    print("Has 'auxiliary_v_head' in source:", has_v_head_in_source, flush=True)
    if not has_v_head_in_source:
        print("[FAIL] source file does not contain auxiliary_v_head", flush=True)
        rc = 1
    print(flush=True)

    # ---- 2. Param count: with vs without ----
    print("=== 2. Unet param count: with vs without v-head ===", flush=True)
    from IMF_denoising.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion import Unet

    common = dict(
        problem_dimension=2, init_dim=64, out_dim=1, channels=1,
        conditional_diffusion=True, condition_channels=1,
        downsample_list=(True, True, True, False),
        upsample_list=(True, True, True, False),
        full_attn=(None, None, False, True),
    )
    m_with = Unet(**common, auxiliary_v_head=True)
    m_without = Unet(**common, auxiliary_v_head=False)

    n_with = sum(p.numel() for p in m_with.parameters())
    n_without = sum(p.numel() for p in m_without.parameters())
    diff = n_with - n_without
    print(f"Params with v-head:    {n_with:,}", flush=True)
    print(f"Params without v-head: {n_without:,}", flush=True)
    print(f"v-head extra params:   {diff:,}", flush=True)
    print(f"m_with.auxiliary_v_head:    {m_with.auxiliary_v_head}", flush=True)
    print(f"m_without.auxiliary_v_head: {m_without.auxiliary_v_head}", flush=True)

    if diff <= 0:
        print("[FAIL] v-head should add params; got non-positive diff.", flush=True)
        rc = 1
    elif m_with.auxiliary_v_head is not True:
        print("[FAIL] auxiliary_v_head flag not set on the model.", flush=True)
        rc = 1
    else:
        print("[OK] v-head adds parameters and the flag is set on the model.", flush=True)
    print(flush=True)

    # ---- 3. GPU forward test ----
    print("=== 3. GPU forward test ===", flush=True)
    if not torch.cuda.is_available():
        print("CUDA not available, skipping forward test.", flush=True)
    else:
        device = torch.device("cuda")
        m = m_with.to(device)
        x = torch.randn(2, 1, 256, 256, device=device)
        t = torch.randint(0, 1000, (2,), device=device).float()
        cond = torch.randn(2, 1, 256, 256, device=device)
        try:
            out = m(x, t, cond)
            if isinstance(out, (tuple, list)):
                print(f"Forward returned {type(out).__name__} of length {len(out)}", flush=True)
                for i, o in enumerate(out):
                    print(f"  output[{i}].shape: {tuple(o.shape)}", flush=True)
            else:
                print(f"Forward returned tensor with shape: {tuple(out.shape)}", flush=True)
            print("[OK] Forward pass with v-head succeeded.", flush=True)
        except Exception as e:
            print(f"[FAIL] Forward failed: {type(e).__name__}: {e}", flush=True)
            rc = 1
    print(flush=True)

    # ---- 4. iMF training-loss path ----
    print("=== 4. ImprovedMeanFlow forward (one iMF step) ===", flush=True)
    import IMF_denoising.improved_mean_flow as imf
    diffusion = imf.ImprovedMeanFlow(
        m_with,
        image_size=256,
        ratio_r_neq_t=0.50,
        clip_or_not=False,
        auto_normalize=False,
        adaptive_weight_power=1.0,
        v_loss_weight=0.5,
    )
    if torch.cuda.is_available():
        diffusion = diffusion.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    img = torch.randn(2, 1, 256, 256, device=device)
    cond = torch.randn(2, 1, 256, 256, device=device)
    try:
        out = diffusion(img=img, condition=cond)
        if isinstance(out, (tuple, list)) and len(out) >= 1:
            loss_forward = out[0]
            print(f"loss_forward: {loss_forward.item():.6f}", flush=True)
            print("[OK] iMF forward returned a valid loss with v-head enabled.", flush=True)
        else:
            print(f"[FAIL] Unexpected return type: {type(out)}, value: {out}", flush=True)
            rc = 1
    except Exception as e:
        print(f"[FAIL] iMF forward failed: {type(e).__name__}: {e}", flush=True)
        rc = 1
    print(flush=True)

    # ---- summary ----
    print("=== Summary ===", flush=True)
    if rc == 0:
        print("[ALL OK] v-head is fully wired and active in this training environment.", flush=True)
    else:
        print(f"[FAIL] {rc} checks failed. See above for details.", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
