"""
imf_gan.py — add a lightweight adversarial loss to the iMF generator (TRAINING ONLY).

    L_total(G) = L_flow + beta * L_adv
    D = conditional PatchGAN, HINGE loss + R1 gradient penalty (R1 is critical for
        stability above ~128px; applied lazily every r1_every steps, StyleGAN2-style).

Design (one-step GENERATION adversarial — option B):
  * adversarial signal is on the TRUE one-step generation from PURE noise (= the NFE=1 inference
    output): z ~ N(0,I);  x0_gen = z - u(z, r=0, t=1, c).  D pushes this cheapest-inference output
    toward the real x2 distribution. This targets the actual under-dispersed object -- NO small-t
    data leakage and NO random-t reconstruction roughness (the t*delta junk that contaminated the
    earlier x0_pred = z - t*V variant; x0_pred at large t is rough because the one-step error =
    t * velocity_error, amplified). One differentiable Euler step (backprops through ONE forward
    into G); one extra forward vs reusing x0_pred. NOTE: adv_nfe is UNUSED (kept for arg compat).
  * recommended workflow: FLOW-pretrain (your existing model-200) -> GAN fine-tune with small
    lr_g and small beta (two-stage, more stable than from-scratch GAN).

What it does, depending on the generator's training target:
  * current NOISY-target model -> pushes one-step samples to match the real NOISY distribution
    (reduces few-step bias; closes the NFE=3 -> NFE=5 gap; ceiling = posterior mean).
  * clean-target (oracle/ambient) model -> pushes toward the clean manifold (sharper / better LPIPS).

INFERENCE IS UNCHANGED: the discriminator is discarded; you still use the same few-step
MeanFlow sampling with K flexible. Reuses ImprovedMeanFlow as the generator; touches nothing.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import spectral_norm
from ema_pytorch import EMA
from tqdm.auto import tqdm

import IMF_denoising.functions_collection as ff


# =============================================================================
#  Conditional PatchGAN discriminator (lightweight, spectral-norm)
# =============================================================================
class PatchDiscriminator(nn.Module):
    """Input = concat([image (img_channels), condition (cond_channels)]) -> patch logits.

    NO-NORM, SINGLE-REGULARIZER (StyleGAN-D style): plain conv + LeakyReLU, Lipschitz controlled
    by R1 ONLY. Two things were crushing this D before, pinning hinge at 2.0 (D ~const for
    real & fake):
      (1) spectral_norm on every layer ON TOP of R1 -> over-constrained, removed.
      (2) InstanceNorm -> normalizes per-instance variance, i.e. it WASHES OUT the very signal
          D needs here (real x2 = full noise / high variance; few-step fake = under-dispersed
          / lower variance). Removed.
    NOTE: R1 is now the ONLY regularizer -> keep r1_gamma > 0 (do NOT set it to 0).
    """

    def __init__(self, img_channels=1, cond_channels=2, base=64, n_layers=3, high_pass=True, hp_kernel=7):
        super().__init__()
        self.cond_channels = cond_channels   # cond_channels=0 -> UNCONDITIONAL D (ignores cond)
        self.high_pass = high_pass           # high-pass front-end: feed D (img - blur) so the noise
        self.hp_kernel = hp_kernel           # (high-freq = the real-vs-fake signal) is not low-passed
        ch_in = img_channels + cond_channels
        seq = [nn.Conv2d(ch_in, base, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        ch = base
        for i in range(1, n_layers):
            nch = min(base * (2 ** i), 512)
            seq += [nn.Conv2d(ch, nch, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            ch = nch
        nch = min(base * (2 ** n_layers), 512)
        seq += [nn.Conv2d(ch, nch, 4, 1, 1), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(nch, 1, 4, 1, 1)]
        self.net = nn.Sequential(*seq)

    def forward(self, img, cond=None):
        if self.high_pass:
            # HIGH-PASS FRONT-END: subtract a blurred copy -> D sees the NOISE RESIDUAL directly.
            # noisy real -> large residual; smooth fake -> small residual. This hands D the high-freq
            # signal up front, instead of relying on it to claw the noise back out after the
            # stride-2 downsampling has already low-passed it away.
            k = self.hp_kernel
            img = img - F.avg_pool2d(img, kernel_size=k, stride=1, padding=k // 2)
        # cond_channels==0 => unconditional: ignore any cond passed in (the noise-vs-smooth signal
        # lives in the image alone, so dropping cond removes its dilution of that subtle signal).
        x = img if (cond is None or self.cond_channels == 0) else torch.cat([img, cond], dim=1)
        return self.net(x)


# ---- hinge losses + R1 ----
def d_hinge(d_real, d_fake):
    return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()


def g_hinge(d_fake):
    return -d_fake.mean()


def r1_penalty(d_real, real_in):
    grad, = torch.autograd.grad(outputs=d_real.sum(), inputs=real_in, create_graph=True)
    return grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()


def _set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


# =============================================================================
#  GAN trainer (plain single-GPU PyTorch; reuses ImprovedMeanFlow as generator)
# =============================================================================
class GANTrainer:
    def __init__(self, diffusion_model, discriminator, generator_train, *,
                 train_batch_size=2, train_num_steps=100, results_folder=None,
                 lr_g=1e-5, lr_d=2e-4, adv_weight=0.1, r1_gamma=1.0, r1_every=16,
                 adv_nfe=3,
                 adv_start_step=0, ema_decay=0.999, save_every=5, max_grad_norm=1.0,
                 device='cuda'):
        self.device = torch.device('cuda' if (device == 'cuda' and torch.cuda.is_available()) else 'cpu')
        self.G = diffusion_model.to(self.device)
        self.D = discriminator.to(self.device)
        self.conditional = self.G.conditional_diffusion

        self.ds = generator_train
        self.dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
        self.opt_g = Adam(self.G.parameters(), lr=lr_g, betas=(0.0, 0.99))
        self.opt_d = Adam(self.D.parameters(), lr=lr_d, betas=(0.0, 0.99))
        self.ema = EMA(self.G, beta=ema_decay, update_every=10); self.ema.to(self.device)

        self.adv_weight = adv_weight
        self.r1_gamma = r1_gamma
        self.r1_every = max(1, int(r1_every))
        self.adv_nfe = max(1, int(adv_nfe))
        self.adv_start = adv_start_step
        self.train_num_steps = train_num_steps
        self.results_folder = results_folder
        self.save_every = save_every
        self.max_grad_norm = max_grad_norm
        if results_folder is None:
            raise ValueError("GANTrainer needs results_folder (where checkpoints / gan_log.xlsx are written).")
        ff.make_folder([results_folder])
        self.step = 0

    def load_generator(self, path, key='model'):
        data = torch.load(path, map_location=self.device)
        self.G.load_state_dict(data[key])
        print(f'[GAN] loaded pretrained generator ({key}) from {path}', flush=True)

    def save(self, tag):
        torch.save({'step': self.step,
                    'model': self.G.state_dict(), 'ema': self.ema.state_dict(),
                    'D': self.D.state_dict(),
                    'opt_g': self.opt_g.state_dict(), 'opt_d': self.opt_d.state_dict()},
                   os.path.join(self.results_folder, f'model-{tag}.pt'))

    @torch.no_grad()
    def _sanity_probe(self):
        """One-time check: is there ANY real-vs-fake signal for D to learn?
        The decisive number is |real-fake| / |real|. If it is tiny, the few-step fake ~= the
        real x2, so D has nothing to discriminate -> GAN-on-x2 is a dead end (no amount of D
        tuning fixes 'nothing to separate')."""
        x0, cond = next(iter(self.dl))
        x0 = x0.to(self.device)
        cond = cond.to(self.device) if self.conditional else None
        img_norm = self.G(img=x0, condition=cond)[-1]
        b = img_norm.shape[0]
        z = torch.randn_like(img_norm)
        t1 = torch.full((b,), 1.0, device=self.device)
        r0 = torch.full((b,), 0.0, device=self.device)
        fake = z - self.G._fn_u(z, r0, t1, cond)   # one-step NFE=1 generation = exactly what D sees as "fake"
        noise = torch.randn_like(img_norm)
        diff = (img_norm - fake).abs().mean().item()
        scale = img_norm.abs().mean().item()
        hf = lambda x: float((x[:, :, 1:, :] - x[:, :, :-1, :]).std())  # high-freq (noise) level proxy
        dr = self.D(img_norm, cond).mean().item()
        df = self.D(fake, cond).mean().item()
        dn = self.D(noise, cond).mean().item()
        # save one real & one fake patch so YOU can LOOK (the human eye is the judge here):
        np.save(os.path.join(self.results_folder, 'probe_real.npy'), img_norm[0, 0].detach().cpu().numpy())
        np.save(os.path.join(self.results_folder, 'probe_fake.npy'), fake[0, 0].detach().cpu().numpy())
        print(f'[probe] |real-fake|={diff:.4f} vs |real|={scale:.4f} (ratio {diff / max(scale, 1e-8):.1%})', flush=True)
        print(f'[probe] high-freq(noise) std:  real={hf(img_norm):.4f}  fake={hf(fake):.4f}  pure_noise={hf(noise):.4f}', flush=True)
        print('[probe]   -> fake << real  => fake is SMOOTHER => distinguishable => D *should* learn (bug if it cannot)', flush=True)
        print('[probe]   -> fake ~= real  => same noise level => genuinely no signal', flush=True)
        print(f'[probe] D(real)={dr:+.3f} D(fake)={df:+.3f} D(pure_noise)={dn:+.3f}  (random-init D, just a baseline)', flush=True)
        print(f'[probe] saved probe_real.npy / probe_fake.npy under {self.results_folder} -- open them and LOOK', flush=True)

    @torch.no_grad()
    def _setup_fv_probe(self):
        """Fix ONE input + ONE noise tensor so the per-epoch F(v) dump shows the SAME slice under the
        SAME noise every epoch -> you watch the model's one-step output evolve (toward x2), not anatomy
        churn. Saves the real x2 once as the reference to compare against."""
        x0, cond = next(iter(self.dl))
        self._fv_x0 = x0.to(self.device)
        self._fv_cond = cond.to(self.device) if self.conditional else None
        self._fv_real = self.G(img=self._fv_x0, condition=self._fv_cond)[-1]   # img_norm = the x2 target
        self._fv_z = torch.randn_like(self._fv_real)                            # FIXED noise (same every epoch)
        self._fv_dir = os.path.join(self.results_folder, 'fv_evolution')
        os.makedirs(self._fv_dir, exist_ok=True)
        np.save(os.path.join(self._fv_dir, 'real_x2.npy'), self._fv_real[0, 0].detach().cpu().numpy())
        print(f'[fv] F(v) evolution -> {self._fv_dir}  (real_x2.npy = target; fv_epoch*.npy = one-step output each epoch)', flush=True)

    @torch.no_grad()
    def _dump_fv(self, epoch):
        """Dump the one-step F(v) = z - u(z, r=0, t=1, c) for the FIXED probe -> fv_epoch{epoch}.npy.
        Same form as NFE=1 inference (t-r=1), but from the fixed noise so epochs are directly comparable."""
        b = self._fv_z.shape[0]
        t1 = torch.full((b,), 1.0, device=self.device)
        r0 = torch.full((b,), 0.0, device=self.device)
        u = self.G._fn_u(self._fv_z, r0, t1, self._fv_cond)
        x0 = self._fv_z - u                                                     # one-step generation
        np.save(os.path.join(self._fv_dir, f'fv_epoch{epoch}.npy'), x0[0, 0].detach().cpu().numpy())

    def train(self):
        self._sanity_probe()
        self._setup_fv_probe()
        log = []
        for epoch in range(self.train_num_steps):
            self.G.train(); self.D.train()
            flow_l, gadv_l, d_l, dr_l, df_l, dh_l = [], [], [], [], [], []
            for batch in tqdm(self.dl, desc=f'epoch {epoch + 1}', leave=False):
                x0, cond = batch
                x0 = x0.to(self.device)
                cond = cond.to(self.device) if self.conditional else None
                use_adv = self.step >= self.adv_start

                # ---------------- Generator: flow loss (+ adversarial) ----------------
                _set_requires_grad(self.D, False)
                loss_forward, V, target_v, x0_pred, img_norm = self.G(img=x0, condition=cond)
                if use_adv:
                    # OPTION B: adversarial on the TRUE one-step generation from PURE noise (= the
                    # NFE=1 inference output). z ~ N(0,I); one differentiable Euler step t=1 -> r=0:
                    #     x0_gen = z - u(z, r=0, t=1, c)
                    # Targets the actual cheapest-inference object -- NO small-t data leakage and NO
                    # random-t reconstruction roughness (one-step error = t * velocity_error, which is
                    # high-freq and t-amplified -> the t*delta junk that made x0_pred rough at large t).
                    # NOT inference_mode, so the adversarial grad flows through u into G. One extra
                    # forward vs reusing x0_pred, but on the right object. (= old rollout with adv_nfe=1.)
                    b = x0.shape[0]
                    z = torch.randn_like(img_norm)
                    t1 = torch.full((b,), 1.0, device=self.device)
                    r0 = torch.full((b,), 0.0, device=self.device)
                    x0_gen = z - self.G._fn_u(z, r0, t1, cond)
                    fake_detached = x0_gen.detach()
                    g_adv = g_hinge(self.D(x0_gen, cond))
                    g_loss = loss_forward + self.adv_weight * g_adv
                else:
                    fake_detached = x0_pred.detach()
                    g_adv = torch.zeros((), device=self.device)
                    g_loss = loss_forward
                self.opt_g.zero_grad(set_to_none=True)
                g_loss.backward()
                nn.utils.clip_grad_norm_(self.G.parameters(), self.max_grad_norm)
                self.opt_g.step()
                self.ema.update()

                # ---------------- Discriminator: hinge (+ lazy R1) ----------------
                d_val = 0.0
                if use_adv:
                    _set_requires_grad(self.D, True)
                    do_r1 = (self.step % self.r1_every == 0)
                    real = img_norm.detach()
                    if do_r1:
                        real = real.requires_grad_(True)
                    d_real = self.D(real, cond)
                    d_fake = self.D(fake_detached, cond)
                    d_loss = d_hinge(d_real, d_fake)
                    d_hinge_val = float(d_loss.item())   # hinge ONLY (the real real-vs-fake signal), BEFORE R1 is added
                    if do_r1:
                        # canonical R1 is (gamma/2)*E||grad||^2; the 0.5 was missing -> R1 was 2x too
                        # strong. r1_every is the lazy-regularization correction (StyleGAN2).
                        d_loss = d_loss + (0.5 * self.r1_gamma * self.r1_every) * r1_penalty(d_real, real)
                    self.opt_d.zero_grad(set_to_none=True)
                    d_loss.backward()
                    self.opt_d.step()
                    d_val = float(d_loss.item())
                    dr_l.append(float(d_real.mean().item()))
                    df_l.append(float(d_fake.mean().item()))
                    dh_l.append(d_hinge_val)

                flow_l.append(float(loss_forward.item()))
                gadv_l.append(float(g_adv.item()))
                d_l.append(d_val)
                self.step += 1

            self.ds.on_epoch_end()
            _dr = np.mean(dr_l) if dr_l else 0.0
            _df = np.mean(df_l) if df_l else 0.0
            _dh = np.mean(dh_l) if dh_l else 0.0
            msg = (f'epoch {epoch + 1}: flow {np.mean(flow_l):.4f} | g_adv {np.mean(gadv_l):.4f} '
                   f'| d_hinge {_dh:.4f} (d+R1 {np.mean(d_l):.4f}) | d_real {_dr:+.3f} d_fake {_df:+.3f} | adv {"on" if use_adv else "off"}')
            print(msg, flush=True)
            log.append([epoch + 1, np.mean(flow_l), np.mean(gadv_l), _dh, np.mean(d_l), _dr, _df])
            pd.DataFrame(log, columns=['epoch', 'flow', 'g_adv', 'd_hinge', 'd_total', 'd_real', 'd_fake']).to_excel(
                os.path.join(self.results_folder, 'gan_log.xlsx'), index=False)
            self._dump_fv(epoch + 1)   # save this epoch's one-step F(v) for the fixed probe (watch it evolve)
            if (epoch + 1) % self.save_every == 0:
                self.save(epoch + 1)
        print('[GAN] training complete', flush=True)
