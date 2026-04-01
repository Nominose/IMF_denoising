"""
improved MeanFlow (iMF) for Noise2Noise CT Denoising — v4.

Changes from v3:
- True auxiliary v-head support (_fn_uv / _fn_u split)
- JVP strictly on u-branch only
- forward returns (loss_forward, V, target_v, x0_pred, img_norm)
- Trainer supports edge_weight and lpips_weight (image-domain losses)
- LPIPS lives in Trainer, not in model (avoids EMA/checkpoint pollution)
"""

import math
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate import Accelerator
from functools import partial
import lpips

from IMF_denoising.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion import (
    Unet,
    exists,
    default,
    identity,
    divisible_by,
    cycle,
    SinusoidalPosEmb,
)
from IMF_denoising.denoising_diffusion_pytorch.denoising_diffusion_pytorch.version import __version__
import IMF_denoising.functions_collection as ff
import IMF_denoising.Data_processing as Data_processing


# =============================================================================
#  TimeMlpWithInterval: proper nn.Module replacement for time_mlp
# =============================================================================

class TimeMlpWithInterval(nn.Module):
    def __init__(self, original_time_mlp, time_dim):
        super().__init__()
        self.original_time_mlp = original_time_mlp

        sinusoidal_dim = max(time_dim // 4, 32)
        self.interval_embed = nn.Sequential(
            SinusoidalPosEmb(sinusoidal_dim),
            nn.Linear(sinusoidal_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self._interval_input = None

    def forward(self, time):
        t_emb = self.original_time_mlp(time)
        if self._interval_input is not None:
            t_emb = t_emb + self.interval_embed(self._interval_input)
        return t_emb


# =============================================================================
#  U-Net wrapper that accepts two time variables (r, t)
# =============================================================================

class UnetWithInterval(nn.Module):
    def __init__(self, base_unet):
        super().__init__()

        self.channels = base_unet.channels
        self.conditional_diffusion = base_unet.conditional_diffusion
        self.problem_dimension = base_unet.problem_dimension
        self.has_v_head = getattr(base_unet, 'auxiliary_v_head', False)

        with torch.no_grad():
            dummy_t = torch.zeros(1)
            try:
                time_dim = base_unet.time_mlp(dummy_t).shape[-1]
            except Exception:
                time_dim = base_unet.init_dim * 4

        wrapped_time_mlp = TimeMlpWithInterval(base_unet.time_mlp, time_dim)
        base_unet.time_mlp = wrapped_time_mlp

        self.base_unet = base_unet

    def forward(self, x, r, t, condition=None):
        interval_input = (t - r) * 999.0
        self.base_unet.time_mlp._interval_input = interval_input

        try:
            t_input = t * 999.0
            if self.conditional_diffusion and condition is not None:
                out = self.base_unet(x, t_input, condition)
            else:
                out = self.base_unet(x, t_input)
        finally:
            self.base_unet.time_mlp._interval_input = None

        return out


# =============================================================================
#  improved MeanFlow (iMF)
# =============================================================================

class ImprovedMeanFlow(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        ratio_r_neq_t=0.5,
        clip_or_not=False,
        clip_range=None,
        auto_normalize=False,
        adaptive_weight_power=1.0,
        v_loss_weight=0.5,
    ):
        super().__init__()

        self.wrapped_model = UnetWithInterval(model)
        self.has_v_head = self.wrapped_model.has_v_head

        self.channels = self.wrapped_model.channels
        self.conditional_diffusion = self.wrapped_model.conditional_diffusion
        self.problem_dimension = self.wrapped_model.problem_dimension
        self.image_size = image_size
        self.ratio_r_neq_t = ratio_r_neq_t
        self.adaptive_weight_power = adaptive_weight_power
        self.v_loss_weight = v_loss_weight

        self.clip_or_not = clip_or_not
        self.clip_range = clip_range or [-1, 1]
        if self.clip_or_not:
            self._maybe_clip = partial(torch.clamp, min=self.clip_range[0], max=self.clip_range[1])
        else:
            self._maybe_clip = identity

        self.normalize = (lambda img: img * 2 - 1) if auto_normalize else identity
        self.unnormalize = (lambda t_val: (t_val + 1) * 0.5) if auto_normalize else identity

        self.num_timesteps = 1000
        self.is_ddim_sampling = False
        self.objective = "pred_average_velocity"

        self._jvp_fallback_count = 0

    @property
    def device(self):
        return next(self.wrapped_model.parameters()).device

    def _sample_t_r(self, batch_size, device):
        eps = 1e-4

        normal_1 = torch.randn(batch_size, device=device) * 1.0 + (-0.4)
        normal_2 = torch.randn(batch_size, device=device) * 1.0 + (-0.4)
        t_raw = torch.sigmoid(normal_1)
        r_raw = torch.sigmoid(normal_2)

        t = torch.max(t_raw, r_raw)
        r = torch.min(t_raw, r_raw)

        t = t.clamp(eps, 1.0 - eps)
        r = torch.clamp(r, min=0.0)
        r = torch.min(r, t - eps)

        mask_fm = torch.rand(batch_size, device=device) > self.ratio_r_neq_t
        r = torch.where(mask_fm, t, r)

        return r, t

    def _fn_uv(self, z, r, t, condition=None):
        """Full forward: returns (u, v) if v-head exists, else (u, None)."""
        if self.conditional_diffusion and not exists(condition):
            raise ValueError("Conditional model but no condition provided.")
        if self.conditional_diffusion:
            out = self.wrapped_model(z, r, t, condition)
        else:
            out = self.wrapped_model(z, r, t)

        if isinstance(out, tuple):
            return out  # (u, v)
        return out, None  # (u, None)

    def _fn_u(self, z, r, t, condition=None):
        """U-only forward: returns u tensor. Used for JVP and sampling."""
        if self.conditional_diffusion and not exists(condition):
            raise ValueError("Conditional model but no condition provided.")
        if self.conditional_diffusion:
            out = self.wrapped_model(z, r, t, condition)
        else:
            out = self.wrapped_model(z, r, t)

        if isinstance(out, tuple):
            return out[0]  # only u
        return out

    def _adaptive_weighted_loss(self, error):
        """Compute per-sample MSE with adaptive weighting."""
        reduce_dims = tuple(range(1, error.ndim))
        loss_raw = (error ** 2).mean(dim=reduce_dims)

        if self.adaptive_weight_power > 0:
            eps_aw = 1e-2
            p = self.adaptive_weight_power
            denom = (loss_raw.detach() + eps_aw) ** p
            return (loss_raw / denom).mean()
        else:
            return loss_raw.mean()

    def forward(self, img, condition=None, *args, **kwargs):
        img_norm = self.normalize(img)
        b = img_norm.shape[0]
        device = img_norm.device

        r, t = self._sample_t_r(b, device)
        e = torch.randn_like(img_norm)

        if img_norm.dim() == 4:
            t_expand = t[:, None, None, None]
            r_expand = r[:, None, None, None]
        else:
            t_expand = t[:, None, None, None, None]
            r_expand = r[:, None, None, None, None]

        z = (1.0 - t_expand) * img_norm + t_expand * e
        target_v = e - img_norm

        # Step 1: Get v from network (u will be recomputed via JVP)
        _, v_pred = self._fn_uv(z, r, t, condition)

        # If no v-head, fall back to boundary condition
        if v_pred is None:
            v_pred = self._fn_u(z, t, t, condition).detach()
            loss_v = torch.tensor(0.0, device=device)
        else:
            # v auxiliary loss
            loss_v = self._adaptive_weighted_loss(v_pred - target_v)

        # Step 2: JVP — strictly on u-branch only
        def fn_for_jvp_u_only(z_in, r_in, t_in):
            return self._fn_u(z_in, r_in, t_in, condition)

        z_jvp = z.detach().requires_grad_(True)
        r_jvp = r.detach().requires_grad_(True)
        t_jvp = t.detach().requires_grad_(True)

        primals = (z_jvp, r_jvp, t_jvp)
        tangents = (v_pred.detach(), torch.zeros_like(r), torch.ones_like(t))

        try:
            u, dudt = torch.func.jvp(fn_for_jvp_u_only, primals, tangents)
        except Exception as ex:
            self._jvp_fallback_count += 1
            if self._jvp_fallback_count <= 5:
                print(f"[WARN] jvp failed (count={self._jvp_fallback_count}): {ex}", flush=True)
            elif self._jvp_fallback_count == 6:
                print("[WARN] jvp keeps failing, suppressing further warnings", flush=True)
            u = self._fn_u(z, r, t, condition)
            eps_fd = 1e-4
            u_plus = self._fn_u(
                z + eps_fd * v_pred.detach(), r, t + eps_fd, condition
            )
            dudt = (u_plus - u) / eps_fd

        # Step 3: Compound function V
        interval = t_expand - r_expand
        V = u + interval * dudt.detach()

        # Step 4: loss_u (v-loss with adaptive weighting)
        loss_u = self._adaptive_weighted_loss(V - target_v)

        # Step 5: Reconstruct x0_pred for image-domain losses (computed in Trainer)
        x0_pred = z - t_expand * V

        # Step 6: Total forward loss (edge and LPIPS are added by Trainer)
        loss_forward = loss_u + self.v_loss_weight * loss_v

        return loss_forward, V, target_v, x0_pred, img_norm

    @torch.inference_mode()
    def sample(self, condition=None, batch_size=16):
        device = self.device

        if self.problem_dimension == "2D":
            shape = (batch_size, self.channels, self.image_size[0], self.image_size[1])
        else:
            shape = (batch_size, self.channels, *self.image_size)

        z1 = torch.randn(shape, device=device)

        r = torch.zeros(batch_size, device=device)
        t = torch.ones(batch_size, device=device)

        u = self._fn_u(z1, r, t, condition)

        z0 = z1 - u
        z0 = self._maybe_clip(z0)
        z0 = self.unnormalize(z0)

        return z0

    @torch.inference_mode()
    def sample_multistep(self, condition=None, batch_size=16, num_steps=2, solver='euler'):
        """
        Multi-step sampling from t=1 (noise) to t=0 (data).
        solver: 'euler' (1st order), 'midpoint' (2nd order), 'heun' (2nd order)
        Note: midpoint and heun use 2x function evaluations per step.
        """
        device = self.device

        if self.problem_dimension == "2D":
            shape = (batch_size, self.channels, self.image_size[0], self.image_size[1])
        else:
            shape = (batch_size, self.channels, *self.image_size)

        z = torch.randn(shape, device=device)

        ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        for i in range(num_steps):
            t_val = ts[i]
            r_val = ts[i + 1]
            dt = t_val - r_val
            t_batch = torch.full((batch_size,), t_val, device=device)
            r_batch = torch.full((batch_size,), r_val, device=device)

            if solver == 'euler':
                u = self._fn_u(z, r_batch, t_batch, condition)
                z = z - dt * u

            elif solver == 'midpoint':
                # Step 1: half step with Euler
                u1 = self._fn_u(z, r_batch, t_batch, condition)
                z_mid = z - (dt / 2) * u1
                # Step 2: full step with midpoint velocity
                t_mid = torch.full((batch_size,), (t_val + r_val) / 2, device=device)
                r_mid = r_batch
                u2 = self._fn_u(z_mid, r_mid, t_mid, condition)
                z = z - dt * u2

            elif solver == 'heun':
                # Step 1: predict with Euler
                u1 = self._fn_u(z, r_batch, t_batch, condition)
                z_pred = z - dt * u1
                # Step 2: correct with average velocity
                u2 = self._fn_u(z_pred, r_batch, r_batch, condition)
                z = z - dt * (u1 + u2) / 2

            else:
                raise ValueError(f"Unknown solver: {solver}. Use 'euler', 'midpoint', or 'heun'.")

        z = self._maybe_clip(z)
        z = self.unnormalize(z)
        return z


# =============================================================================
#  Trainer
# =============================================================================

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        generator_train,
        generator_val,
        train_batch_size,
        *,
        accum_iter=1,
        train_num_steps=100000,
        results_folder=None,
        train_lr=1e-4,
        train_lr_decay_every=200,
        save_models_every=1,
        validation_every=200,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.95),
        amp=False,
        mixed_precision_type="fp16",
        split_batches=True,
        max_grad_norm=1.0,
        lpips_weight=0.0,
        edge_weight=0.0,
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else "no",
        )

        self.model = diffusion_model
        self.conditional_diffusion = self.model.conditional_diffusion
        self.channels = diffusion_model.channels

        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps
        self.accum_iter = accum_iter

        self.ds = generator_train
        dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=False, pin_memory=True, num_workers=0)
        self.dl = self.accelerator.prepare(dl)

        self.ds_val = generator_val
        dl_val = DataLoader(self.ds_val, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)
        self.dl_val = self.accelerator.prepare(dl_val)

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)
        self.scheduler = StepLR(self.opt, step_size=1, gamma=0.95)
        self.max_grad_norm = max_grad_norm
        self.train_lr_decay_every = train_lr_decay_every
        self.save_model_every = save_models_every

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = results_folder
        ff.make_folder([self.results_folder])

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self.validation_every = validation_every

        # LPIPS in Trainer (not in model) — avoids EMA/checkpoint pollution
        self.lpips_weight = lpips_weight
        self.edge_weight = edge_weight
        if lpips_weight > 0:
            self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
            self.lpips_fn.eval()
            for p in self.lpips_fn.parameters():
                p.requires_grad = False
        else:
            self.lpips_fn = None

    @property
    def device(self):
        return self.accelerator.device

    def save(self, step_num):
        if not self.accelerator.is_main_process:
            return
        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "decay_steps": self.scheduler.state_dict(),
            "scaler": self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            "version": __version__,
        }
        torch.save(data, os.path.join(self.results_folder, f"model-{step_num}.pt"))

    def load_model(self, path):
        device = self.accelerator.device
        data = torch.load(path, map_location=device)
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])
        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])
        self.scheduler.load_state_dict(data["decay_steps"])
        if exists(self.accelerator.scaler) and exists(data.get("scaler")):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    def _compute_lpips(self, x0_pred, img_norm):
        """Compute LPIPS loss in float32, outside autocast. Uses clamped inputs."""
        with torch.cuda.amp.autocast(enabled=False):
            pred_3ch = x0_pred.float().clamp(-1, 1).repeat(1, 3, 1, 1) if x0_pred.shape[1] == 1 else x0_pred.float().clamp(-1, 1)
            target_3ch = img_norm.float().clamp(-1, 1).repeat(1, 3, 1, 1) if img_norm.shape[1] == 1 else img_norm.float().clamp(-1, 1)
            return self.lpips_fn(pred_3ch, target_3ch).mean()

    def _compute_edge_loss(self, pred, target):
        """Sobel-based edge loss in image domain. Uses unclamped inputs."""
        if pred.dim() == 4:
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
            edge_pred = torch.sqrt(F.conv2d(pred, sobel_x, padding=1) ** 2 + F.conv2d(pred, sobel_y, padding=1) ** 2 + 1e-6)
            edge_target = torch.sqrt(F.conv2d(target, sobel_x, padding=1) ** 2 + F.conv2d(target, sobel_y, padding=1) ** 2 + 1e-6)
        else:
            edge_pred = pred
            edge_target = target
        return F.mse_loss(edge_pred, edge_target)

    def train(self, pre_trained_model=None, start_step=None):
        accelerator = self.accelerator
        device = accelerator.device

        if pre_trained_model is not None:
            self.load_model(pre_trained_model)
            print("model loaded from", pre_trained_model)

        if start_step is not None:
            self.step = start_step
        elif pre_trained_model is None:
            self.step = 0

        training_log = []
        val_loss = float("inf")
        val_total = float("inf")
        val_lpips = 0.0
        val_edge = 0.0

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                print(f"training epoch: {self.step + 1}", flush=True)
                print(f"learning rate: {self.scheduler.get_last_lr()[0]}", flush=True)

                avg_loss = []
                avg_loss_lpips = []
                self.opt.zero_grad()
                torch.cuda.empty_cache()

                for count, batch in enumerate(self.dl):
                    batch_x0, batch_cond = batch
                    data_x0 = batch_x0.to(device)
                    data_cond = batch_cond.to(device) if self.conditional_diffusion else None

                    with self.accelerator.autocast():
                        loss_forward, V, target_v, x0_pred, img_norm = self.model(img=data_x0, condition=data_cond)

                    # Image-domain losses (computed in Trainer, outside autocast)
                    aux_loss = torch.tensor(0.0, device=device)

                    if self.edge_weight > 0:
                        edge_loss = self._compute_edge_loss(x0_pred, img_norm)
                        aux_loss = aux_loss + self.edge_weight * edge_loss

                    if self.lpips_weight > 0 and self.lpips_fn is not None:
                        lpips_loss = self._compute_lpips(x0_pred, img_norm)
                        aux_loss = aux_loss + self.lpips_weight * lpips_loss
                    else:
                        lpips_loss = torch.tensor(0.0, device=device)

                    total_loss = (loss_forward + aux_loss) / self.accum_iter

                    self.accelerator.backward(total_loss)

                    if ((count + 1) % self.accum_iter == 0) or (count == len(self.dl) - 1):
                        accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.opt.step()
                        self.opt.zero_grad()

                    if count % 200 == 0:
                        print(f"  epoch {self.step+1} | batch {count} | loss {total_loss.item():.4f} | lpips {lpips_loss.item():.4f}", flush=True)

                    avg_loss.append(total_loss.item() * self.accum_iter)
                    avg_loss_lpips.append(lpips_loss.item())

                avg_loss = sum(avg_loss) / len(avg_loss)
                avg_lpips = sum(avg_loss_lpips) / len(avg_loss_lpips)
                pbar.set_description(f"loss:{avg_loss:.4f}")

                accelerator.wait_for_everyone()
                self.step += 1

                if self.step != 0 and divisible_by(self.step, self.save_model_every):
                    self.save(self.step)
                if self.step != 0 and divisible_by(self.step, self.train_lr_decay_every):
                    self.scheduler.step()

                if self.accelerator.is_main_process:
                    self.ema.update()

                if self.step != 0 and divisible_by(self.step, self.validation_every):
                    print(f"validation at step {self.step}")
                    self.model.eval()
                    with torch.no_grad():
                        vl = []
                        vl_lpips = []
                        vl_edge = []
                        for vbatch in self.dl_val:
                            vx0, vcond = vbatch
                            vx0 = vx0.to(device)
                            vcond = vcond.to(device) if self.conditional_diffusion else None
                            with self.accelerator.autocast():
                                vloss, _, _, vx0_pred, vimg_norm = self.model(img=vx0, condition=vcond)
                            vl.append(vloss.item())
                            if self.edge_weight > 0:
                                vl_edge.append(self._compute_edge_loss(vx0_pred, vimg_norm).item())
                            if self.lpips_weight > 0 and self.lpips_fn is not None:
                                vl_lpips.append(self._compute_lpips(vx0_pred, vimg_norm).item())
                        val_loss = sum(vl) / len(vl)
                        val_lpips = sum(vl_lpips) / len(vl_lpips) if vl_lpips else 0.0
                        val_edge = sum(vl_edge) / len(vl_edge) if vl_edge else 0.0
                        val_total = val_loss + self.edge_weight * val_edge + self.lpips_weight * val_lpips
                        print(f"  val_total={val_total:.4f} | val_forward={val_loss:.4f} | val_lpips={val_lpips:.4f} | val_edge={val_edge:.4f}")
                    self.model.train(True)

                training_log.append([
                    self.step, self.scheduler.get_last_lr()[0], avg_loss, val_loss, val_total, avg_lpips, val_lpips, val_edge,
                ])
                df = pd.DataFrame(training_log, columns=[
                    "iteration", "lr", "train_loss", "val_forward", "val_total", "train_lpips", "val_lpips", "val_edge",
                ])
                log_dir = os.path.join(os.path.dirname(self.results_folder), "log")
                ff.make_folder([log_dir])
                df.to_excel(os.path.join(log_dir, "training_log.xlsx"), index=False)

                self.ds.on_epoch_end()
                self.ds_val.on_epoch_end()
                pbar.update(1)

        accelerator.print("training complete")


# =============================================================================
#  Sampler
# =============================================================================

class Sampler(object):
    def __init__(self, diffusion_model, generator, batch_size, device="cuda"):
        super().__init__()

        self.model = diffusion_model
        self.device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")

        self.conditional_diffusion = self.model.conditional_diffusion
        self.channels = diffusion_model.channels
        self.image_size = diffusion_model.image_size
        self.batch_size = batch_size

        assert batch_size == 1, "sample_2D currently requires batch_size=1"

        self.generator = generator

        self.histogram_equalization = getattr(generator, "histogram_equalization", False)
        self.bins = getattr(generator, "bins", None)
        self.bins_mapped = getattr(generator, "bins_mapped", None)
        self.background_cutoff = getattr(generator, "background_cutoff", None)
        self.maximum_cutoff = getattr(generator, "maximum_cutoff", None)
        self.normalize_factor = getattr(generator, "normalize_factor", None)

        self.ema = EMA(diffusion_model)
        self.ema.to(self.device)
        self.model.to(self.device)

    def load_model(self, path):
        data = torch.load(path, map_location=self.device)
        self.model.load_state_dict(data["model"])
        self.step = data["step"]
        self.ema.load_state_dict(data["ema"])

    def sample_2D(
        self,
        trained_model_filename,
        condition_img,
        direct_use_of_model=False,
        need_change_dim=True,
        need_denormalize=True,
        num_steps=1,
        solver='euler',
    ):
        bg = self.background_cutoff
        mx = self.maximum_cutoff
        nf = self.normalize_factor

        if not direct_use_of_model:
            self.load_model(trained_model_filename)

        device = self.device

        if direct_use_of_model:
            sampling_model = self.model
        else:
            sampling_model = self.ema.ema_model

        sampling_model.eval()
        print("model device:", next(sampling_model.parameters()).device)

        pred = np.zeros((self.image_size[0], self.image_size[1], condition_img.shape[-1]), dtype=np.float32)

        with torch.inference_mode():
            for z in tqdm(range(condition_img.shape[-1]), desc="sampling", leave=False):

                if self.conditional_diffusion:
                    datas = self.generator[z]
                    cond = datas[1]
                    if isinstance(cond, np.ndarray):
                        cond = torch.from_numpy(cond).float()
                    if cond.dim() == 2:
                        cond = cond.unsqueeze(0).unsqueeze(0)
                    elif cond.dim() == 3:
                        cond = cond.unsqueeze(0)
                    data_cond = cond.to(device)
                else:
                    data_cond = None

                if num_steps == 1:
                    out = sampling_model.sample(condition=data_cond, batch_size=self.batch_size)
                else:
                    out = sampling_model.sample_multistep(condition=data_cond, batch_size=self.batch_size, num_steps=num_steps, solver=solver)
                pred[:, :, z] = out[0, 0].detach().cpu().numpy()

        if need_change_dim:
            pred = Data_processing.crop_or_pad(
                pred,
                [condition_img.shape[0], condition_img.shape[1], condition_img.shape[-1]],
                value=np.min(condition_img),
            )
        if need_denormalize:
            pred = Data_processing.normalize_image(pred, normalize_factor=nf, image_max=mx, image_min=bg, invert=True)
        if self.histogram_equalization:
            pred = Data_processing.apply_transfer_to_img(pred, self.bins, self.bins_mapped, reverse=True)
        if need_change_dim:
            pred = Data_processing.correct_shift_caused_in_pad_crop_loop(pred)

        return pred
