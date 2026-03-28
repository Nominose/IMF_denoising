"""
Conditional Flow Matching — final version.

All review fixes applied:
  v1: float t, grad accum, x_t/t_expand return, LPIPS on x0_pred,
      EMA guard, final-only clip, batch_size==1 assert
  v2: condition_img directly used via generator[z] in sample_2D,
      auto_normalize-safe auxiliary losses (compare in normalized space),
      GT clamped for LPIPS, direct_use_of_model properly branches,
      3D LPIPS guard, auxiliary loss refactored into helper method
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
import nibabel as nb

from Diffusion_denoising_thin_slice.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion import (
    Unet,
    exists,
    default,
    identity,
    divisible_by,
    cycle,
)
from Diffusion_denoising_thin_slice.denoising_diffusion_pytorch.denoising_diffusion_pytorch.version import __version__
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Data_processing as Data_processing
import Diffusion_denoising_thin_slice.denoising_diffusion_pytorch.denoising_diffusion_pytorch.edge_loss as edge_loss_fn
import Diffusion_denoising_thin_slice.denoising_diffusion_pytorch.denoising_diffusion_pytorch.kernel as kernel


class ConditionalFlowMatching(nn.Module):
    """
    Optimal-transport conditional flow matching (OT-CFM).

    Forward path:   x_t = (1 - t) * x_0  +  t * z,   z ~ N(0,I),  t ~ U(0,1)
    Target:         v = z - x_0
    Recover x0:     x0_pred = x_t - t * v_pred

    forward() returns: (loss, model_out, target, x_t, t_expand, img_normalized)
      - img_normalized: x0 after normalize(), so Trainer can compute auxiliary
        losses in the same space as x_t / x0_pred.
    """

    def __init__(
        self,
        model,
        *,
        image_size,
        sampling_timesteps=50,
        clip_or_not=False,
        clip_range=None,
        auto_normalize=False,
    ):
        super().__init__()

        self.model = model
        self.channels = model.channels
        self.conditional_diffusion = model.conditional_diffusion
        self.self_condition = getattr(model, "self_condition", False)
        self.problem_dimension = model.problem_dimension

        self.image_size = image_size
        self.sampling_timesteps = sampling_timesteps

        self.clip_or_not = clip_or_not
        self.clip_range = clip_range or [-1, 1]
        if self.clip_or_not:
            self._maybe_clip = partial(torch.clamp, min=self.clip_range[0], max=self.clip_range[1])
        else:
            self._maybe_clip = identity

        self.normalize = (lambda img: img * 2 - 1) if auto_normalize else identity
        self.unnormalize = (lambda t: (t + 1) * 0.5) if auto_normalize else identity

        # placeholders for API compat
        self.num_timesteps = 1000
        self.is_ddim_sampling = False
        self.objective = "pred_velocity"

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(self, img, condition=None, *args, **kwargs):
        """
        Returns: (loss, model_out, target, x_t, t_expand, img_normalized)
        """
        img_norm = self.normalize(img)
        b = img_norm.shape[0]

        t = torch.rand(b, device=img_norm.device, dtype=img_norm.dtype)
        z = torch.randn_like(img_norm)

        if img_norm.dim() == 4:
            t_expand = t[:, None, None, None]
        else:
            t_expand = t[:, None, None, None, None]

        x_t = (1.0 - t_expand) * img_norm + t_expand * z
        target = z - img_norm

        t_input = t * 999.0  # float, for sinusoidal embedding

        if self.conditional_diffusion:
            if not exists(condition):
                raise ValueError("Conditional model but no condition provided.")
            model_out = self.model(x_t, t_input, condition)
        else:
            model_out = self.model(x_t, t_input)

        loss = F.mse_loss(model_out, target)

        return loss, model_out, target, x_t, t_expand, img_norm

    @torch.inference_mode()
    def sample(self, condition=None, batch_size=16):
        """Euler ODE solver from t=1 (noise) to t=0 (data)."""
        device = self.device
        N = self.sampling_timesteps
        dt = -1.0 / N

        if self.problem_dimension == "2D":
            shape = (batch_size, self.channels, self.image_size[0], self.image_size[1])
        else:
            shape = (batch_size, self.channels, *self.image_size)

        x = torch.randn(shape, device=device)
        ts = torch.linspace(1.0, 1.0 / N, N, device=device)

        for i, t_val in enumerate(tqdm(ts, desc="FM sampling", leave=False)):
            t_batch = torch.full((batch_size,), t_val, device=device, dtype=torch.float32)
            t_input = t_batch * 999.0

            if self.conditional_diffusion:
                v = self.model(x, t_input, condition)
            else:
                v = self.model(x, t_input)

            x = x + dt * v
            if i == len(ts) - 1:
                x = self._maybe_clip(x)

        x = self.unnormalize(x)
        return x

    @torch.inference_mode()
    def sample_midpoint(self, condition=None, batch_size=16):
        """Midpoint (2nd-order) ODE solver."""
        device = self.device
        N = self.sampling_timesteps
        dt = -1.0 / N

        if self.problem_dimension == "2D":
            shape = (batch_size, self.channels, self.image_size[0], self.image_size[1])
        else:
            shape = (batch_size, self.channels, *self.image_size)

        x = torch.randn(shape, device=device)
        ts = torch.linspace(1.0, 1.0 / N, N, device=device)

        for i, t_val in enumerate(tqdm(ts, desc="FM midpoint", leave=False)):
            t_batch = torch.full((batch_size,), t_val, device=device, dtype=torch.float32)
            t_input = t_batch * 999.0

            if self.conditional_diffusion:
                v1 = self.model(x, t_input, condition)
            else:
                v1 = self.model(x, t_input)

            x_mid = x + 0.5 * dt * v1
            t_mid_val = max(t_val.item() + 0.5 * dt, 0.0)
            t_mid_batch = torch.full((batch_size,), t_mid_val, device=device, dtype=torch.float32)
            t_mid_input = t_mid_batch * 999.0

            if self.conditional_diffusion:
                v2 = self.model(x_mid, t_mid_input, condition)
            else:
                v2 = self.model(x_mid, t_mid_input)

            x = x + dt * v2
            if i == len(ts) - 1:
                x = self._maybe_clip(x)

        x = self.unnormalize(x)
        return x


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
        validation_every=1,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        amp=False,
        mixed_precision_type="fp16",
        split_batches=True,
        max_grad_norm=1.0,
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
        dl_val = DataLoader(self.ds_val, batch_size=train_batch_size, shuffle=False, pin_memory=True, num_workers=0)
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

        self.lpips_loss_fn = lpips.LPIPS(net="vgg").to(self.device)
        self.edge_loss_fn = edge_loss_fn.edge_loss_fn

    @property
    def device(self):
        return self.accelerator.device

    def save(self, step_num):
        if not self.accelerator.is_local_main_process:
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

    def _compute_auxiliary_losses(self, x0_pred, x0_gt, beta, lpips_weight, edge_weight, device):
        """
        Compute bias / LPIPS / edge losses.
        Both x0_pred and x0_gt must be in the SAME space (both normalized).
        """
        bias_loss = torch.tensor(0.0, device=device)
        l_lpips = torch.tensor(0.0, device=device)
        l_edge = torch.tensor(0.0, device=device)

        if beta > 0:
            gauss_kernel = kernel.get_gaussian_kernel(kernel_size=37, sigma=6)
            lp_out = kernel.apply_lowpass_gaussian(x0_pred, gauss_kernel)
            lp_tgt = kernel.apply_lowpass_gaussian(x0_gt.clone(), gauss_kernel)
            bias_loss = F.mse_loss(lp_out, lp_tgt)

        if lpips_weight > 0:
            if x0_pred.dim() != 4:
                raise NotImplementedError("LPIPS auxiliary loss currently supports 2D only.")
            x0_clip = torch.clamp(x0_pred, -1, 1)
            gt_clip = torch.clamp(x0_gt, -1, 1)
            x0_rgb = x0_clip.repeat(1, 3, 1, 1) if x0_clip.shape[1] == 1 else x0_clip
            gt_rgb = gt_clip.repeat(1, 3, 1, 1) if gt_clip.shape[1] == 1 else gt_clip
            l_lpips = self.lpips_loss_fn(x0_rgb, gt_rgb).mean()

        if edge_weight > 0:
            l_edge = self.edge_loss_fn(x0_pred, x0_gt)

        return bias_loss, l_lpips, l_edge

    def train(self, pre_trained_model='/gpfs/work/aac/xingyiyao23/results/flow_matching_unsupervised_gaussian_mayo/models/model-190.pt', start_step=None, beta=0, lpips_weight=0, edge_weight=0):
        accelerator = self.accelerator
        device = accelerator.device

        if pre_trained_model is not None:
            self.load_model(pre_trained_model)
            print("model loaded from", pre_trained_model)

        if start_step is not None:
            self.step = start_step
        elif pre_trained_model is None:
            self.step = 0
        # else: self.step was already set by load_model()

        self.scheduler.step_size = 1
        val_loss = float("inf")
        val_fm_loss = float("inf")
        val_bias_loss = float("inf")
        val_lpips_loss = float("inf")
        val_edge_loss = float("inf")
        training_log = []

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                print(f"training epoch: {self.step + 1}")
                print(f"learning rate: {self.scheduler.get_last_lr()[0]}")

                avg_loss = []
                avg_fm = []
                avg_bias = []
                avg_lpips = []
                avg_edge = []

                self.opt.zero_grad()

                for count, batch in enumerate(self.dl):
                    batch_x0, batch_cond = batch
                    data_x0 = batch_x0.to(device)
                    data_cond = batch_cond.to(device) if self.conditional_diffusion else None

                    with self.accelerator.autocast():
                        # forward now returns img_norm (x0 in normalized space)
                        fm_loss, model_out, target, x_t, t_expand, img_norm = self.model(
                            img=data_x0, condition=data_cond
                        )

                        # recover x0_pred in normalized space
                        x0_pred = x_t - t_expand * model_out

                        # auxiliary losses: both x0_pred and img_norm are in
                        # the same (normalized) space — safe for auto_normalize
                        bias_loss, l_lpips, l_edge = self._compute_auxiliary_losses(
                            x0_pred, img_norm, beta, lpips_weight, edge_weight, device
                        )

                        loss = (fm_loss + beta * bias_loss + lpips_weight * l_lpips + edge_weight * l_edge)
                        loss = loss / self.accum_iter

                    self.accelerator.backward(loss)

                    if ((count + 1) % self.accum_iter == 0) or (count == len(self.dl) - 1):
                        accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.opt.step()
                        self.opt.zero_grad()

                    avg_loss.append(loss.item() * self.accum_iter)
                    avg_fm.append(fm_loss.item())
                    avg_bias.append(bias_loss.item())
                    avg_lpips.append(l_lpips.item())
                    avg_edge.append(l_edge.item())

                avg_loss = sum(avg_loss) / len(avg_loss)
                avg_fm = sum(avg_fm) / len(avg_fm)
                avg_bias = sum(avg_bias) / len(avg_bias)
                avg_lpips = sum(avg_lpips) / len(avg_lpips)
                avg_edge = sum(avg_edge) / len(avg_edge)

                pbar.set_description(f"loss:{avg_loss:.4f} fm:{avg_fm:.4f} bias:{avg_bias:.4f}")

                accelerator.wait_for_everyone()
                self.step += 1

                if self.step != 0 and divisible_by(self.step, self.save_model_every):
                    self.save(self.step)
                if self.step != 0 and divisible_by(self.step, self.train_lr_decay_every):
                    self.scheduler.step()

                if self.accelerator.is_main_process:
                    self.ema.update()

                # validation
                if self.step != 0 and divisible_by(self.step, self.validation_every):
                    print(f"validation at step {self.step}")
                    self.model.eval()
                    with torch.no_grad():
                        vl = []; vfm = []; vb = []; vlp = []; ve = []
                        for vbatch in self.dl_val:
                            vx0, vcond = vbatch
                            vx0 = vx0.to(device)
                            vcond = vcond.to(device) if self.conditional_diffusion else None
                            with self.accelerator.autocast():
                                fl, mo, tgt, vx_t, vt_exp, vimg_norm = self.model(img=vx0, condition=vcond)
                                vx0_pred = vx_t - vt_exp * mo

                                bl, ll, el = self._compute_auxiliary_losses(
                                    vx0_pred, vimg_norm, beta, lpips_weight, edge_weight, device
                                )
                                vtotal = fl + beta * bl + lpips_weight * ll + edge_weight * el

                            vl.append(vtotal.item())
                            vfm.append(fl.item()); vb.append(bl.item())
                            vlp.append(ll.item()); ve.append(el.item())

                        val_loss = sum(vl) / len(vl)
                        val_fm_loss = sum(vfm) / len(vfm)
                        val_bias_loss = sum(vb) / len(vb)
                        val_lpips_loss = sum(vlp) / len(vlp)
                        val_edge_loss = sum(ve) / len(ve)
                        print(f"  val loss={val_loss:.4f} fm={val_fm_loss:.4f} bias={val_bias_loss:.4f}")
                    self.model.train(True)

                training_log.append([
                    self.step, self.scheduler.get_last_lr()[0],
                    avg_loss, avg_fm, avg_bias, avg_lpips, avg_edge,
                    val_loss, val_fm_loss, val_bias_loss, val_lpips_loss, val_edge_loss,
                ])
                df = pd.DataFrame(training_log, columns=[
                    "iteration", "lr",
                    "train_loss", "train_fm", "train_bias", "train_lpips", "train_edge",
                    "val_loss", "val_fm", "val_bias", "val_lpips", "val_edge",
                ])
                log_dir = os.path.join(os.path.dirname(self.results_folder), "log")
                ff.make_folder([log_dir])
                df.to_excel(os.path.join(log_dir, "training_log.xlsx"), index=False)

                self.ds.on_epoch_end()
                self.ds_val.on_epoch_end()
                pbar.update(1)

        accelerator.print("training complete")


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
        dl = DataLoader(generator, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

        self.histogram_equalization = getattr(generator, "histogram_equalization", False)
        self.bins = getattr(generator, "bins", None)
        self.bins_mapped = getattr(generator, "bins_mapped", None)
        self.background_cutoff = getattr(generator, "background_cutoff", None)
        self.maximum_cutoff = getattr(generator, "maximum_cutoff", None)
        self.normalize_factor = getattr(generator, "normalize_factor", None)

        self.dl = dl
        self.cycle_dl = cycle(dl)

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
    ):
        """
        Run slice-by-slice inference.

        condition_img: (H, W, num_slices) — used for shape info.
        Condition data comes from generator[z] to ensure proper preprocessing.
        direct_use_of_model: if True, use self.model; otherwise use EMA weights.
        """
        bg = self.background_cutoff
        mx = self.maximum_cutoff
        nf = self.normalize_factor

        if not direct_use_of_model:
            self.load_model(trained_model_filename)

        device = self.device

        # FIX: branch on direct_use_of_model
        if direct_use_of_model:
            sampling_model = self.model
        else:
            sampling_model = self.ema.ema_model

        sampling_model.eval()
        print("model device:", next(sampling_model.parameters()).device)

        pred = np.zeros((self.image_size[0], self.image_size[1], condition_img.shape[-1]), dtype=np.float32)

        with torch.inference_mode():
            for z in range(condition_img.shape[-1]):
                print(f"  slice {z + 1} / {condition_img.shape[-1]}")

                # FIX: use generator[z] directly for deterministic slice alignment
                # instead of next(cycle_dl) which could drift out of sync
                if self.conditional_diffusion:
                    datas = self.generator[z]  # returns (target_slice, cond_slice)
                    cond = datas[1]
                    if isinstance(cond, np.ndarray):
                        cond = torch.from_numpy(cond).float()
                    if cond.dim() == 2:
                        cond = cond.unsqueeze(0).unsqueeze(0)  # (H,W) -> (1,1,H,W)
                    elif cond.dim() == 3:
                        cond = cond.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
                    data_cond = cond.to(device)
                else:
                    data_cond = None

                out = sampling_model.sample(condition=data_cond, batch_size=self.batch_size)
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