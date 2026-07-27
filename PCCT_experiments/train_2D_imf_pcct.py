"""
train_2D_imf_pcct.py — real-world PCCT denoising with improved MeanFlow (iMF). **NO GAN.**

Self-supervised Noise2Noise on ADJACENT SLICES (same pairing as brain CT, hence
`Generator_thinslice` + `Build_thinsliceCT`): the network predicts the current slice from its two
neighbours, so `condition_channels=2` and no clean image is ever used. This is real clinical PCCT
data -- there IS no ground truth, which is also why the evaluation is CNR (a no-reference metric,
see PCCT_experiments/eval_pcct_cnr.py) rather than MAE/SSIM.

Stage 1 of the two-stage recipe. Stage 2 (optional adversarial fine-tune) is
gan/train_2D_imf_gan_pcct.py, which fine-tunes THIS model's checkpoint.

Data (see unpack_pcct_hpc.sh + fix_pcct_xlsx.py)
-----------------------------------------------
  list   : <base>/Data/PCCT/Patient_lists/PCCT_split_hpc.xlsx
           batch 0 = train (22 cases), 1 = val (3), 2 = test (8, cases 29-36 -- the ones with ROIs)
  volumes: <base>/Data/PCCT/soft_thins_xy/<case>/soft_thins_0_noblank_sliced.nii.gz, each (512,512,50)
  models : <base>/projects/denoising/models/<trial_name>/models/

Config differs from the brain-CT v2 model in two ways, both inherited from the original PCCT
cDDPM script (PCCT_experiments/train_2D.py) because they describe THIS data:
  * NO histogram equalisation (the brain-CT bins do not apply to PCCT)
  * HU window [-1000, 1000]  (brain CT uses [-1000, 2000])
Backbone is otherwise identical: U-Net (+auxiliary v-head) + ImprovedMeanFlow.

    python PCCT_experiments/train_2D_imf_pcct.py
    python PCCT_experiments/train_2D_imf_pcct.py --train_num_steps 200 --train_batch_size 32
"""
import os
import sys
import argparse

# --- make `import IMF_denoising...` work regardless of where the repo is mounted ---
_HERE = os.path.dirname(os.path.abspath(__file__))            # .../IMF_denoising/PCCT_experiments
_REPO_PARENT = os.path.dirname(os.path.dirname(_HERE))        # dir that CONTAINS the IMF_denoising folder
for _p in (_REPO_PARENT, '/gpfs/work/aac/xingyiyao23/Code', '/host/c/Users/ROG/Documents/GitHub'):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np
import torch  # noqa: F401  (imported so a missing-torch env fails early, before data work)

import IMF_denoising.improved_mean_flow as imf
import IMF_denoising.functions_collection as ff
import IMF_denoising.Build_lists.Build_list as Build_list
import IMF_denoising.Generator_thinslice as Generator      # adjacent-slice N2N (NOT the odd/even Generator)
from IMF_denoising.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion import Unet


def _detect_base():
    for b in ('/gpfs/work/aac/xingyiyao23', '/host/d/research', '/host/d'):
        if os.path.isdir(os.path.join(b, 'Data')):
            return b
    return '/gpfs/work/aac/xingyiyao23'


_BASE = _detect_base()
_PCCT = os.path.join(_BASE, 'Data', 'PCCT')
_SLICED = 'soft_thins_0_noblank_sliced.nii.gz'


def _remap(p, case=None):
    """Resolve a patient-list path onto the uploaded data. The shipped list still points at the old
    docker box (/host/d/Data/PCCT/data/soft_thins/<case>/soft_thins_0_noblank.nii.gz) AND uses the
    pre-slicing filename, so both the directory and the filename may need substituting. Prefer
    fix_pcct_xlsx.py (which rewrites the list once); this is the runtime safety net."""
    if p is None:
        return p
    p = str(p)
    if os.path.exists(p):
        return p
    if case is not None:
        for name in (_SLICED, 'soft_thins_0_noblank.nii.gz'):
            cand = os.path.join(_PCCT, 'soft_thins_xy', str(case), name)
            if os.path.isfile(cand):
                return cand
    # last resort: keep the tail after the old data root
    for marker in ('/soft_thins/', '/soft_thins_xy/'):
        if marker in p:
            cand = os.path.join(_PCCT, 'soft_thins_xy', p.split(marker, 1)[1])
            if os.path.isfile(cand):
                return cand
            cand = os.path.join(os.path.dirname(cand), _SLICED)
            if os.path.isfile(cand):
                return cand
    return p


def get_args():
    ap = argparse.ArgumentParser('PCCT iMF (no GAN) training')
    ap.add_argument('--trial_name', default='imf_v2_unsupervised_PCCT')
    ap.add_argument('--patient_list_file',
                    default=os.path.join(_PCCT, 'Patient_lists', 'PCCT_split_hpc.xlsx'))
    ap.add_argument('--study_folder', default=os.path.join(_BASE, 'projects/denoising/models'))
    ap.add_argument('--batch_train', nargs='+', type=int, default=[0], help="xlsx 'batch' value(s) for training")
    ap.add_argument('--batch_val', nargs='+', type=int, default=[1], help="xlsx 'batch' value(s) for validation")
    ap.add_argument('--train_num_steps', type=int, default=200, help='number of epochs')
    ap.add_argument('--train_batch_size', type=int, default=32,
                    help='128x128 patches are small: 32 keeps the A800 busy (brain used 20). Raising '
                         'it further lowers updates/epoch, so raise --train_num_steps alongside.')
    ap.add_argument('--train_lr', type=float, default=1e-4)
    ap.add_argument('--patch_size', type=int, nargs=2, default=[128, 128])
    ap.add_argument('--num_patches_per_slice', type=int, default=2)
    ap.add_argument('--background_cutoff', type=float, default=-1000.0, help='PCCT window low (HU)')
    ap.add_argument('--maximum_cutoff', type=float, default=1000.0, help='PCCT window high (HU)')
    ap.add_argument('--save_every', type=int, default=10, help='epochs between checkpoints (each ~571MB)')
    ap.add_argument('--validation_every', type=int, default=10)
    ap.add_argument('--pre_trained_model', default=None)
    ap.add_argument('--start_step', type=int, default=0)
    return ap.parse_args()


def main():
    args = get_args()
    supervision = 'unsupervised'
    condition_channel = 2            # adjacent slices (previous + next) as the condition
    image_size = [512, 512]
    histogram_equalization = False   # PCCT: plain HU window, no hist-eq (brain-CT bins do not apply)
    normalize_factor = 'equation'

    print('data base   :', _BASE)
    print('patient list:', args.patient_list_file)
    if not os.path.isfile(args.patient_list_file):
        raise FileNotFoundError(
            f'{args.patient_list_file}\nRun PCCT_experiments/unpack_pcct_hpc.sh then '
            'python PCCT_experiments/fix_pcct_xlsx.py --write')

    # ---- patient list. Build_thinsliceCT returns (batch, pid, subid, random_num, noise, gt);
    # PCCT_split has no random_num / ground_truth columns -> those come back as None, which is fine
    # here because self-supervised training uses the noisy volume as BOTH input and target.
    bs = Build_list.Build_thinsliceCT(args.patient_list_file)
    _, pid_tr, _, _, cond_tr, _ = bs.__build__(batch_list=args.batch_train)
    _, pid_va, _, _, cond_va, _ = bs.__build__(batch_list=args.batch_val)

    cond_tr = np.array([_remap(p, c) for p, c in zip(cond_tr, pid_tr)])
    cond_va = np.array([_remap(p, c) for p, c in zip(cond_va, pid_va)])
    x0_tr, x0_va = cond_tr, cond_va          # N2N: target and condition come from the SAME volume

    for name, arr in (('train', cond_tr), ('val', cond_va)):
        miss = [p for p in arr if not os.path.isfile(p)]
        if miss:
            raise FileNotFoundError(f'{len(miss)} {name} volume(s) missing, e.g. {miss[:3]}')
    print(f'train cases: {len(cond_tr)} | val cases: {len(cond_va)}')
    print('train[0]:', cond_tr[0])

    # ---- backbone: same U-Net (+aux v-head) as the brain-CT v2 model ----
    base_model = Unet(
        problem_dimension='2D', init_dim=64, out_dim=1, channels=1,
        conditional_diffusion=True, condition_channels=condition_channel,
        downsample_list=(True, True, True, False), upsample_list=(True, True, True, False),
        full_attn=(None, None, False, True), auxiliary_v_head=True,
    )
    diffusion_model = imf.ImprovedMeanFlow(
        base_model, image_size=args.patch_size, ratio_r_neq_t=0.50,
        clip_or_not=False, auto_normalize=False, adaptive_weight_power=1.0, v_loss_weight=0.5,
    )

    G = Generator.Dataset_2D
    gen_tr = G(
        supervision=supervision,
        img_list=x0_tr, condition_list=cond_tr, image_size=image_size,
        # 48, NOT 50: with adjacent-slice conditioning the first and last slice have no neighbour,
        # so Generator_thinslice narrows slice_range=None to [1, n-1] -> 48 usable slices out of 50.
        # num_slices_per_image only sets __len__, it is NOT clipped against that list, so asking for
        # 50 indexes past the end and dies with IndexError partway through epoch 1.
        num_slices_per_image=48, random_pick_slice=True, slice_range=None,
        num_patches_per_slice=args.num_patches_per_slice, patch_size=args.patch_size,
        histogram_equalization=histogram_equalization, bins=None, bins_mapped=None,
        background_cutoff=args.background_cutoff, maximum_cutoff=args.maximum_cutoff,
        normalize_factor=normalize_factor, shuffle=True, augment=True, augment_frequency=0.5,
    )
    gen_va = G(
        supervision=supervision,
        img_list=x0_va, condition_list=cond_va, image_size=image_size,
        num_slices_per_image=20, random_pick_slice=False, slice_range=[20, 40],  # mid-volume, inside 50
        num_patches_per_slice=1, patch_size=[512, 512],
        histogram_equalization=histogram_equalization, bins=None, bins_mapped=None,
        background_cutoff=args.background_cutoff, maximum_cutoff=args.maximum_cutoff,
        normalize_factor=normalize_factor,
    )

    save_folder = os.path.join(args.study_folder, args.trial_name, 'models')
    ff.make_folder([os.path.dirname(save_folder), save_folder])
    print('save models ->', save_folder)

    trainer = imf.Trainer(
        diffusion_model=diffusion_model, generator_train=gen_tr, generator_val=gen_va,
        train_batch_size=args.train_batch_size, accum_iter=1,
        train_num_steps=args.train_num_steps, results_folder=save_folder,
        train_lr=args.train_lr, train_lr_decay_every=200,
        save_models_every=args.save_every, validation_every=args.validation_every,
    )
    trainer.train(pre_trained_model=args.pre_trained_model, start_step=args.start_step)


if __name__ == '__main__':
    main()
