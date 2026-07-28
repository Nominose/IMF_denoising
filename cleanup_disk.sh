#!/bin/bash
# cleanup_disk.sh — reclaim gpfs quota across finished trials, in three tiers.
#
# DRY-RUN BY DEFAULT. Look at the preview, then re-run with DRY=0 to actually delete.
#     bash cleanup_disk.sh            # preview everything, delete nothing
#     bash cleanup_disk.sh 2>&1 | tail -40
#     DRY=0 bash cleanup_disk.sh      # execute
#     DRY=0 TIERS="1" bash cleanup_disk.sh    # only tier 1
#
# Tier 1 (~45G, regenerable): prediction volumes whose metrics are already extracted into xlsx or
#         downloaded, the unpacked PCCT zip, and the EMA trial's non-deployed checkpoints.
# Tier 2 (~85G, judgement): three GAN trials that are almost pure checkpoint stacks. Each is pruned
#         to its LAST epoch rather than deleted outright -- keeping ~571MB per trial preserves the
#         ability to re-evaluate without repeating tens of GPU-hours, which is cheap insurance given
#         none of these three has published metrics yet.
# Tier 3 (~4G): the v1 (pre-v2) models, superseded by the imf_v2 trials.
#
# PROTECTED: the two PCCT trials are refused unconditionally -- they are in use by the running
# pipeline, and a partial delete there would cost ~19h of training. The guard is a hard check, not
# a convention, so a mistyped trial name cannot reach them.
set -u

DRY="${DRY:-1}"
TIERS="${TIERS:-1 2 3}"
BASE=/gpfs/work/aac/xingyiyao23
M="$BASE/projects/denoising/models"
PROTECT="imf_v2_unsupervised_PCCT imf_gan_unsupervised_PCCT"

# --- safety: never touch a protected trial, whatever the caller asks for ---
guard () {
  for p in $PROTECT; do
    case "$1" in
      *"/$p/"*|*"/$p") echo "  !! REFUSED (protected, in use): $1"; return 1;;
    esac
  done
  return 0
}

del () {   # del <path...>
  for t in "$@"; do
    [ -e "$t" ] || continue
    guard "$t" || continue
    sz=$(du -sh "$t" 2>/dev/null | cut -f1)
    if [ "$DRY" = "1" ]; then
      echo "  [预览] rm -rf  $sz  $t"
    else
      rm -rf "$t" && echo "  [已删] $sz  $t"
    fi
  done
}

prune_to () {   # prune_to <trial> <epoch-to-keep>
  local d="$M/$1/models" keep="model-$2.pt" n=0
  [ -d "$d" ] || { echo "  (不存在,跳过) $1"; return; }
  guard "$d" || return
  [ -f "$d/$keep" ] || { echo "  !! $1: 要保留的 $keep 不存在 — 跳过,不删任何东西"; return; }
  for f in "$d"/model-*.pt; do
    [ -f "$f" ] || continue
    [ "$(basename "$f")" = "$keep" ] && continue
    if [ "$DRY" = "1" ]; then n=$((n+1)); else rm -f "$f"; n=$((n+1)); fi
  done
  echo "  $1: 保留 $keep,$([ "$DRY" = 1 ] && echo 将删 || echo 已删) $n 个其它 checkpoint"
}

prune_to_last () {   # prune_to_last <trial>  -- keep the highest-numbered checkpoint
  local d="$M/$1/models"
  [ -d "$d" ] || { echo "  (不存在,跳过) $1"; return; }
  local last
  last=$(ls "$d"/model-*.pt 2>/dev/null | sed 's/.*model-\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
  [ -n "$last" ] || { echo "  ($1 无 checkpoint)"; return; }
  prune_to "$1" "$last"
}

echo "==================== 磁盘清理 ===================="
echo "模式: $([ "$DRY" = 1 ] && echo '预览(不删任何东西)' || echo '*** 实际删除 ***')   档位: $TIERS"
echo "受保护(运行中,拒绝操作): $PROTECT"
echo; echo "--- 清理前 ---"; mmlsquota --block-size auto 2>/dev/null | tail -1

# ---------------- tier 1 ----------------
case " $TIERS " in *" 1 "*)
echo; echo "########## 第一档:已提取指标的预测结果 (~45G) ##########"
for t in imf_v2_unsupervised_gaussian_brainCT \
         imf_gan_unsupervised_gaussian_brainCT \
         imf_gan_ema_unsupervised_gaussian_brainCT \
         imf_v2_unsupervised_gaussian_mayo; do
  for p in "$M/$t"/pred_images_*; do del "$p"; done
done
del "$BASE/Data/soft_thins_xy.zip"          # already unpacked into Data/PCCT/
echo "-- EMA trial: 只留部署用的 epoch 28 --"
prune_to imf_gan_ema_unsupervised_gaussian_brainCT 28
;; esac

# ---------------- tier 2 ----------------
case " $TIERS " in *" 2 "*)
echo; echo "########## 第二档:三个 GAN trial,各留最后一个 checkpoint (~85G) ##########"
echo "(没有整个删除:每个留 ~571MB,以后想复评不必重训几十小时)"
for t in imf_gan_unsupervised_gaussian_mayo \
         imf_gan_nfe3_aw0.5_brainCT \
         imf_gan_aw0.8_nfe1_brainCT; do
  prune_to_last "$t"
  for p in "$M/$t"/pred_images_*; do del "$p"; done
done
;; esac

# ---------------- tier 3 ----------------
case " $TIERS " in *" 3 "*)
echo; echo "########## 第三档:v1 老模型 (~4G) ##########"
del "$M/imf_unsupervised_gaussian_brainCT" "$M/imf_unsupervised_gaussian_mayo"
;; esac

echo; echo "--- 清理后 ---"; mmlsquota --block-size auto 2>/dev/null | tail -1
if [ "$DRY" = "1" ]; then
  echo; echo "以上仅为预览。确认无误后执行:  DRY=0 bash cleanup_disk.sh"
else
  echo; echo "完成。各 trial 现状:"
  du -sh "$M"/*/ 2>/dev/null | sort -h
fi
