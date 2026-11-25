# Paper experiments

The [study protocol](protocol.json) gives the cohort sizes, training parameters, evaluation windows and sampling settings. NFE counts model evaluations per generated output. K counts independent outputs averaged **per conditioning input**.

| Main study | Conditioning input | NFE | K | Evaluations per denoised slice |
| --- | --- | ---: | ---: | ---: |
| Simulated brain CT | Two neighboring slices stacked as one input | 3 | 10 | 30 |
| Simulated abdominal CT | Odd and even reconstructions, each sampled independently | 3 | 10 | 60 |
| Real-world brain PCCT | Two neighboring slices stacked as one input | 5 | 20 | 100 |

The main brain CT and PCCT results use adversarially fine-tuned generators. The brain CT averaging study uses the generator before fine-tuning. Abdominal CT uses the flow model throughout.

## Read the results

Run these commands from the repository root. Printing the table uses the Python standard library. Plotting requires matplotlib 3.5 or newer.

```sh
python paper_experiments/report.py
python paper_experiments/report.py --plots /tmp/n2nimf-plots
```

- [Main comparison](results/main_comparison.csv) gives MAE, SSIM and LPIPS for the self-supervised methods and the noisy input (FBP). Values retain the precision of the manuscript table.
- [Sampling comparison](results/sampler_comparison.csv) gives the same metrics for DDIM, EDM, flow matching and N2N-iMF at seven NFE values with K = 10. The NFE = 3 rows reproduce the sampling comparison table at workbook precision.
- [Averaging study](results/averaging_study.csv) gives case means and sample standard deviations at each integer K from 1 to 20, with NFE = 3. Each case contributes once to each mean.
- [PCCT CNR](results/pcct_cnr.csv) reports gray–white matter contrast-to-noise ratio across eight scans. The blank K for Noise2Noise denotes a deterministic prediction.
- [Inference time](results/runtime.csv) gives median seconds per generated output for a single 512 × 512 slice on the same GPU at batch size one.

To recompute the K curve from per-case measurements, use the original CSV columns `dataset,patient,nfe,k,mae,ssim,lpips`. The brain input also has a `model` column, where `flow` denotes the generator before fine-tuning. This command emits only cohort statistics.

```sh
python paper_experiments/aggregate_k.py --brain "$BRAIN_METRICS" --abdominal "$ABDOMINAL_METRICS" > averaging_study.csv
```

## Run inference

The launcher uses the dataset-specific prediction scripts, including their preprocessing and sample averaging. It expects the existing [Docker data layout](../docker/README.md), the prepared test data and a compatible checkpoint. Clone the repository under its default name `IMF_denoising` so its package imports resolve. Supply the trial and epoch for the checkpoint being evaluated. For brain CT and PCCT, select the raw or EMA weights that correspond to your checkpoint.

```sh
python paper_experiments/infer.py brain --trial "$BRAIN_TRIAL" --epoch "$BRAIN_EPOCH" --weights "$BRAIN_WEIGHTS"
python paper_experiments/infer.py abdominal --trial "$ABDOMINAL_TRIAL" --epoch "$ABDOMINAL_EPOCH"
python paper_experiments/infer.py pcct --checkpoint "$PCCT_CHECKPOINT" --weights "$PCCT_WEIGHTS" \
  --data "$PCCT_DATA" --output "$PCCT_OUTPUT" --work "$PCCT_SAMPLE_CACHE"
```

Add `--dry-run` to inspect the exact sampling and averaging commands. For PCCT, use a separate output directory and sample cache for each checkpoint and weight selection. For abdominal CT, the main comparison uses the average saved at `pred_img_scans10.nii.gz`, which combines ten draws from each split reconstruction.
