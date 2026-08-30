# ExPLoRe: Expert Patch-Level Loss Routing for Multi-Objective Masked Image Modeling

**Official PyTorch implementation of the ECCV 2026 ExPLoRe paper.**

ExPLoRe repurposes Soft Mixture of Experts (MoE) dispatch weights as **learned, per-patch loss coefficients** for multi-objective masked image modeling. The key mechanism is **loss-coupling**: loss gradients flow through dispatch weights to the router, enabling content-dependent specialization where different patches receive different emphases across training objectives. With ViT-Base + CLIP-B/16 teacher, ExPLoRe reaches **80.6% linear probe** and **85.3% finetuning** accuracy on ImageNet-1K, competitive with state-of-the-art at lower inference FLOPs.

**Paper:** [arXiv:2606.31201](https://arxiv.org/abs/2606.31201) (ECCV 2026) | **[MEDiC (no-MoE baseline)](https://github.com/aicip/MEDiC)** | **[MaskDistill-PyTorch](https://github.com/drkostas/MaskDistill-PyTorch)**

---

## Key Results

ViT-Base/16, ImageNet-1K, 300 epochs, frozen CLIP ViT-B/16 teacher:

| Model | Experts | Params | kNN | Linear | Finetune | mIoU (ADE20K) |
|---|---:|---:|---:|---:|---:|---:|
| **MEDiC (no-MoE baseline)** | — | 86M | 75.7 | 76.2 | 84.8 | 50.8 |
| **ExPLoRe (practical)** | 2 | 116M | 75.4 | 79.6 | 84.1 | 51.1* |
| **ExPLoRe (SOTA)** | 64 | 1.86B | 76.2 | **80.6** | **85.3** | **52.8** |

\* with FR+FA+ExD adaptation recipe (paper Tab. 4.8)

### Mechanism isolation (paper Tab. 4.5)

| Ablation | kNN | Δ |
|---|---:|---:|
| ExPLoRe E=2 (default) | 75.4 | — |
| − loss-coupling (detach) | 73.8 | −1.6 |
| Combine weights (no entropy reg) | 2.1 | −73.3 |

The **detach ablation** confirms loss-coupling is the core mechanism, not just adding MoE capacity.

---

## Setup

```bash
git clone https://github.com/aicip/ExPLoRe.git
cd ExPLoRe

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.1+, CUDA 11.8+.

For semantic segmentation:
```bash
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install mmsegmentation==0.30.0
```

### Dataset paths

Download ImageNet-1K and organize as:
```
/path/to/imagenet/
├── train/{class_name}/...
└── val/{class_name}/...
```

Edit `data.data_path`, `data.train_dir`, and `data.val_dir` in the pretrain YAMLs to point at your dataset root. For values shared across configs (data paths, W&B entity, SLURM defaults), copy `.env.example` to `.env` and edit; the pretrain and smoke scripts source `.env` at startup.

### Cluster / SLURM

The SLURM scripts under `scripts/` do **not** hardcode `--account`, `--qos`, or `--partition`. Pass them on the sbatch command line (or via a wrapper script):

```bash
sbatch --account=YOUR_ACCT --qos=YOUR_QOS --partition=YOUR_PART scripts/pretrain.sh configs/pretrain_explore_2exp.yaml
```

---

## Reproducing the paper

### Smoke test (validate setup works, ~15 min on 1 GPU)

```bash
sbatch --account=A --qos=Q --partition=P scripts/smoke_train_real.sh configs/pretrain_explore_smoke.yaml
```

This trains the 2-expert model for 2 epochs on a 10-image-per-class subset and verifies all losses, dispatch routing, and checkpoint I/O work end-to-end.

### Pretraining

**Practical recipe (2 experts, 116M params):**
```bash
sbatch scripts/pretrain.sh configs/pretrain_explore_2exp.yaml
```
~1 day on 4× H100. Reaches the kNN/LP/FT numbers in the "practical" row of the table above.

**SOTA recipe (64 experts, 1.86B params):**
```bash
sbatch scripts/pretrain.sh configs/pretrain_explore_64exp.yaml
```
~5–7 days on 4× H100. Reaches 80.6 linear probe / 85.3 finetune.

### Downstream evaluation

```bash
# kNN (frozen backbone, k=20, cosine similarity)
sbatch scripts/eval_knn.sh /path/to/checkpoint.pth

# Linear probe (frozen [CLS])
sbatch scripts/linprobe.sh /path/to/checkpoint.pth

# Full finetuning (with FR+FA+ExD recipe, paper Tab. 4.7)
sbatch scripts/finetune.sh /path/to/checkpoint.pth

# Semantic segmentation on ADE20K (UperNet, 160K iter)
sbatch scripts/eval_semseg.sh /path/to/checkpoint.pth
```

The default finetune config enables Freeze Routing + Freeze Attention + Expert Dropout (FR+FA+ExD, paper §4.6.3) which contributes +1.5% FT over vanilla MoE fine-tuning and +0.5% over the no-MoE baseline.

### Visualizing expert specialization

Reproduce paper Fig. 4.7 — per-patch dispatch-weight overlays showing the foreground/background expert split:

```bash
python scripts/visualize_dispatch_weights.py \
    --checkpoint output/pretrain_explore_2exp/checkpoint-299.pth \
    --config configs/pretrain_explore_2exp.yaml \
    --image_dir /path/to/your/images/ \
    --output_dir output/fig_4_7/ \
    --block_idx 11
```

The two experts learn complementary spatial specialization without any explicit supervision — Expert 0 concentrates on foreground objects, Expert 1 on background/context.

---

## How ExPLoRe works

ExPLoRe integrates Soft-MoE [Puigcerver et al., 2023] into a MEDiC-style multi-objective MIM framework at alternating ViT blocks `{1, 3, 5, 7, 9, 11}`. The Soft-MoE dispatch weights `D ∈ R^{B×N×E}` (softmax over patches per expert) are reused as **per-patch loss coefficients**:

```
L_token = (1/B) Σ_b [ Σ_n D[b, n, expert_0] · L_smooth_l1(student_n, teacher_n) ]
                   / Σ_n D[b, n, expert_0]
```

Because dispatch weights are differentiable, loss gradients flow back through `D` to the router parameters, enabling content-dependent specialization (paper §4.4.4). A detach ablation (Tab. 4.5) confirms loss-coupling — not just MoE capacity — is what drives the improvement.

**Dispatch (not combine) weights** are used because combine weights have a degeneracy that lets the router minimize loss by zeroing all expert contributions to a patch (paper §4.4.3); dispatch weights are normalized over patches per expert so each expert must distribute its attention somewhere.

---

## Repo structure

```
ExPLoRe/
├── configs/                                 # 6 YAML configs
│   ├── pretrain_explore_2exp.yaml          # Practical 2-expert recipe
│   ├── pretrain_explore_64exp.yaml         # SOTA 64-expert recipe
│   ├── pretrain_explore_smoke.yaml         # Smoke test
│   ├── finetune_explore.yaml               # Finetune w/ FR+FA+ExD defaults
│   ├── linprobe_explore.yaml               # Linear probe
│   └── semseg_explore.yaml                 # ADE20K UperNet
├── src/
│   ├── models/
│   │   ├── soft_moe.py                     # Soft-MoE layer (the core)
│   │   ├── vision_transformer_mim.py       # MoE-aware ViT student
│   │   ├── medic_model.py                  # MEDiCModel + MultiBlockWeightAggregator
│   │   ├── clip_teacher.py                 # Frozen CLIP teacher
│   │   └── decoder_mae.py                  # Pixel decoder (for §4.4 Token+Pixel extension)
│   ├── utils/
│   │   ├── losses.py                       # apply_moe_loss_weighting + compute_loss
│   │   └── masking_generator.py            # Block masking
│   ├── downstream/                         # Finetune, linprobe, semseg
│   ├── train.py                            # Pretraining loop w/ MoE logging
│   └── eval_knn.py                         # kNN evaluation
├── scripts/                                # SLURM scripts
└── tests/                                  # 167 CPU tests
```

---

## Tests

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -v
```

167 tests cover: model construction, loss components, masking, soft-MoE routing axes,
multi-block weight aggregator, MoE loss-weighting contract.

---

## Citation

```bibtex
@inproceedings{georgiou2026explore,
  title     = {ExPLoRe: Expert Patch-Level Loss Routing for Multi-Objective Masked Image Modeling},
  author    = {Georgiou, Konstantinos and Tang, Maofeng and Qi, Hairong},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

---

## License

Apache 2.0. See [LICENSE](LICENSE).

## Acknowledgements

This repo extends [aicip/MEDiC](https://github.com/aicip/MEDiC) (which itself extends
[drkostas/MaskDistill-PyTorch](https://github.com/drkostas/MaskDistill-PyTorch)).
Soft-MoE is from [Puigcerver et al., ICLR 2024](https://arxiv.org/abs/2308.00951).
