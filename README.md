# GTX: Global Tabular eXplanations for Individual Neurons

Code and per-layer results for the paper *Global Tabular eXplanations for Individual Neurons* (preprint; link to be added).

A **Global Tabular eXplanation (GTX)** explains a single neuron with a small table of `C` input–activation pairs. A new input is approximated by
`u_weighted(x) = sim(x, x*) · u(x*)`, where `x*` is the most similar input in the table (cosine similarity in a feature space of a model *different*
from the explained one). The table is chosen to maximize the Pearson correlation between the true and the approximated activations, with a
two-stage search: `k`-means reduction of the training set (`k = 2000`; 500 for CIFAR-100) followed by a greedy search (Algorithm 1 of the paper).

> **Status.** This is research code, written for the cluster on which the paper was run (Slurm, one AMD MI210 GPU, 25-minute resumable jobs).
> Paths follow that layout (`GTX_RAIZ`, default `~/clusters/data/beyondcolor`). It is released as used, not as a polished library.

## Contents

| Path | What |
|---|---|
| `src/bc_gtx.py` | GTX, the selection baselines, ETA estimation, the optional open-pool mode and the heat-map plan. Main entry point. |
| `src/bc_redes.py`, `src/bc_inactivas.py` | Networks and layer pooling used for the ImageNet networks; classification of live / inhibitory / dead neurons. |
| `src/splits_le.py` | Reconstruction of the train/val/test partitions of Linear Explanations on ImageNet-val (the id files are **not** included; see below). |
| `hpc/bc16_gtx.sh`, `hpc/comun.example.sh` | Resumable Slurm launcher and an example of the cluster definitions (adapt to your cluster). |
| `tests/prueba_gtx.py` | Tests with synthetic data (no GPU, no datasets): batched greedy search against the naive Algorithm 1, clustering, baselines, resumption, output format, pool mode and heat maps. |
| `analysis/tabla_resultados.py` | Summarizes the per-neuron `jsonl` outputs into mean ± standard error per network, layer, method and `C`. |
| `results/` | `tabla_resultados.csv` / `.md`: mean ± standard error of every run (all methods, the networks and layers of the paper with `C ∈ {5, 6, 10, 11, 20}`, plus the extra layers and the open-pool runs with `C ∈ {5, 10}`). |

## Methods

`--metodos` accepts: `gtx`, `aleatorio` (random, five draws per neuron), `topk`, `extremos` (half highest / half lowest activations), `kmedoids`,
`kcenter`, `dpp` (greedy MAP of a DPP with kernel `L = F Fᵀ`) and `gtxpropio` (similarity computed with the explained layer's own activations: a leaky
upper bound, not a fair method). All of them are evaluated with the same `u_weighted`, on the held-out test partition. The greedy search is nested, so one run
gives every `C ≤ max(C)`.

## Data and models (not included)

* **ImageNet-val** (50 000 images, parquet shards in `GTX_RAIZ/imagenet1k/data/validation-NNNNN-of-00014.parquet`), **Places365-val** (small version) and the
  **CIFAR-100** test set. ImageNet images are *not* redistributed.
* **Partitions of Linear Explanations** (Oikarinen and Weng, 2024): `data/data_splits/{imagenet_val,places365_val,cifar100_val}/{train,val,test}_ids.pt` from
  the LE repository, placed under `GTX_REPO_LE` (default `~/clusters/ext/Linear-Explanations`). If they are missing the code falls back to a fixed-seed
  70/10/20 split and labels the partition `semilla0`: such results are **not** comparable with LE. LE's code is not included (that repository has no license).
* Network weights: torchvision ResNet-50 (V1) and ViT-B/16 (V1), ViT-L/32, ResNet-18 trained on Places365, the `cifar100_vgg16_bn` model of
  `chenyaofo/pytorch-cifar-models`, and, for similarity, torchvision's ResNet-101 and `google/vit-base-patch16-224`. `python src/bc_gtx.py --plan descarga` and
  `--plan descarga_ext` fetch the ones that are not in torchvision's cache (run them where there is internet).
* Activations for the ImageNet networks are read from per-shard files `atlas/cache/act_{net}_{NN}.pt` (a dict `{layer: [images, neurons]}` in the order of the
  parquet); activations of the other networks and of the optional open-pool mode are computed by `bc_gtx.py` itself.

## Usage

```bash
cp hpc/comun.example.sh hpc/comun.sh            # adapt to your cluster
# smoke test on one shard; it also prints an ETA of the full run (primitives measured at full size on synthetic data)
PLAN=humo ARGS="--red resnet50" bash hpc/bc16_gtx.sh 1
# GTX on ResNet-50 (layers 1-4) with the tables of the paper, then the baselines
PLAN=evalua ARGS="--C 5,6,10,11,20 --metodos gtx --red resnet50 --capas layer1,layer2,layer3,layer4" bash hpc/bc16_gtx.sh 4
PLAN=evalua ARGS="--C 5,6,10,11,20 --metodos aleatorio,topk,extremos,kmedoids,kcenter,dpp --red resnet50 --capas layer1,layer2,layer3,layer4" bash hpc/bc16_gtx.sh 2
# summary table
python analysis/tabla_resultados.py <folder-with-the-jsonl-outputs>
# tests (CPU, synthetic data)
python tests/prueba_gtx.py
```

The runs are resumable and idempotent: every job stops before a cell that does not fit in its time limit, rows are appended per neuron, and finished work is skipped
(`--existe` exits with 0 when nothing is left).

## Output

One JSON row per neuron: the selected examples (`indices`, in the order of the dataset), their activations, the correlation on the validation and test partitions,
the neuron's mean activation on the train partition, the partition used and the similarity model. GTX → `cosy/gtx/{net}__{layer}__gtx__C5-k2000.jsonl`; baselines →
`gtx/baselines/{net}__{layer}__{method}__C5.jsonl` (under `GTX_RAIZ`).

## Reproduction

The GTX results of the paper's Table 2 were obtained with the original (notebook) implementation. This repository is an independent, batched re-implementation.
On the same partitions it reproduces the 36 GTX values (12 layers × `C ∈ {5, 10, 20}`) within 0.015 in test correlation (within 0.01 in 32 of 36); see
Appendix E of the paper. The main differences are the clustering step (here `k`-means++ with 40 Lloyd iterations on L2-normalized features, instead of scikit-learn with
ten restarts) and the batching of the greedy search. In the VGG16 `features43` layer the crossing point with LE moves from `C = 11` (original) to between `C = 11` and `C = 20`.

## License and third-party material

Code: MIT (see `LICENSE`). Linear Explanations, CLIP-Dissect and other third-party methods are not included. ImageNet, Places365 and CIFAR-100 are subject to their own terms; no
images from them are distributed here.

## Citation

```
[to be added with the arXiv identifier]
```
