"""Particiones train/val/test de Linear Explanations (Oikarinen y Weng, ICML 2024) sobre ImageNet-val en el orden del parquet.

Los archivos {train,val,test}_ids.pt NO se redistribuyen aqui: se obtienen del repositorio de LE (data/data_splits/imagenet_val/) y se ponen en
<LE_REPO>/data/data_splits/imagenet_val/ (variable de entorno GTX_REPO_LE; por omision ~/clusters/ext/Linear-Explanations). Sus ids indexan las imagenes
en el orden de torchvision.ImageFolder (por clase y luego por numero de imagen); aqui se reconstruye ese orden a partir de la etiqueta y del numero
`val_<n>` de cada imagen y se traduce al orden del parquet. Si no se puede reconstruir, se usa una particion 70/10/20 con semilla fija ('semilla0').
"""
import os
import re

import numpy as np
import torch

REPO_LE = os.path.expanduser(os.environ.get("GTX_REPO_LE", "~/clusters/ext/Linear-Explanations"))


def particion(datos, cache, plan):
    """datos: dict con 'label' [N] (tensor) y 'path' (lista de N rutas con 'val_<n>'), en el orden del parquet.
    Devuelve ({'train','val','test': indices ordenados}, 'oficial' | 'semilla0')."""
    N = len(datos["label"])
    if plan != "humo":
        d = os.path.join(REPO_LE, "data", "data_splits", "imagenet_val")
        nums = [re.search(r"val_(\d+)", p or "") for p in datos["path"]]
        if N == 50000 and all(nums) and all(os.path.exists(os.path.join(d, f"{k}_ids.pt")) for k in ("train", "val", "test")):
            orden = np.lexsort((np.array([int(m.group(1)) for m in nums]), datos["label"].numpy()))
            ids = {k: torch.load(os.path.join(d, f"{k}_ids.pt")).numpy() for k in ("train", "val", "test")}
            if all(len(v) > 0 and v.max() < N for v in ids.values()):
                return {k: torch.from_numpy(np.sort(orden[v])) for k, v in ids.items()}, "oficial"
    rng = np.random.default_rng(0)
    p = rng.permutation(N)
    n1, n2 = int(0.7 * N), int(0.8 * N)
    return {"train": torch.from_numpy(np.sort(p[:n1])), "val": torch.from_numpy(np.sort(p[n1:n2])),
            "test": torch.from_numpy(np.sort(p[n2:]))}, "semilla0"
