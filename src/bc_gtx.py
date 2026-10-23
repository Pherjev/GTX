#!/usr/bin/env python3
"""GTX (Global Tabular eXplanations) y baselines de seleccion de ejemplos sobre ImageNet-val, con las activaciones del atlas.

QUE HACE. Para cada neurona elige una TABLA de C imagenes de entrenamiento y mide, en test, la correlacion de Pearson entre la activacion
real y la aproximada u_pesada(x) = sim(x, x*) * u(x*), con x* = vecino mas parecido dentro de la tabla (coseno sobre un modelo de similitud
DISTINTO del analizado: ResNet-101 para ViT/CLIP, ViT-B/16 de Hugging Face para ResNet). Mismas particiones que LE (35 000/5 000/10 000)
y mismas activaciones agrupadas del atlas (`atlas/cache/act_{red}_{NN}.pt`), asi que es comparable con `atlas/resultados/*__le__siglip.jsonl`.

METODOS (--metodos; todos se evaluan con la MISMA u_pesada, solo cambia como se elige la tabla):
  gtx        k-means (k=2000) sobre las caracteristicas de similitud de TRAIN -> representante por clusters -> busqueda voraz de rho en VAL (Alg. 1 del paper).
             La voraz es anidada: la tabla de C=5 es el prefijo de la de C=10 (una sola corrida da todas las C).
  aleatorio  C imagenes de train al azar (--semillas semillas; se promedia). Baseline pedido por los revisores.
  topk       las C imagenes de train con mayor activacion.
  extremos   mitad mayores y mitad menores activaciones de train.
  kmedoids   C medoides (distancia coseno) sobre una submuestra de train; NO depende de la neurona (solo diversidad).
  kcenter    farthest-first sobre la misma submuestra; NO depende de la neurona.
  dpp        MAP voraz de un DPP con nucleo L = F F^T (volumen maximo) sobre la misma submuestra; NO depende de la neurona.
  gtxpropio  como gtx pero con la similitud calculada con las activaciones de la PROPIA capa (centradas): el k-NN con el embedding del modelo analizado
             que pidio un revisor. Contiene a la propia neurona: es una cota con fuga, no un metodo justo.

DESVIACIONES CONOCIDAS respecto del cuaderno (Colab): k-means propio en GPU (Lloyd con inicio k-means++, --kmeans-iters) sobre vectores normalizados en L2,
en vez de sklearn con n_init=10; representante = el miembro mas cercano al centroide (como dice el paper; el cuaderno usa la media del cluster en algunas
celdas); en la voraz, un candidato ya elegido no puede repetirse (el Alg. 1 le pone rho=0). El vector de ResNet-101 es --sim-vec logits (los 1000 logits,
como el cuaderno; avgpool = 2048). NADA de esto esta validado contra los numeros del paper todavia: eso es el humo + `--resumen` frente a LE.

SALIDA (una fila JSON por neurona, formato de GTX_INSTRUCCIONES.md sec. 7): gtx -> <raiz>/cosy/gtx/{red}__{capa}__gtx__C5-k2000.jsonl (lo lee el sitio);
baselines -> <raiz>/gtx/baselines/{red}__{capa}__{metodo}__C5.jsonl. `indices` = posicion en el orden del parquet de ImageNet-val.

REDES DEL PAPER FUERA DEL ATLAS (EXT): vitl32 (ImageNet, bloque23_mlp), resnet18_places (Places365 val small, layer1-4), vgg16_cifar (CIFAR-100 test, features_43 y classifier_4):
`--plan descarga_ext` en el nodo de acceso; luego `--plan evalua --red <una de ellas>` calcula primero sus caracteristicas de similitud y sus activaciones (celdas 'sim' y 'act')
y despues los metodos. k por omision: 2000 (ImageNet, Places365), 500 (CIFAR-100).

FUENTE POOL (sitio publico): `--fuente pool` repite todo sobre el pool abierto (pool_abierto/pool_listo.jsonl, 14 trozos): caracteristicas de similitud, activaciones de las capas
(cache/act_{red}_pool_NN.pt), particion 70/10/20 con semilla 0 en pool_abierto/gtx_particion.json y salida en cosy/gtx_pool/. `--plan miniaturas_pool` exporta miniaturas y creditos de las
imagenes que quedaron en alguna tabla. Las correlaciones del pool NO son comparables con las de ImageNet ni con LE.

MAPAS DE CALOR (--plan mapas [--fuente pool]): para cada imagen de las tablas GTX de C=10 y cada neurona, el mapa espacial de la activacion de esa unidad (una pasada por trozo, reanudable),
ensamblado en gtx/mapas/{imagenet|pool}/{red}__{capa}.npz (uint8 normalizado por el maximo de los 10 mapas de la neurona, parte positiva). Capas espaciales y sus gemelas tokmean; las vectoriales no tienen mapa.
Requiere haber corrido antes --metodos gtx --C 10 (o 5,10) en esa fuente.

USO (cluster; ver hpc/noche/bc16_gtx.sh):
  python3 src/bc_gtx.py --plan descarga                       # SOLO en el nodo de acceso (internet): ResNet-101 y ViT-B/16 HF
  python3 src/bc_gtx.py --plan humo --exige-gpu               # shard 0, resnet50/layer1, 16 neuronas + ETA de la corrida completa
  python3 src/bc_gtx.py --plan evalua --red resnet50 --capas layer1,layer4 --minutos 25 --exige-gpu
  python3 src/bc_gtx.py --plan evalua --existe | --resumen
Banderas de la guia: --plan, --minutos, --existe, --resumen, --exige-gpu.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

_pl = os.path.expanduser("~/clusters/pylibs")
if os.path.isdir(_pl):
    sys.path.insert(0, _pl)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

RAIZ = os.path.expanduser(os.environ.get("GTX_RAIZ", "~/clusters/data/beyondcolor"))  # raiz de datos (ver README)
NSHARD = 14
SIM_DE = {"resnet50": "vitb16hf", "vitb16": "resnet101", "clipvitb16": "resnet101"}  # modelo de similitud (siempre distinto del analizado)
METODOS = ("gtx", "aleatorio", "topk", "extremos", "kmedoids", "kcenter", "dpp", "gtxpropio")
INDEP = ("kmedoids", "kcenter", "dpp")  # no dependen de la neurona
REPO_LE = os.path.expanduser(os.environ.get("GTX_REPO_LE", "~/clusters/ext/Linear-Explanations"))
FUENTE = "imagenet"  # --fuente pool: la misma tabla GTX pero sobre el POOL ABIERTO (imagenes redistribuibles, para el sitio publico)
POOL_DIR = os.path.join(RAIZ, "pool_abierto")
_POOL = {}
WORKERS = 8
# Redes del paper que NO estan en el atlas: aqui se calculan sus activaciones (una pasada) y sus caracteristicas de similitud. ds = conjunto de datos; n = imagenes.
EXT = {
    "vitl32": dict(ds="imagenet", sim="resnet101", k=2000, n=50000, capas=[("bloque23_mlp", 4096)]),
    "resnet18_places": dict(ds="places365", sim="vitb16hf", k=2000, n=36500, capas=[("layer1", 64), ("layer2", 128), ("layer3", 256), ("layer4", 512)]),
    "vgg16_cifar": dict(ds="cifar100", sim="vitb16hf", k=500, n=10000, capas=[("features_43", 512), ("classifier_4", 512)]),
}
SPLIT_LE = {"places365": "places365_val", "cifar100": "cifar100_val"}
SIM_DE.update({r: v["sim"] for r, v in EXT.items()})
KFULL = 2000  # k del k-means de la corrida completa (el humo usa uno menor para correr rapido, pero el ETA se mide con este)
TAMS_ETA = {"resnet50": [256, 512, 1024, 2048, 1000], "vitb16": [3072] * 4 + [768] + [3072] * 3, "clipvitb16": [3072] * 4 + [512] + [3072] * 3,
            **{r: [u for _, u in v["capas"]] for r, v in EXT.items()}}


# ====================================================================== nucleo numerico (torch, sin E/S)
def normaliza(X):
    return torch.nn.functional.normalize(X.float(), dim=1)


def corr_col(pred, u):
    """Pearson por fila de [B, M]; NaN (varianza cero) -> 0, como el paper."""
    p = pred - pred.mean(1, keepdim=True)
    q = u - u.mean(1, keepdim=True)
    r = (p * q).sum(1) / (p.pow(2).sum(1).sqrt() * q.pow(2).sum(1).sqrt())
    return torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def kmeans(X, k, iters=40, seed=0, trozo=8192):
    """Lloyd en torch sobre X [N, D] (ya normalizado). Devuelve (centros [k, D], etiquetas [N]). Clusters vacios se reinician con puntos al azar."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    N = X.shape[0]
    k = min(k, N)
    i0 = int(torch.randint(0, N, (1,), generator=g))  # inicio k-means++ (D^2): k pasos de un producto matriz-vector
    C = torch.empty(k, X.shape[1], device=X.device, dtype=X.dtype)
    C[0] = X[i0]
    d2 = ((X - X[i0]) ** 2).sum(1)
    for j in range(1, k):
        i = int(torch.multinomial((d2 / d2.sum().clamp(min=1e-30)).cpu(), 1, generator=g)) if d2.sum() > 0 else int(torch.randint(0, N, (1,), generator=g))
        C[j] = X[i]
        d2 = torch.minimum(d2, ((X - X[i]) ** 2).sum(1))
    lab = torch.zeros(N, dtype=torch.long, device=X.device)
    for _ in range(iters):
        c2 = (C * C).sum(1)
        for i in range(0, N, trozo):
            lab[i:i + trozo] = (c2[None, :] - 2 * X[i:i + trozo] @ C.T).argmin(1)
        S = torch.zeros_like(C).index_add_(0, lab, X)
        n = torch.bincount(lab, minlength=k).float()
        vacio = n == 0
        C = torch.where(vacio[:, None], X[torch.randint(0, N, (k,), generator=g).to(X.device)], S / n.clamp(min=1)[:, None])
    return C, lab


def representantes(X, C, lab, trozo=8192):
    """Para cada cluster no vacio, el indice del miembro de X mas cercano a su centro."""
    d = torch.empty(X.shape[0], device=X.device)
    for i in range(0, X.shape[0], trozo):
        d[i:i + trozo] = ((X[i:i + trozo] - C[lab[i:i + trozo]]) ** 2).sum(1)
    reps = []
    for c in torch.unique(lab).tolist():
        m = (lab == c).nonzero().squeeze(1)
        reps.append(int(m[d[m].argmin()]))
    return torch.tensor(sorted(reps), dtype=torch.long)


def voraz_lote(S, a, u, Cmax, trozo=256):
    """Busqueda voraz de rho para un lote de neuronas (Algoritmo 1 del paper).
    S [k, M]: similitud de los k candidatos con las M imagenes de validacion. a [B, k]: activacion de cada neurona en cada candidato. u [B, M]: activacion real en validacion.
    Devuelve (sel [B, Cmax] indices de candidato en orden de eleccion, rho [B, Cmax] correlacion en val tras cada paso)."""
    B, k = a.shape
    M = S.shape[1]
    dev = S.device
    mejor = torch.full((B, M), -float("inf"), device=dev)
    cur = torch.zeros(B, M, device=dev)
    uc = u - u.mean(1, keepdim=True)
    su = uc.pow(2).sum(1).sqrt()
    tomado = torch.zeros(B, k, dtype=torch.bool, device=dev)
    sel, hist = [], []
    ar = torch.arange(B, device=dev)
    for _ in range(min(Cmax, k)):
        rho = torch.empty(B, k, device=dev)
        for j0 in range(0, k, trozo):
            Sj = S[j0:j0 + trozo]
            aj = a[:, j0:j0 + trozo]
            pred = torch.where(Sj[None] > mejor[:, None, :], Sj[None] * aj[:, :, None], cur[:, None, :])
            pm = pred - pred.mean(2, keepdim=True)
            rho[:, j0:j0 + trozo] = torch.nan_to_num((pm * uc[:, None, :]).sum(2) / (pm.pow(2).sum(2).sqrt() * su[:, None]), nan=0.0, posinf=0.0, neginf=0.0)
        rho[tomado] = -2.0
        r, i = rho.max(1)
        Si = S[i]
        cur = torch.where(Si > mejor, Si * a[ar, i][:, None], cur)
        mejor = torch.maximum(mejor, Si)
        tomado[ar, i] = True
        sel.append(i)
        hist.append(r)
    return torch.stack(sel, 1), torch.stack(hist, 1)


def pred_pesada(Fn, idx, act_sel, Fe):
    """u_pesada sobre las imagenes de evaluacion Fe [M, D] con tablas idx [B, C] (indices en Fn) y sus activaciones act_sel [B, C]. -> [B, M]"""
    V = Fn[idx]  # [B, C, D]
    s = torch.einsum("bcd,md->bcm", V, Fe)
    m, j = s.max(1)
    return m * torch.gather(act_sel, 1, j)


def sel_kcenter(F, C, seed=0):
    """Farthest-first (distancia coseno) sobre F normalizado [n, D]; arranca de un punto elegido por semilla."""
    g = torch.Generator().manual_seed(seed)
    i = int(torch.randint(0, F.shape[0], (1,), generator=g))
    out, d = [i], 1 - F @ F[i]
    for _ in range(C - 1):
        i = int(d.argmax())
        out.append(i)
        d = torch.minimum(d, 1 - F @ F[i])
    return torch.tensor(out)


def sel_dpp(F, C):
    """MAP voraz de un DPP con L = F F^T: en cada paso el elemento con mayor residuo tras proyectar fuera lo ya elegido (Gram-Schmidt)."""
    R = F.clone()
    out = []
    for _ in range(C):
        n = R.pow(2).sum(1)
        if out:
            n[torch.tensor(out)] = -1
        i = int(n.argmax())
        out.append(i)
        q = R[i] / R[i].norm().clamp(min=1e-12)
        R = R - (R @ q)[:, None] * q[None, :]
    return torch.tensor(out)


def sel_kmedoids(F, C, seed=0, iters=20):
    """k-medoids alternante (Voronoi) con distancia coseno sobre F normalizado [n, D]. Inicio: farthest-first."""
    n = F.shape[0]
    med = sel_kcenter(F, C, seed)
    for _ in range(iters):
        asig = (F @ F[med].T).argmax(1)
        nuevo = med.clone()
        for c in range(C):
            m = (asig == c).nonzero().squeeze(1)
            if len(m) == 0:
                continue
            G = F[m] @ F[m].T
            nuevo[c] = m[G.sum(1).argmax()]  # minimiza la suma de (1 - coseno)
        if torch.equal(nuevo, med):
            break
        med = nuevo
    return med


# ====================================================================== datos
def rutas(salida=None):
    H = {"cache_act": os.path.join(RAIZ, "atlas", "cache"), "le_cache": os.path.join(RAIZ, "le", "cache"), "pesos": os.path.join(RAIZ, "pesos"),
         "gtx": os.path.join(RAIZ, "gtx"), "atlas_res": os.path.join(RAIZ, "atlas", "resultados")}
    H["cache"] = os.path.join(H["gtx"], "cache")
    suf = "_pool" if FUENTE == "pool" else ""
    H["gtx_dir"] = os.path.join(salida or os.path.join(RAIZ, "cosy"), "gtx" + suf) if salida != "HUMO" else os.path.join(H["gtx"], "humo", "cosy_gtx" + suf)
    H["base_dir"] = os.path.join(H["gtx"], "baselines" + suf) if salida != "HUMO" else os.path.join(H["gtx"], "humo", "baselines" + suf)
    return H


def dir_de(H, metodo):
    return H["gtx_dir"] if metodo == "gtx" else H["base_dir"]


def config_de(metodo, C, k):
    return f"C{C}-k{k}" if metodo in ("gtx", "gtxpropio") else f"C{C}"


def ruta_salida(H, red, capa, metodo, C, k):
    return os.path.join(dir_de(H, metodo), f"{red}__{capa}__{metodo}__{config_de(metodo, C, k)}.jsonl")


def lee_hechas(ruta):
    out = set()
    if os.path.exists(ruta):
        for l in open(ruta, encoding="utf-8"):
            if l.strip():
                try:
                    out.add(json.loads(l)["neurona"])
                except Exception:
                    pass
    return out


def particion_le(H, shards):
    """Particion train/val/test de LE (35 000/5 000/10 000) en indices globales del orden del parquet, restringida a `shards`. -> (dict, tipo, offsets)"""
    import splits_le as bc_le
    labs, paths, offs, o = [], [], [], 0
    for i in range(NSHARD):
        d = torch.load(os.path.join(H["le_cache"], f"le_feats_{i:02d}.pt"))
        labs.append(d["label"])
        paths += d["path"]
        offs.append(o)
        o += len(d["label"])
        del d
    datos = {"label": torch.cat(labs), "path": paths}
    p, tipo = bc_le.particion(datos, H["le_cache"], "evalua")
    corte = offs[max(shards)] + len(labs[max(shards)]) if shards != list(range(NSHARD)) else o
    p = {k: v[v < corte] for k, v in p.items()}
    return p, tipo, offs


def ds_de(red):
    if red in EXT:
        return EXT[red]["ds"]
    return "pool" if FUENTE == "pool" else "imagenet"


def pool_filas():
    """pool_listo.jsonl (pid consecutivo, ruta de cada imagen y sus creditos). Se lee una vez."""
    if "f" not in _POOL:
        ruta = os.path.join(POOL_DIR, "pool_listo.jsonl")
        filas = [json.loads(l) for l in open(ruta, encoding="utf-8") if l.strip()]
        for i, f in enumerate(filas):
            assert f["pid"] == i, "pool_listo.jsonl debe tener pid consecutivos"
        _POOL["f"] = filas
    return _POOL["f"]


def limites_pool(i):
    """[lo, hi) de los pid del 'shard' i del pool (NSHARD trozos iguales, para que cada pasada quepa en un eslabon y sea reanudable)."""
    n = len(pool_filas())
    t = -(-n // NSHARD)
    return i * t, min((i + 1) * t, n)


def particion_pool(shards):
    """70/10/20 con semilla fija sobre TODO el pool, guardada en pool_abierto/gtx_particion.json (la misma para cualquier red y corrida). Con shards parciales (humo) se recorta."""
    ruta = os.path.join(POOL_DIR, "gtx_particion.json")
    n = len(pool_filas())
    if os.path.exists(ruta):
        d = json.load(open(ruta))
        p = {k: torch.tensor(v, dtype=torch.long) for k, v in d.items()}
        assert sum(len(v) for v in p.values()) == n, "gtx_particion.json no corresponde a este pool"
    else:
        q = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        n1, n2 = int(0.7 * n), int(0.8 * n)
        p = {"train": q[:n1].sort().values, "val": q[n1:n2].sort().values, "test": q[n2:].sort().values}
        json.dump({k: v.tolist() for k, v in p.items()}, open(ruta + ".tmp", "w"))
        os.replace(ruta + ".tmp", ruta)
    corte = limites_pool(max(shards))[1]
    return {k: v[v < corte] for k, v in p.items()}, "pool_semilla0", [0]


def abre_imagen(ruta):
    from PIL import Image
    return Image.open(ruta).convert("RGB")


class DatosPool(torch.utils.data.Dataset):
    """Imagenes del pool abierto con pid en [lo, hi); el 'indice' que devuelve es local al trozo."""

    def __init__(self, i, tf):
        self.lo, self.hi = limites_pool(i)
        self.rutas = [os.path.join(POOL_DIR, f["ruta"]) for f in pool_filas()[self.lo:self.hi]]
        self.tf = tf

    def __len__(self):
        return self.hi - self.lo

    def __getitem__(self, k):
        return self.tf(abre_imagen(self.rutas[k])), 0, k


def ruta_act(H, red, i):
    """Activaciones calculadas aqui (redes de EXT y cualquier red sobre el pool). ImageNet: por shard, como el atlas; pool: por trozo; otros conjuntos: un solo archivo."""
    ds = ds_de(red)
    return os.path.join(H["cache"], f"act_{red}_{i:02d}.pt" if ds == "imagenet" else f"act_{red}_pool_{i:02d}.pt" if ds == "pool" else f"act_{red}.pt")


def carga_act(H, red, capa, shards):
    if red in EXT or ds_de(red) == "pool":
        return torch.cat([torch.load(ruta_act(H, red, i))[capa] for i in shards]).float()
    return torch.cat([torch.load(os.path.join(H["cache_act"], f"act_{red}_{i:02d}.pt"))[capa] for i in shards]).float()


def particion_ext(ds, N):
    """Particion de LE para places365/cifar100 (ids sobre el orden de torchvision). Si el repositorio de LE no las trae: semilla fija 70/10/20 (NO comparable con LE)."""
    d = os.path.join(REPO_LE, "data", "data_splits", SPLIT_LE[ds])
    if all(os.path.exists(os.path.join(d, f"{k}_ids.pt")) for k in ("train", "val", "test")):
        ids = {k: torch.load(os.path.join(d, f"{k}_ids.pt")).long().sort().values for k in ("train", "val", "test")}
        if all(v.max() < N for v in ids.values()):
            return ids, "oficial", [0]
    p = torch.randperm(N, generator=torch.Generator().manual_seed(0))
    n1, n2 = int(0.7 * N), int(0.8 * N)
    return {"train": p[:n1].sort().values, "val": p[n1:n2].sort().values, "test": p[n2:].sort().values}, "semilla0", [0]


class DatosExt(torch.utils.data.Dataset):
    """Places365 (val, small) o CIFAR-100 (test) de torchvision, en su orden, ya descargados con --plan descarga_ext."""

    def __init__(self, H, ds, tf):
        self.d, self.tf = abre_ds(H, ds), tf

    def __len__(self):
        return len(self.d)

    def __getitem__(self, i):
        im, lab = self.d[i]
        return self.tf(im.convert("RGB")), lab, i


def abre_ds(H, ds, descarga=False):
    from torchvision import datasets
    raiz = os.path.join(H["gtx"], "datos")
    if ds == "places365":
        return datasets.Places365(raiz, split="val", small=True, download=descarga)
    return datasets.CIFAR100(raiz, train=False, download=descarga)


def recorre(H, ds, i, tf, f, dev, bloque=64):
    """Pasa todo un shard de ImageNet-val (o el conjunto completo) por f(x) -> tensor o dict de tensores, y devuelve el resultado en el orden del conjunto."""
    if ds == "imagenet":
        import bc_redes
        arch = os.path.join(RAIZ, "imagenet1k", "data", f"validation-{i:05d}-of-{NSHARD:05d}.parquet")
        dl = torch.utils.data.DataLoader(bc_redes.ValUno(arch, tf), batch_size=bloque, num_workers=WORKERS, worker_init_fn=bc_redes.trabajador)
    elif ds == "pool":
        dl = torch.utils.data.DataLoader(DatosPool(i, tf), batch_size=bloque, num_workers=WORKERS)
    else:
        dl = torch.utils.data.DataLoader(DatosExt(H, ds, tf), batch_size=bloque, num_workers=WORKERS)
    out, ix = [], []
    with torch.no_grad():
        for x, _, k in dl:
            out.append(f(x.to(dev)))
            ix.append(k)
    o = torch.argsort(torch.cat(ix))
    if isinstance(out[0], dict):
        return {c: torch.cat([d[c] for d in out])[o] for c in out[0]}
    return torch.cat(out)[o]


def carga_red_atlas(red, H, dev):
    """(transform, f) de una red del atlas (resnet50, vitb16, clipvitb16) con TODAS sus capas, para calcular sus activaciones sobre el pool (mismo agrupado que el atlas)."""
    import bc_redes
    r = bc_redes.Red(red, RAIZ, dev)

    def f(x):
        r.forward(x)
        return {c["id"]: r.pooled(c["id"]).float().cpu().half() for c in r.capas}
    return r.transform, f


def carga_red_ext(red, H, dev):
    """(transform, f) de una red de EXT; f(x) -> {capa: [B, U] fp16} con el agrupado del paper (promedio espacial en convolucionales, token CLS en ViT)."""
    import torchvision
    from torchvision import transforms as T
    os.environ.setdefault("TORCH_HOME", os.path.join(H["pesos"], "torch"))
    cap = {}
    if red == "vitl32":
        w = torchvision.models.ViT_L_32_Weights.IMAGENET1K_V1
        m = torchvision.models.vit_l_32()
        m.load_state_dict(torch.load(os.path.join(H["pesos"], "vit_l_32-IMAGENET1K_V1.pth"), map_location="cpu"))
        tf = w.transforms()
        m.encoder.layers[23].mlp[1].register_forward_hook(lambda mod, i, o: cap.__setitem__("bloque23_mlp", o[:, 0, :]))
    elif red == "resnet18_places":
        m = torchvision.models.resnet18(num_classes=365)
        sd = torch.load(os.path.join(H["pesos"], "resnet18_places365.pth.tar"), map_location="cpu")["state_dict"]
        m.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
        tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        for n in ("layer1", "layer2", "layer3", "layer4"):
            getattr(m, n).register_forward_hook(lambda mod, i, o, n=n: cap.__setitem__(n, o.mean((2, 3))))
    else:  # vgg16_cifar
        m = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=True)
        tf = T.Compose([T.Resize(32), T.ToTensor(), T.Normalize([0.5070, 0.4865, 0.4409], [0.2673, 0.2564, 0.2761])])
        m.features[43].register_forward_hook(lambda mod, i, o: cap.__setitem__("features_43", o[:, :, 0, 0]))
        m.classifier[4].register_forward_hook(lambda mod, i, o: cap.__setitem__("classifier_4", o))
    m = m.eval().to(dev)

    def f(x):
        cap.clear()
        m(x)
        return {c: v.float().cpu().half() for c, v in cap.items()}
    return tf, f


# ---- modelos de similitud
def carga_modelo_sim(nombre, H, dev):
    import torchvision
    from torchvision import transforms as T
    os.environ.setdefault("HF_HOME", os.path.join(H["pesos"], "hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if nombre == "resnet101":
        m = torchvision.models.resnet101()
        m.load_state_dict(torch.load(os.path.join(H["pesos"], "resnet101-DEFAULT.pth"), map_location="cpu"))
        tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return m.eval().to(dev), tf, lambda m, x: m(x)  # los 1000 logits, como el cuaderno
    from transformers import ViTModel
    m = ViTModel.from_pretrained("google/vit-base-patch16-224", cache_dir=os.path.join(H["pesos"], "hf"))
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])  # ViTImageProcessor: 224x224, media/desv 0.5
    return m.eval().to(dev), tf, lambda m, x: m(pixel_values=x).last_hidden_state[:, 0]


def extrae_sim_shard(nombre, H, i, dev, ds="imagenet"):
    """Caracteristicas de similitud (fp16) de un shard de ImageNet-val (o de todo otro conjunto), en el orden del conjunto."""
    m, tf, f = carga_modelo_sim(nombre, H, dev)
    return recorre(H, ds, i, tf, lambda x: f(m, x).float().cpu().half(), dev)


def ruta_sim(H, nombre, i, ds="imagenet"):
    return os.path.join(H["cache"], f"sim_{nombre}_{i:02d}.pt" if ds == "imagenet" else f"sim_{nombre}_pool_{i:02d}.pt" if ds == "pool" else f"sim_{nombre}_{ds}.pt")


def carga_sim(H, nombre, shards, ds="imagenet"):
    return normaliza(torch.cat([torch.load(ruta_sim(H, nombre, i, ds)) for i in shards]))


# ====================================================================== plan
class Plan:
    def __init__(self, a):
        self.a = a
        self.H = rutas("HUMO" if a.plan == "humo" else a.salida)
        self.shards_im = [0] if a.plan == "humo" else list(range(NSHARD))  # trozos de ImageNet-val (o del pool) en uso
        self.Cs = sorted({int(c) for c in a.C.split(",")})
        self.k = a.k  # 0 = el de cada red (k_de)
        self.metodos = [m for m in a.metodos.split(",") if m]
        for m in self.metodos:
            assert m in METODOS, f"metodo {m!r} desconocido: {METODOS}"
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"

    def shards_de(self, red):
        return self.shards_im if ds_de(red) in ("imagenet", "pool") else [0]

    def k_de(self, red):
        k = self.a.k or EXT.get(red, {}).get("k", KFULL)
        return min(k, 300) if self.a.plan == "humo" else k

    def capas_de(self, red):
        if red in EXT:
            cs = [c for c, _ in EXT[red]["capas"]]
        else:
            import bc_redes
            cs = [c["id"] for c in bc_redes.capas(red)]
        sel = [c for c in self.a.capas.split(",") if c] if self.a.capas else cs
        for c in sel:
            assert c in cs, f"{red}: capa {c!r} no esta en {cs}"
        return sel

    def neuronas(self, red, capa, U):
        a = self.a
        if a.neuronas:
            p = [int(x) if x else None for x in a.neuronas.split(":")]
            sel = list(range(p[0] or 0, p[1] if len(p) > 1 and p[1] is not None else U, p[2] if len(p) > 2 and p[2] else 1))
        elif a.muestra:
            sel = list(range(0, U, max(1, U // a.muestra)))[:a.muestra]  # las mismas que bc_line/bc_cosy --muestra K
        else:
            sel = list(range(U))
        import bc_inactivas as inact
        if a.estado != "todas":
            try:
                sel = inact.filtra(sel, red, capa, a.estado, atlas_dir=self.H["atlas_res"])
            except FileNotFoundError:
                print(f"  AVISO: sin topk de {red}/{capa}; --estado {a.estado} no se aplica", flush=True)
        return sel

    def estados(self, red, capa):
        try:
            import bc_inactivas as inact
            return inact.carga_estados(red, capa, atlas_dir=self.H["atlas_res"])
        except Exception:
            return {}

    def celdas(self):
        """[(red, capa, metodo)] en orden; el modelo de similitud (shards) va antes, como celda ('sim', nombre, shard)."""
        out = []
        for red in self.a.red.split(","):
            sim, ds = SIM_DE[red], ds_de(red)
            for i in self.shards_de(red):
                if not os.path.exists(ruta_sim(self.H, sim, i, ds)):
                    out.append(("sim", sim, i, ds))
                if (red in EXT or ds == "pool") and not os.path.exists(ruta_act(self.H, red, i)):
                    out.append(("act", red, i, ds))
            for capa in self.capas_de(red):
                for m in self.metodos:
                    out.append((red, capa, m))
        return list(dict.fromkeys(out))

    def pendiente(self, c, U_de):
        if c[0] == "sim":
            return not os.path.exists(ruta_sim(self.H, c[1], c[2], c[3]))
        if c[0] == "act":
            return not os.path.exists(ruta_act(self.H, c[1], c[2]))
        red, capa, m = c
        sel = self.neuronas(red, capa, U_de(red, capa))
        return any(len(lee_hechas(ruta_salida(self.H, red, capa, m, C, self.k_de(red))) & set(sel)) < len(sel) for C in self.Cs)


def U_de(red, capa):
    if red in EXT:
        return next(u for c, u in EXT[red]["capas"] if c == capa)
    import bc_redes
    return next(c["u"] for c in bc_redes.capas(red) if c["id"] == capa)


def fila(red, capa, metodo, C, k, u, corr_val, corr_test, idx, acts, sim, tipo, estado, seg, media=None, fuente="imagenet"):
    cfg = config_de(metodo, C, k)
    return dict(neurona=int(u), hipotesis=f"tabla de {C} ejemplos", puntaje=float(corr_test), longitud=C, uid=f"{red}/{capa}/{u}/{metodo}/{cfg}",
                extra=dict(metrica="gtx_corr_test" if metodo == "gtx" else f"{metodo}_corr_test", corr_val=float(corr_val), corr_test=float(corr_test), C=C, k=k,
                           indices=[int(i) for i in idx], activaciones=[float(x) for x in acts], similitudes_modelo=sim, particion=tipo,
                           estado_atlas=estado, segundos=round(seg, 4), fuente=fuente, media_train=None if media is None else float(media)),
                red=red, capa=capa, explicador=metodo, config=cfg)


def agrega(ruta, filas):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "a", encoding="utf-8") as f:
        for r in filas:
            f.write(json.dumps(r) + "\n")


class Contexto:
    """Carga perezosa y con cache de lo que comparten las celdas de un eslabon."""

    def __init__(self, plan):
        self.p = plan
        self._part = {}
        self._fn = {}
        self._act = (None, None)
        self._km = {}
        self.img_s = None

    def part(self, red):
        ds = ds_de(red)
        if ds not in self._part:
            self._part[ds] = (particion_le(self.p.H, self.p.shards_im) if ds == "imagenet" else particion_pool(self.p.shards_im) if ds == "pool"
                              else particion_ext(ds, EXT[red]["n"]))
            print(f"  particion {ds} {self._part[ds][1]}: " + " ".join(f"{k}={len(v)}" for k, v in self._part[ds][0].items()), flush=True)
        return self._part[ds]

    def fn(self, red):
        sim, ds = SIM_DE[red], ds_de(red)
        if (sim, ds) not in self._fn:
            self._fn[(sim, ds)] = carga_sim(self.p.H, sim, self.p.shards_de(red), ds).to(self.p.dev)
        return self._fn[(sim, ds)]

    def act(self, red, capa):
        if self._act[0] != (red, capa):
            self._act = ((red, capa), carga_act(self.p.H, red, capa, self.p.shards_de(red)))
        return self._act[1]

    def reps(self, clave, X_train_idx, X, k):
        """Representantes del k-means (indices globales), con cache en disco."""
        k = min(k, len(X_train_idx))
        ruta = os.path.join(self.p.H["cache"], f"km_{clave}_k{k}_n{len(X_train_idx)}_s{self.p.a.semilla}_i{self.p.a.kmeans_iters}.pt")
        if os.path.exists(ruta):
            return torch.load(ruta)
        t0 = time.time()
        Xt = X[X_train_idx.to(X.device)]
        C, lab = kmeans(Xt, k, self.p.a.kmeans_iters, self.p.a.semilla)
        r = X_train_idx[representantes(Xt, C, lab).cpu()]
        os.makedirs(self.p.H["cache"], exist_ok=True)
        torch.save(r, ruta)
        print(f"  k-means {clave}: k={k} sobre {len(X_train_idx)} -> {len(r)} representantes en {time.time() - t0:.0f} s", flush=True)
        return r


# ====================================================================== celdas
def celda_sim(ctx, c, quedan):
    _, nombre, i, ds = c
    t0 = time.time()
    os.makedirs(ctx.p.H["cache"], exist_ok=True)
    f = extrae_sim_shard(nombre, ctx.p.H, i, ctx.p.dev, ds)
    torch.save(f, ruta_sim(ctx.p.H, nombre, i, ds))
    ctx.img_s = len(f) / max(time.time() - t0, 1e-6)
    print(f"  sim {nombre} {ds} shard {i}: {len(f)} imgs, {ctx.img_s:.0f} img/s", flush=True)
    return True


def celda_act(ctx, c, quedan):
    """Activaciones de una red de EXT sobre su conjunto, en el orden del conjunto, fp16: mismo formato que atlas/cache/act_*.pt."""
    _, red, i, ds = c
    t0 = time.time()
    tf, f = (carga_red_ext if red in EXT else carga_red_atlas)(red, ctx.p.H, ctx.p.dev)
    d = recorre(ctx.p.H, ds, i, tf, f, ctx.p.dev)
    os.makedirs(ctx.p.H["cache"], exist_ok=True)
    torch.save(d, ruta_act(ctx.p.H, red, i))
    n = len(next(iter(d.values())))
    print(f"  act {red} {ds} shard {i}: {n} imgs, {n / max(time.time() - t0, 1e-6):.0f} img/s; capas: " + " ".join(f"{k}{tuple(v.shape)}" for k, v in d.items()), flush=True)
    return True


def tablas_neuronales(ctx, red, capa, metodo, Cs, units, A, Fn, tr, va):
    """{C: idx [B, C] globales} para un lote de neuronas, de los metodos que dependen de la neurona (A: activaciones de la capa, [N, U])."""
    a = ctx.p.a
    dev = Fn.device
    if metodo in ("gtx", "gtxpropio"):
        if metodo == "gtx":
            reps = ctx.reps(f"{SIM_DE[red]}_{ds_de(red)}", tr, Fn, ctx.p.k_de(red))
            Fv = Fn
        else:
            Fv = ctx.propio(red, capa)
            reps = ctx.reps(f"propio_{red}_{capa}", tr, Fv, ctx.p.k_de(red))
        S = Fv[reps.to(dev)] @ Fv[va.to(dev)].T
        aR = A[reps.to(dev)][:, units].T.contiguous()
        u = A[va.to(dev)][:, units].T.contiguous()
        sel, _ = voraz_lote(S, aR, u, max(Cs))
        glob = reps.to(dev)[sel]
        return {C: glob[:, :C] for C in Cs}
    At = A[tr.to(dev)][:, units]  # [Ntr, B]
    trd = tr.to(dev)
    out = {}
    for C in Cs:
        if metodo == "topk":
            out[C] = trd[At.topk(C, dim=0).indices.T]
        elif metodo == "extremos":
            hi = (C + 1) // 2
            out[C] = torch.cat([trd[At.topk(hi, dim=0).indices.T], trd[(-At).topk(C - hi, dim=0).indices.T]], 1)
    return out


def celda_metodo(ctx, c, quedan):
    red, capa, metodo = c
    p, a, H = ctx.p, ctx.p.a, ctx.p.H
    dev, Cs, k = p.dev, p.Cs, p.k_de(red)
    part, tipo, _ = ctx.part(red)
    tr, va, te = part["train"], part["val"], part["test"]
    sim = SIM_DE[red]
    Fn = ctx.fn(red)
    A = ctx.act(red, capa).to(dev)
    U = A.shape[1]
    sel = p.neuronas(red, capa, U)
    estados = p.estados(red, capa)
    hechas = {C: lee_hechas(ruta_salida(H, red, capa, metodo, C, k)) for C in Cs}
    pend = [u for u in sel if any(u not in hechas[C] for C in Cs)]
    print(f"{red}/{capa}/{metodo}: {len(pend)} de {len(sel)} neuronas pendientes", flush=True)
    if not pend:
        return True
    lote = a.lote
    ted, vad = te.to(dev), va.to(dev)
    # --- tablas que no dependen de la neurona: se calculan una vez
    indep = None
    if metodo in INDEP or metodo == "aleatorio":
        g = torch.Generator().manual_seed(a.semilla)
        sub = tr[torch.randperm(len(tr), generator=g)[:a.sub]]
        Fs = Fn[sub.to(dev)].cpu()
        if metodo == "aleatorio":
            indep = {C: [tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(1000 * s + C))[:C]] for s in range(a.semillas)] for C in Cs}
        else:
            f = {"kmedoids": sel_kmedoids, "kcenter": sel_kcenter, "dpp": lambda F, C, seed=0: sel_dpp(F, C)}[metodo]
            indep = {C: [sub[f(Fs, C, a.semilla)]] for C in Cs}
    Fv = ctx.propio(red, capa) if metodo == "gtxpropio" else Fn
    Fvv, Fvt = Fv[vad], Fv[ted]
    trd0 = tr.to(dev)
    mu_tr = torch.stack([A[trd0[i:i + 20000]].sum(0) for i in range(0, len(trd0), 20000)]).sum(0) / len(trd0)  # media de cada neurona en train (para colorear sobre/bajo el promedio)
    t_lote = 0.0
    for b0 in range(0, len(pend), lote):
        if quedan() < max(a.reserva, 2 * t_lote):
            print("  se acaba el eslabon: se reanuda en el siguiente", flush=True)
            return False
        t0 = time.time()
        units = pend[b0:b0 + lote]
        ud = torch.tensor(units, device=dev)
        B = len(units)
        if indep is None:
            tabs = tablas_neuronales(ctx, red, capa, metodo, Cs, ud, A, Fn, tr, va)
            reps_ = {C: [tabs[C]] for C in Cs}  # [1][B, C]
        else:
            reps_ = {C: [t.to(dev)[None].expand(B, -1) for t in indep[C]] for C in Cs}
        uv = A[vad[:, None], ud[None, :]].T
        ut = A[ted[:, None], ud[None, :]].T
        for C in Cs:
            cv, ct, idx_fin, acts_fin = 0.0, 0.0, None, None
            for idx in reps_[C]:
                acts_sel = A[idx, ud[:, None]]  # [B, C]: activacion de CADA neurona en SUS imagenes
                cv = cv + corr_col(pred_pesada(Fv, idx, acts_sel, Fvv), uv) / len(reps_[C])
                ct = ct + corr_col(pred_pesada(Fv, idx, acts_sel, Fvt), ut) / len(reps_[C])
                if idx_fin is None:
                    idx_fin, acts_fin = idx, acts_sel
            seg = (time.time() - t0) / B
            filas = [fila(red, capa, metodo, C, k, units[j], cv[j], ct[j], idx_fin[j].tolist(), acts_fin[j].tolist(), sim if metodo != "gtxpropio" else "activaciones_propias",
                          tipo, estados.get(units[j], "viva"), seg, mu_tr[units[j]], ds_de(red)) for j in range(B) if units[j] not in hechas[C]]
            agrega(ruta_salida(H, red, capa, metodo, C, k), filas)
            hechas[C] |= set(units)
        t_lote = max(t_lote, time.time() - t0)
        if (b0 // lote) % 10 == 0:
            print(f"  {red}/{capa}/{metodo}: {min(b0 + lote, len(pend))}/{len(pend)}  {t_lote:.1f} s/lote (B={B})", flush=True)
    return True


def _propio(self, red, capa):
    """Vectores de similitud 'propios' = activaciones de la capa centradas y normalizadas (fuga deliberada)."""
    if self._km.get(("propio", red, capa)) is None:
        A = self.act(red, capa)
        self._km = {("propio", red, capa): normaliza(A - A.mean(0, keepdim=True)).to(self.p.dev)}
    return self._km[("propio", red, capa)]


Contexto.propio = _propio


# ====================================================================== ETA (humo)
def eta(plan, ctx_img_s, t_carga):
    """Proyecta el tiempo de la corrida completa midiendo cada primitiva a TAMANO COMPLETO con datos sinteticos. No ejecuta la corrida."""
    a, dev = plan.a, plan.dev
    k, Mv, Mt, Ntr, B, Cmax = KFULL, 5000, 10000, 35000, a.lote, max(plan.Cs)
    sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)
    g = torch.Generator(device="cpu").manual_seed(0)
    rnd = lambda *s: torch.randn(*s, generator=g).to(dev)

    def cron(f, rep=1):
        f()
        sync()
        t = time.time()
        for _ in range(rep):
            f()
        sync()
        return (time.time() - t) / rep

    t = {}
    S, aR, u = normaliza(rnd(k, 64)) @ normaliza(rnd(Mv, 64)).T, rnd(B, k).abs(), rnd(B, Mv)
    t["voraz_lote"] = cron(lambda: voraz_lote(S, aR, u, Cmax))
    for nombre, D in (("resnet101", 1000), ("vitb16hf", 768)):
        X = normaliza(rnd(Ntr, D))
        t1 = cron(lambda: kmeans(X, k, 1, 0))
        t3 = cron(lambda: kmeans(X, k, 3, 0))
        t[f"kmeans_iter_{nombre}"] = max((t3 - t1) / 2, 1e-6)
        t[f"kmeans_inicio_{nombre}"] = max(t1 - t[f"kmeans_iter_{nombre}"], 0.0)
    Fn = normaliza(rnd(Ntr + Mv + Mt, 1000))
    idx = torch.randint(0, Ntr, (B, max(plan.Cs)), device=dev)
    acts = rnd(B, idx.shape[1])
    t["eval_lote_(val+test)"] = cron(lambda: (pred_pesada(Fn, idx, acts, Fn[:Mv]), pred_pesada(Fn, idx, acts, Fn[Mv:Mv + Mt])))
    Fs = normaliza(rnd(a.sub, 1000)).cpu()
    t["kmedoids_(1 vez/celda)"] = cron(lambda: sel_kmedoids(Fs, max(plan.Cs)))
    t["dpp_(1 vez/celda)"] = cron(lambda: sel_dpp(Fs, max(plan.Cs)))
    print("\n--- ETA de la corrida completa (ESTIMACION: primitivas medidas a tamano completo con datos sinteticos; NO se midio la corrida real) ---", flush=True)
    print(f"dispositivo: {dev}   k(k-means)={k} val={Mv} test={Mt} train={Ntr} lote={B} Cmax={Cmax} metodos={','.join(plan.metodos)}")
    for n, v in t.items():
        print(f"  {n:28s} {v:9.3f} s")
    por_neurona = {"gtx": t["voraz_lote"] / B + len(plan.Cs) * t["eval_lote_(val+test)"] / B}
    for m in plan.metodos:
        por_neurona.setdefault(m, len(plan.Cs) * t["eval_lote_(val+test)"] / B * (a.semillas if m == "aleatorio" else 1))
    por_neurona["gtxpropio"] = por_neurona["gtx"]
    print("  segundos por neurona: " + "  ".join(f"{m}={v:.4f}" for m, v in por_neurona.items() if m in plan.metodos))
    tot, filas = 0.0, []
    for red in a.eta_redes.split(","):
        sim = SIM_DE[red]
        n_img = EXT[red]["n"] if red in EXT else 50000
        t_feat = n_img * (2 if red in EXT else 1) / max(ctx_img_s, 1e-9) if ctx_img_s else float("nan")  # EXT: pasada de similitud + pasada de activaciones (se toma la misma velocidad)
        t_red = t_feat
        for U in TAMS_ETA[red]:
            nn = min(U, a.muestra_eta) if a.muestra_eta else U
            km = t[f"kmeans_inicio_{sim}"] + a.kmeans_iters * t[f"kmeans_iter_{sim}"]
            c = t_carga + km + nn * sum(por_neurona[m] for m in plan.metodos if m in por_neurona) \
                + (km if "gtxpropio" in plan.metodos else 0) + (t["kmedoids_(1 vez/celda)"] + t["dpp_(1 vez/celda)"] if True else 0)
            t_red += c
        filas.append((red, sum(TAMS_ETA[red]), t_feat, t_red))
        tot += t_red
    print(f"\n  {'red':12s}{'neuronas':>10s}{'caract.sim (s)':>16s}{'total (h)':>11s}{'eslabones de 25 min':>22s}")
    for red, n, tf, tr_ in filas:
        print(f"  {red:12s}{n:>10d}{tf:>16.0f}{tr_ / 3600:>11.2f}{math.ceil(tr_ / (25 * 60 * 0.9)):>22d}")
    print(f"  TOTAL        {'':>10s}{'':>16s}{tot / 3600:>11.2f}{math.ceil(tot / (25 * 60 * 0.9)):>22d}   (neuronas por capa: {a.muestra_eta or 'todas'}; eslabon util = 90 % de 25 min)")
    print("  Supuestos: velocidad de extraccion = la medida en el humo sobre imagenes reales (si no hay, nan); carga de activaciones = la medida en el humo por capa;")
    print("  no incluye reintentos ni espera en cola. Si el dispositivo es 'cpu' estos numeros NO sirven para la GPU.\n", flush=True)


# ====================================================================== resumen
def resumen(plan):
    H = plan.H
    le_dir = H["atlas_res"]
    print(f"{'red/capa':32s}{'metodo':11s}{'C':>3s}{'n':>6s}{'test':>8s}{'val':>8s}{'LE':>8s}{'gana a LE':>11s}")
    for red in plan.a.red.split(","):
        for capa in plan.capas_de(red):
            le = {}
            f = os.path.join(le_dir, f"{red}__{capa}__le__siglip.jsonl")
            if os.path.exists(f):
                for l in open(f, encoding="utf-8"):
                    if l.strip():
                        r = json.loads(l)
                        le[r["neurona"]] = r["puntaje"]
            for m in plan.metodos:
                for C in plan.Cs:
                    f = ruta_salida(H, red, capa, m, C, plan.k_de(red))
                    if not os.path.exists(f):
                        continue
                    rs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
                    if not rs:
                        continue
                    t = np.mean([r["puntaje"] for r in rs])
                    v = np.mean([r["extra"]["corr_val"] for r in rs])
                    com = [(r["puntaje"], le[r["neurona"]]) for r in rs if r["neurona"] in le]
                    ml = np.mean([y for _, y in com]) if com else float("nan")
                    gana = np.mean([x > y for x, y in com]) if com else float("nan")
                    print(f"{red + '/' + capa:32s}{m:11s}{C:>3d}{len(rs):>6d}{t:>8.4f}{v:>8.4f}{ml:>8.4f}{gana:>11.2f}")


# ====================================================================== mapas de calor de las tablas GTX
LADO_MAX = 28  # los mapas mas grandes (ResNet layer1: 56x56) se reducen con promedio a 28x28, como en el atlas
K_MAPA = 10  # imagenes de la tabla con mapa: la tabla de C = 10 (la de C = 5 es su prefijo)


def inicio_trozo(ds, i):
    """Indice global de la primera imagen del trozo i: suma de las filas de los parquet anteriores (ImageNet-val) o primer pid del trozo (pool)."""
    if ds == "pool":
        return limites_pool(i)[0]
    import pyarrow.parquet as pq
    return sum(pq.ParquetFile(os.path.join(RAIZ, "imagenet1k", "data", f"validation-{j:05d}-of-{NSHARD:05d}.parquet")).metadata.num_rows for j in range(i))


def pares_mapas(H, red, k):
    """{capa: dict(g=indice global de cada (neurona, imagen de la tabla) aplanado [U*K], u=neurona de cada par, U, K, idx [U, K])} para las capas con mapa espacial
    (las espaciales y sus gemelas tokmean). Sale de las tablas GTX de C = K_MAPA de la fuente actual; -1 = neurona sin tabla."""
    import bc_redes
    out = {}
    for c in bc_redes.capas(red):
        if not (c["tipo"] == "espacial" or c["id"].endswith("_tokmean")):
            continue
        ruta = ruta_salida(H, red, c["id"], "gtx", K_MAPA, k)
        if not os.path.exists(ruta):
            continue
        U = c["u"]
        idx = torch.full((U, K_MAPA), -1, dtype=torch.long)
        for l in open(ruta, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                idx[r["neurona"], :len(r["extra"]["indices"])] = torch.tensor(r["extra"]["indices"])
        out[c["id"]] = dict(g=idx.reshape(-1), u=torch.arange(U).repeat_interleave(K_MAPA), U=U, K=K_MAPA, idx=idx)
    return out


def ruta_mapa_trozo(H, ds, red, i):
    return os.path.join(H["gtx"], "mapas_raw", ds, f"{red}_{i:02d}.pt")


def ruta_mapa_final(H, ds, red, capa):
    return os.path.join(H["gtx"], "mapas", ds, f"{red}__{capa}.npz")


def mapas_trozo(plan, red, i, ds):
    """Una pasada por el trozo i: para cada par (neurona, imagen de su tabla) guarda el mapa espacial de la activacion de la unidad (fp16, sin normalizar)."""
    import bc_redes
    import torch.nn.functional as F
    dev = plan.dev
    pares = pares_mapas(plan.H, red, plan.k_de(red))
    if not pares:
        raise RuntimeError(f"{red}: no hay tablas GTX de C={K_MAPA} en {plan.H['gtx_dir']}: corre antes --metodos gtx --C {K_MAPA}")
    base = inicio_trozo(ds, i)
    r = bc_redes.Red(red, RAIZ, dev)
    if ds == "pool":
        dl = torch.utils.data.DataLoader(DatosPool(i, r.transform), batch_size=64, num_workers=WORKERS)
    else:
        arch = os.path.join(RAIZ, "imagenet1k", "data", f"validation-{i:05d}-of-{NSHARD:05d}.parquet")
        dl = torch.utils.data.DataLoader(bc_redes.ValUno(arch, r.transform), batch_size=64, num_workers=WORKERS, worker_init_fn=bc_redes.trabajador)
    g = {cid: p["g"].to(dev) for cid, p in pares.items()}
    uu = {cid: p["u"].to(dev) for cid, p in pares.items()}
    acum = {cid: dict(sel=[], raw=[], nat=None) for cid in pares}
    with torch.no_grad():
        for x, _, k in dl:
            r.forward(x.to(dev))
            gb = k.to(dev).long() + base
            for cid in pares:
                m = torch.isin(g[cid], gb)
                if not bool(m.any()):
                    continue
                sel = m.nonzero().flatten()
                pos = (g[cid][sel][:, None] == gb[None, :]).float().argmax(1)  # posicion en el lote de la imagen de cada par
                sub = r.espacial(cid)[pos, uu[cid][sel]].float()  # [n, h, w]
                if acum[cid]["nat"] is None:
                    acum[cid]["nat"] = tuple(sub.shape[-2:])
                if sub.shape[-1] > LADO_MAX:
                    sub = F.adaptive_avg_pool2d(sub[:, None], LADO_MAX)[:, 0]
                acum[cid]["sel"].append(sel.cpu())
                acum[cid]["raw"].append(sub.half().cpu())
    out = {cid: dict(sel=torch.cat(a["sel"]) if a["sel"] else torch.zeros(0, dtype=torch.long), raw=torch.cat(a["raw"]) if a["raw"] else None, nat=a["nat"]) for cid, a in acum.items()}
    ruta = ruta_mapa_trozo(plan.H, ds, red, i)
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    torch.save(out, ruta + ".tmp")
    os.replace(ruta + ".tmp", ruta)
    print(f"  mapas {red} {ds} trozo {i:02d}: {sum(len(a['sel']) for a in out.values())} pares", flush=True)


def mapas_final(plan, red, ds):
    """Junta los trozos y escribe un npz por capa (m uint8 [U, K, s, s] normalizado por el maximo de los K mapas de la neurona, parte positiva; ref; indices para verificar la alineacion)."""
    import bc_redes
    pares = pares_mapas(plan.H, red, plan.k_de(red))
    acc = {cid: dict(raw=None, hecho=torch.zeros(p["g"].numel(), dtype=torch.bool), nat=None) for cid, p in pares.items()}
    for i in plan.shards_de(red):
        d = torch.load(ruta_mapa_trozo(plan.H, ds, red, i))
        for cid, p in pares.items():
            if d[cid]["raw"] is None:
                continue
            a = acc[cid]
            if a["raw"] is None:
                a["raw"] = torch.zeros(p["g"].numel(), *d[cid]["raw"].shape[-2:], dtype=torch.float16)
                a["nat"] = d[cid]["nat"]
            a["raw"][d[cid]["sel"]] = d[cid]["raw"]
            a["hecho"][d[cid]["sel"]] = True
    os.makedirs(os.path.join(plan.H["gtx"], "mapas", ds), exist_ok=True)
    for cid, p in pares.items():
        a = acc[cid]
        falta = int(((p["g"] >= 0) & ~a["hecho"]).sum())
        if a["raw"] is None or falta:
            raise RuntimeError(f"mapas {red}/{cid}: faltan {falta} de {int((p['g'] >= 0).sum())} pares (imagenes no encontradas en los trozos procesados)")
        raw = a["raw"].float().clamp_min(0).view(p["U"], p["K"], *a["raw"].shape[-2:])
        ref = raw.amax((1, 2, 3)).clamp_min(1e-8)
        m8 = (255.0 * raw / ref[:, None, None, None]).round().clamp(0, 255).to(torch.uint8)
        ruta = ruta_mapa_final(plan.H, ds, red, cid)
        np.savez_compressed(ruta + ".tmp.npz", m=m8.numpy(), ref=ref.numpy(), lado=m8.shape[-1], nativo=np.array(a["nat"]), cobertura=bc_redes.cobertura_mapa(red),
                            indices=p["idx"].numpy(), C=K_MAPA)
        os.replace(ruta + ".tmp.npz", ruta)
        print(f"  mapas {red}/{cid}: {tuple(m8.shape)} uint8 -> {os.path.basename(ruta)}", flush=True)


def plan_mapas(plan, quedan):
    """Plan 'mapas': mapas de calor de las imagenes de las tablas GTX (C=10) de la fuente actual, un trozo por celda (reanudable) y un ensamblado final por red."""
    a, ds = plan.a, ("pool" if FUENTE == "pool" else "imagenet")
    reds = a.red.split(",")
    pend = []
    for red in reds:
        pend += [("trozo", red, i) for i in plan.shards_de(red) if not os.path.exists(ruta_mapa_trozo(plan.H, ds, red, i))]
        import bc_redes
        capas = [c["id"] for c in bc_redes.capas(red) if c["tipo"] == "espacial" or c["id"].endswith("_tokmean")]
        if any(not os.path.exists(ruta_mapa_final(plan.H, ds, red, c)) for c in capas):
            pend.append(("fin", red, None))
    print(f"celdas de mapas ({ds}): {len(pend)} pendientes", flush=True)
    if a.existe:
        sys.exit(0 if not pend else 1)
    for tipo, red, i in pend:
        if quedan() < a.reserva + 120:
            print("  se acaba el eslabon: se reanuda en el siguiente", flush=True)
            return
        if tipo == "trozo":
            mapas_trozo(plan, red, i, ds)
        else:
            mapas_final(plan, red, ds)
    print("mapas: plan completo", flush=True)


def miniaturas_pool(a):
    """Para el sitio publico: miniaturas de 112 px y creditos de las imagenes del pool que aparecen en alguna tabla GTX (cosy/gtx_pool). Salida en gtx/: pool_gtx_thumbs.pkl y pool_gtx_creditos.jsonl."""
    import io
    import pickle
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image, ImageOps
    H = rutas()
    filas = pool_filas()
    usados = set()
    for f in glob.glob(os.path.join(H["gtx_dir"], "*__gtx__*.jsonl")):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                usados.update(json.loads(l)["extra"]["indices"])
    usados = sorted(usados)

    def mini(pid):
        b = io.BytesIO()
        ImageOps.fit(Image.open(os.path.join(POOL_DIR, filas[pid]["ruta"])).convert("RGB"), (112, 112), Image.BICUBIC).save(b, "JPEG", quality=80)
        return b.getvalue()

    with ThreadPoolExecutor(max_workers=8) as ex:
        thumbs = dict(zip(usados, ex.map(mini, usados)))
    os.makedirs(H["gtx"], exist_ok=True)
    with open(os.path.join(H["gtx"], "pool_gtx_thumbs.pkl"), "wb") as f:
        pickle.dump(thumbs, f)
    campos = ("pid", "pool", "orig_id", "autor", "autor_url", "titulo", "licencia", "landing", "enlace", "caption", "fuente", "etiquetas", "orig")
    with open(os.path.join(H["gtx"], "pool_gtx_creditos.jsonl"), "w", encoding="utf-8") as f:
        for pid in usados:
            f.write(json.dumps({k: filas[pid].get(k) for k in campos}, ensure_ascii=False) + "\n")
    print(f"miniaturas del pool para GTX: {len(usados)} imagenes unicas -> {H['gtx']}/pool_gtx_thumbs.pkl y pool_gtx_creditos.jsonl")


def descarga_ext(a):
    """EN EL NODO DE ACCESO (internet): pesos y datos de las redes de EXT que se piden en --red (por omision, las tres)."""
    import urllib.request
    import torchvision
    H = rutas()
    os.makedirs(H["pesos"], exist_ok=True)
    os.environ["TORCH_HOME"] = os.path.join(H["pesos"], "torch")
    reds = [r for r in a.red.split(",") if r in EXT] or list(EXT)
    for r in reds:
        if r == "vitl32":
            torch.save(torchvision.models.vit_l_32(weights=torchvision.models.ViT_L_32_Weights.IMAGENET1K_V1).state_dict(), os.path.join(H["pesos"], "vit_l_32-IMAGENET1K_V1.pth"))
        elif r == "resnet18_places":
            urllib.request.urlretrieve("http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar", os.path.join(H["pesos"], "resnet18_places365.pth.tar"))
            abre_ds(H, "places365", descarga=True)
        else:
            torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=True)
            abre_ds(H, "cifar100", descarga=True)
        print("descargado:", r, flush=True)
    for ds, nombre in SPLIT_LE.items():
        ok = all(os.path.exists(os.path.join(REPO_LE, "data", "data_splits", nombre, f"{k}_ids.pt")) for k in ("train", "val", "test"))
        print(f"particion de LE {nombre}: {'ENCONTRADA' if ok else 'NO ESTA en ' + REPO_LE + ' (se usara semilla fija 70/10/20; no comparable con LE)'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", choices=["descarga", "descarga_ext", "miniaturas_pool", "mapas", "humo", "evalua"], required=True)
    ap.add_argument("--fuente", choices=["imagenet", "pool"], default="imagenet", help="pool = el pool abierto (para el sitio publico); solo redes del atlas")
    ap.add_argument("--red", default="resnet50", help="resnet50,vitb16,clipvitb16 (separadas por coma)")
    ap.add_argument("--capas", default="", help="ids de capa de bc_redes.capas (por omision, todas las de cada red)")
    ap.add_argument("--neuronas", default="", help="a:b o a:b:paso")
    ap.add_argument("--muestra", type=int, default=0, help="K neuronas repartidas por capa (las mismas que bc_line/bc_cosy --muestra K)")
    ap.add_argument("--estado", default="todas", choices=["todas", "vivas", "muertas", "inhibitorias"])
    ap.add_argument("--metodos", default=",".join(METODOS))
    ap.add_argument("--C", default="5,10", help="longitudes de tabla; la voraz es anidada, una corrida da todas")
    ap.add_argument("--k", type=int, default=0, help="k del k-means previo (0 = el del paper: 2000 en ImageNet y Places365, 500 en CIFAR-100)")
    ap.add_argument("--kmeans-iters", type=int, default=40, dest="kmeans_iters")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--semillas", type=int, default=5, help="repeticiones del baseline aleatorio (se promedia)")
    ap.add_argument("--sub", type=int, default=8000, help="submuestra de train para kmedoids/kcenter/dpp")
    ap.add_argument("--lote", type=int, default=32, help="neuronas por lote en la voraz (memoria ~ lote*256*5000*4 B * ~6)")
    ap.add_argument("--reserva", type=float, default=90, help="segundos que deben quedar para empezar otro lote")
    ap.add_argument("--minutos", type=float, default=25)
    ap.add_argument("--eta-redes", default="resnet50,vitb16,clipvitb16,vitl32,resnet18_places,vgg16_cifar", dest="eta_redes")
    ap.add_argument("--salida", default=None)
    ap.add_argument("--exige-gpu", action="store_true", dest="exige_gpu")
    ap.add_argument("--existe", action="store_true")
    ap.add_argument("--resumen", action="store_true")
    a = ap.parse_args()
    global FUENTE
    FUENTE = a.fuente
    if FUENTE == "pool" and any(r in EXT for r in a.red.split(",")):
        sys.exit("--fuente pool solo vale para las redes del atlas (resnet50, vitb16, clipvitb16)")
    t0 = time.time()
    quedan = lambda: a.minutos * 60 - (time.time() - t0)

    if a.plan == "descarga":
        import torchvision
        H = rutas()
        os.makedirs(H["pesos"], exist_ok=True)
        os.environ["HF_HOME"] = os.path.join(H["pesos"], "hf")
        w = torchvision.models.ResNet101_Weights.DEFAULT
        torch.save(torchvision.models.resnet101(weights=w).state_dict(), os.path.join(H["pesos"], "resnet101-DEFAULT.pth"))
        from transformers import ViTModel
        ViTModel.from_pretrained("google/vit-base-patch16-224", cache_dir=os.path.join(H["pesos"], "hf"))
        print("descargado: resnet101-DEFAULT.pth y google/vit-base-patch16-224 en", H["pesos"])
        return
    if a.plan == "descarga_ext":
        return descarga_ext(a)
    if a.plan == "miniaturas_pool":
        FUENTE = "pool"
        return miniaturas_pool(a)
    if a.plan == "humo":
        a.red = a.red.split(",")[0]
        a.capas = a.capas or ("layer1" if a.red == "resnet50" else "")
        a.muestra_eta = a.muestra  # el ETA proyecta con esto (0 = todas las neuronas), no con las 16 del humo
        a.muestra = a.muestra or 16
    plan = Plan(a)
    if a.exige_gpu and plan.dev != "cuda":
        sys.exit("--exige-gpu: no hay GPU")
    print(f"entorno: cuda={plan.dev == 'cuda'} torch={torch.__version__} plan={a.plan} fuente={FUENTE} redes={a.red} C={plan.Cs} k={a.k or 'segun la red'}", flush=True)
    if a.resumen:
        return resumen(plan)
    if a.plan == "mapas":
        return plan_mapas(plan, quedan)
    cel = plan.celdas()
    pend = [c for c in cel if plan.pendiente(c, U_de)]
    print(f"celdas: {len(cel) - len(pend)} de {len(cel)} hechas", flush=True)
    if a.existe:
        sys.exit(0 if not pend else 1)
    ctx = Contexto(plan)
    t_carga = 0.0
    for c in pend:
        if quedan() < a.reserva:
            break
        t1 = time.time()
        ok = celda_sim(ctx, c, quedan) if c[0] == "sim" else celda_act(ctx, c, quedan) if c[0] == "act" else celda_metodo(ctx, c, quedan)
        if not ok:
            break
    if a.plan == "humo":
        t1 = time.time()
        _ = ctx.act(a.red, plan.capas_de(a.red)[0])
        t_carga = (time.time() - t1) * NSHARD / len(plan.shards_de(a.red)) if ds_de(a.red) == "imagenet" else 1  # carga de una capa con los 14 shards
        eta(plan, ctx.img_s, t_carga)
        resumen(plan)
    print(f"fin del eslabon: {(time.time() - t0) / 60:.1f} min; pendientes: {len([c for c in cel if plan.pendiente(c, U_de)])}", flush=True)


if __name__ == "__main__":
    main()
