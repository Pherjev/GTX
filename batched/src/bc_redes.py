#!/usr/bin/env python3
"""Registro de redes para el atlas de neuronas: carga, capas objetivo, activaciones agrupadas y mapas espaciales.

Una red es una clase `Red` con:
  .capas                lista de dict(id, tipo='espacial'|'vector', u=<unidades>)
  .forward(x)           ejecuta la red (los hooks llenan self.cap)
  .pooled(id)           activacion por imagen [B, U]   (resnet: promedio espacial; ViT/CLIP: token CLS)
  .espacial(id)         mapa [B, U, h, w]              (resnet: la salida; ViT/CLIP: los tokens de parche como cuadricula)
  .transform            preproceso PIL -> tensor propio de la red
Para anadir una red: agregarla a `capas()` y a `Red.__init__`.
"""
import io
import os
import sys

_pl = os.path.expanduser("~/clusters/pylibs")
if os.path.isdir(_pl):
    sys.path.insert(0, _pl)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyarrow.parquet as pq
import torch
import torchvision
from PIL import Image

PESOS_R50 = "pesos/resnet50-0676ba61.pth"
CHUNK = 256
REDES = ("resnet50", "vitb16", "clipvitb16")
BLOQUES = (2, 5, 8, 11)
TOKMEAN = (2, 5, 8)  # bloques que ademas se agrupan promediando todos los tokens (id "bloque{k}_mlp_tokmean")


def capas(red):
    """Capas objetivo. En ViT/CLIP, "bloque{k}_mlp" se agrupa con el token CLS y "bloque{k}_mlp_tokmean" con el promedio de tokens
    (como el repositorio de Linear Explanations en capas intermedias); el CLS de bloques tempranos casi no lleva senal de la imagen."""
    if red == "resnet50":
        return [dict(id="layer1", tipo="espacial", u=256), dict(id="layer2", tipo="espacial", u=512),
                dict(id="layer3", tipo="espacial", u=1024), dict(id="layer4", tipo="espacial", u=2048),
                dict(id="fc", tipo="vector", u=1000)]
    tm = [dict(id=f"bloque{k}_mlp_tokmean", tipo="vector", u=3072) for k in TOKMEAN]  # "vector": Network Dissection no aplica (mismos mapas)
    if red == "vitb16":
        return [dict(id=f"bloque{k}_mlp", tipo="espacial", u=3072) for k in BLOQUES] + [dict(id="ln_final", tipo="vector", u=768)] + tm
    if red == "clipvitb16":
        return [dict(id=f"bloque{k}_mlp", tipo="espacial", u=3072) for k in BLOQUES] + [dict(id="embedding", tipo="vector", u=512)] + tm
    raise ValueError(red)


def cobertura_mapa(red):
    """Fraccion del lado de la miniatura (recorte cuadrado de la imagen) que cubre el mapa espacial de la red.
    ResNet-50 y ViT (torchvision): Resize(256) + CenterCrop(224) -> 224/256 del lado; CLIP (open_clip): Resize(224) + CenterCrop(224) -> todo el lado."""
    return 1.0 if red == "clipvitb16" else 224.0 / 256.0


class Red:
    def __init__(self, nombre, raiz, dev):
        self.nombre, self.dev, self.cap, self.capas = nombre, dev, {}, capas(nombre)
        hook = lambda cid: (lambda mod, inp, out: self.cap.__setitem__(cid, out))
        if nombre == "resnet50":
            m = torchvision.models.resnet50()
            m.load_state_dict(torch.load(os.path.join(raiz, PESOS_R50), map_location="cpu"))
            self.transform = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms()
            for c in ("layer1", "layer2", "layer3", "layer4"):
                getattr(m, c).register_forward_hook(hook(c))
            m.fc.register_forward_hook(hook("fc"))
            self.m, self._fwd = m.eval().to(dev), m
        elif nombre == "vitb16":
            w = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
            m = torchvision.models.vit_b_16(weights=w)
            self.transform = w.transforms()
            for k in BLOQUES:
                m.encoder.layers[k].mlp[1].register_forward_hook(hook(f"bloque{k}_mlp"))  # salida de la GELU (3072)
            m.encoder.ln.register_forward_hook(hook("ln_final"))
            self.m, self._fwd = m.eval().to(dev), m
        elif nombre == "clipvitb16":
            import open_clip

            m, _, pre = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai", device=dev)
            self.transform = pre
            for k in BLOQUES:
                m.visual.transformer.resblocks[k].mlp.gelu.register_forward_hook(hook(f"bloque{k}_mlp"))
            self.m = m.eval()
            self._fwd = lambda x: self.cap.__setitem__("embedding", m.encode_image(x)) or self.cap["embedding"]
        else:
            raise ValueError(nombre)

    def forward(self, x):
        self.cap.clear()
        return self._fwd(x)

    @staticmethod
    def _nld(o):
        """CLIP puede dar [L, B, D] o [B, L, D]; se normaliza a [B, L, D] (L = 197 = CLS + 14x14)."""
        return o.transpose(0, 1) if (o.shape[0] == 197 and o.shape[1] != 197) else o

    def pooled(self, cid):
        if cid.endswith("_tokmean"):  # promedio sobre todos los tokens (incluido CLS) del mismo bloque que "bloque{k}_mlp"
            o = self.cap[cid[:-len("_tokmean")]]
            return (o if self.nombre == "vitb16" else self._nld(o)).mean(1)
        o = self.cap[cid]
        if self.nombre == "resnet50":
            return o.mean((2, 3)) if o.dim() == 4 else o
        if self.nombre == "vitb16":
            return o[:, 0] if o.dim() == 3 else o
        return o if cid == "embedding" else self._nld(o)[:, 0]

    def espacial(self, cid):
        if cid.endswith("_tokmean"):  # la capa tokmean comparte el mapa de parches con su bloque (solo cambia el agrupado)
            cid = cid[:-len("_tokmean")]
        o = self.cap[cid]
        if self.nombre == "resnet50":
            return o
        t = (o if self.nombre == "vitb16" else self._nld(o))[:, 1:, :]
        b, n, d = t.shape
        h = int(round(n ** 0.5))
        return t.transpose(1, 2).reshape(b, d, h, h)

    def extractor_espacial(self, cid):
        def f(x):
            self.forward(x)
            return self.espacial(cid)
        return f


# ---------------------------------------------------------------- ImageNet-val en parquet (orden determinista)
class ValUno(torch.utils.data.IterableDataset):
    def __init__(self, archivo, tf):
        self.archivo, self.tf, self.tareas, off = archivo, tf, [], 0
        md = pq.ParquetFile(archivo).metadata
        for g in range(md.num_row_groups):
            n = md.row_group(g).num_rows
            self.tareas += [(g, s, min(CHUNK, n - s), off + s) for s in range(0, n, CHUNK)]
            off += n

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        w, W = (info.id, info.num_workers) if info else (0, 1)
        for g, s, n, off in self.tareas[w::W]:
            tabla = pq.ParquetFile(self.archivo).read_row_group(g, columns=["image", "label"])
            for j, r in enumerate(tabla.slice(s, n).to_pylist()):
                im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
                yield self.tf(im), r["label"], off + j


def trabajador(_):
    torch.set_num_threads(1)


def extrae_shard(red, raiz, i, workers, bloque=64):
    """Activaciones agrupadas de todas las capas de `red` para un shard de ImageNet-val, en el orden del parquet. fp16."""
    arch = os.path.join(raiz, "imagenet1k", "data", f"validation-{i:05d}-of-00014.parquet")
    dl = torch.utils.data.DataLoader(ValUno(arch, red.transform), batch_size=bloque, num_workers=workers, worker_init_fn=trabajador)
    acc, ix = {c["id"]: [] for c in red.capas}, []
    with torch.no_grad():
        for x, _, k in dl:
            red.forward(x.to(red.dev))
            for c in red.capas:
                acc[c["id"]].append(red.pooled(c["id"]).float().cpu().half())
            ix.append(k)
    o = torch.argsort(torch.cat(ix))
    return {cid: torch.cat(v)[o] for cid, v in acc.items()}


def miniaturas_shard(raiz, i, workers, lado=112, calidad=80):
    """Miniaturas JPEG (recorte cuadrado) de un shard de ImageNet-val, en el orden del parquet: lista de bytes."""
    from PIL import ImageOps

    def mini(im):
        b = io.BytesIO()
        ImageOps.fit(im, (lado, lado), Image.BICUBIC).save(b, "JPEG", quality=calidad)
        return b.getvalue()

    arch = os.path.join(raiz, "imagenet1k", "data", f"validation-{i:05d}-of-00014.parquet")
    dl = torch.utils.data.DataLoader(ValUno(arch, mini), batch_size=64, num_workers=workers, worker_init_fn=trabajador)
    datos, ix = [], []
    for x, _, k in dl:
        datos += list(x)
        ix.append(k)
    return [datos[j] for j in torch.argsort(torch.cat(ix)).tolist()]


def prueba_ganchos(nombre, raiz, dev):
    """Comprobacion sin datos: pasa un lote aleatorio y devuelve las formas de cada capa."""
    red = Red(nombre, raiz, dev)
    x = torch.randn(2, 3, 224, 224).to(dev)
    with torch.no_grad():
        red.forward(x)
        out = []
        for c in red.capas:
            p = red.pooled(c["id"])
            e = tuple(red.espacial(c["id"]).shape) if c["tipo"] == "espacial" else None
            out.append((c["id"], tuple(p.shape), e, p.shape[1] == c["u"]))
    return out
