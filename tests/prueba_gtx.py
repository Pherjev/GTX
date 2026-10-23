"""Prueba de bc_gtx.py con datos FALSOS (necesita torch, CPU basta; no necesita cluster, parquet ni modelos de similitud).
Comprueba: la voraz en lote == la voraz secuencial escrita a mano (mismas selecciones y rho), u_pesada contra calculo directo, k-means/representantes,
selectores independientes de la neurona, y una corrida completa de una celda con particion/activaciones/caracteristicas falsas (formato de salida,
reanudacion tras cortar el eslabon, GTX > aleatorio en una neurona plantada, baselines presentes).
NO comprueba: la extraccion de caracteristicas con ResNet-101/ViT-B/16, la lectura del parquet ni los numeros del paper (eso lo hace el humo en el cluster)."""
import itertools
import json
import os
import sys
import tempfile
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import bc_gtx as G

torch.manual_seed(0)


# ---------------------------------------------------------------- 1. voraz en lote == voraz secuencial
def voraz_ref(S, a, u, C):
    """Algoritmo 1 del paper, ingenuo: para cada candidato calcula u_pesada completa y su Pearson (NaN -> 0)."""
    k, M = S.shape
    D, hist = [], []
    for _ in range(C):
        mejor, bi = -9.0, None
        for j in range(k):
            if j in D:
                continue
            T = D + [j]
            sims = S[T]  # [|T|, M]
            m, w = sims.max(0)
            pred = m * a[torch.tensor(T)][w]
            r = torch.corrcoef(torch.stack([pred, u]))[0, 1]
            r = 0.0 if torch.isnan(r) else float(r)
            if r > mejor:
                mejor, bi = r, j
        D.append(bi)
        hist.append(mejor)
    return D, hist


k, M, B, C = 40, 120, 5, 4
F = torch.nn.functional.normalize(torch.randn(k + M, 16), dim=1)
S = F[:k] @ F[k:].T
a = torch.randn(B, k)
u = torch.randn(B, M) + 0.5 * (a @ S) / 3
sel, rho = G.voraz_lote(S, a, u, C, trozo=7)  # trozo no divisor de k: prueba el recorte del ultimo trozo
for b in range(B):
    D, h = voraz_ref(S, a[b], u[b], C)
    assert sel[b].tolist() == D, (b, sel[b].tolist(), D)
    assert torch.allclose(rho[b], torch.tensor(h), atol=1e-5), (rho[b], h)
print("voraz en lote == voraz secuencial (5 neuronas, 4 pasos, trozo no divisor de k): OK")

# ---------------------------------------------------------------- 2. u_pesada
Fn = G.normaliza(torch.randn(50, 8))
idx = torch.tensor([[1, 7, 9], [3, 4, 5]])
acts = torch.randn(2, 3)
Fe = G.normaliza(torch.randn(11, 8))
p = G.pred_pesada(Fn, idx, acts, Fe)
for b in range(2):
    for m in range(11):
        s = Fn[idx[b]] @ Fe[m]
        j = int(s.argmax())
        assert abs(float(p[b, m]) - float(s[j] * acts[b, j])) < 1e-6
c = G.corr_col(torch.ones(2, 6), torch.randn(2, 6))
assert torch.all(c == 0), "prediccion constante -> NaN -> 0"
print("u_pesada = sim(x, x*) * u(x*) y NaN -> 0: OK")

# ---------------------------------------------------------------- 3. k-means y representantes con grupos plantados
centros = G.normaliza(torch.randn(6, 32) * 3)
X = G.normaliza(centros.repeat_interleave(100, 0) + 0.05 * torch.randn(600, 32))
Cc, lab = G.kmeans(X, 6, iters=30, seed=1)
verd = torch.arange(6).repeat_interleave(100)
puros = sum(len(set(verd[lab == c].tolist())) == 1 for c in range(6))
assert puros == 6, f"clusters puros: {puros}/6"
R = G.representantes(X, Cc, lab)
assert len(R) == 6 and len(set(verd[R].tolist())) == 6
print("k-means recupera 6 grupos plantados y da un representante por grupo: OK")

# ---------------------------------------------------------------- 4. selectores independientes de la neurona
Xp = G.normaliza(torch.cat([centros[:4].repeat_interleave(50, 0) + 0.02 * torch.randn(200, 32)]))
vp = torch.arange(4).repeat_interleave(50)
for nombre, f in (("kcenter", lambda: G.sel_kcenter(Xp, 4, 0)), ("kmedoids", lambda: G.sel_kmedoids(Xp, 4, 0)), ("dpp", lambda: G.sel_dpp(Xp, 4))):
    s = f()
    assert len(set(s.tolist())) == 4, nombre
    assert len(set(vp[s].tolist())) == 4, f"{nombre} debe tomar uno de cada grupo plantado, tomo {vp[s].tolist()}"
print("kcenter / kmedoids / dpp toman un ejemplo de cada grupo plantado: OK")

# ---------------------------------------------------------------- 5. celda completa con archivos falsos
tmp = tempfile.mkdtemp()
G.RAIZ = tmp
G.NSHARD = 2
NS = [300, 260]  # imagenes por shard
N = sum(NS)
D = 16
grupo = torch.randint(0, 8, (N,))
cent = G.normaliza(torch.randn(8, D) * 3)
feat = G.normaliza(cent[grupo] + 0.1 * torch.randn(N, D))
# neurona 0: activa solo en el grupo 3; neurona 1: activa en los grupos 0 y 5, negativa en el 7; neurona 2: ruido; neurona 3: constante (muerta)
A = torch.zeros(N, 4)
A[:, 0] = (grupo == 3).float() * 2 + 0.05 * torch.randn(N)
A[:, 1] = (grupo == 0).float() * 1.5 + (grupo == 5).float() * 1.0 - (grupo == 7).float() * 1.0 + 0.05 * torch.randn(N)
A[:, 2] = torch.randn(N)
A[:, 3] = 0.0
os.makedirs(os.path.join(tmp, "atlas", "cache"))
os.makedirs(os.path.join(tmp, "le", "cache"))
os.makedirs(os.path.join(tmp, "gtx", "cache"))
o = 0
for i, n in enumerate(NS):
    torch.save({"capaX": A[o:o + n].half()}, os.path.join(tmp, "atlas", "cache", f"act_redfalsa_{i:02d}.pt"))
    torch.save({"label": torch.zeros(n).long(), "path": [f"val_{o + j}.JPEG" for j in range(n)]}, os.path.join(tmp, "le", "cache", f"le_feats_{i:02d}.pt"))
    torch.save(feat[o:o + n].half(), os.path.join(tmp, "gtx", "cache", f"sim_resnet101_{i:02d}.pt"))
    o += n

# modulos del proyecto que no se pueden importar aqui (parquet, torchvision): sustitutos minimos
fake_le = types.ModuleType("splits_le")
def particion(datos, cache, plan):
    p = torch.randperm(N, generator=torch.Generator().manual_seed(3))
    return {"train": p[:int(.7 * N)].sort().values, "val": p[int(.7 * N):int(.8 * N)].sort().values, "test": p[int(.8 * N):].sort().values}, "falsa"
fake_le.particion = particion
fake_red = types.ModuleType("bc_redes")
fake_red.capas = lambda red: [dict(id="capaX", tipo="espacial", u=4)]
sys.modules["splits_le"], sys.modules["bc_redes"] = fake_le, fake_red
G.SIM_DE["redfalsa"] = "resnet101"

import argparse
def args(**kw):
    base = dict(plan="evalua", red="redfalsa", capas="capaX", neuronas="", muestra=0, estado="todas", metodos="gtx,aleatorio,topk,extremos,kmedoids,kcenter,dpp,gtxpropio",
                C="3,5", k=60, kmeans_iters=15, semilla=0, semillas=3, sub=200, lote=2, reserva=0, minutos=60, eta_redes="redfalsa", muestra_eta=0, salida=None, exige_gpu=False, existe=False, resumen=False)
    base.update(kw)
    return argparse.Namespace(**base)

plan = G.Plan(args())
_U0 = G.U_de
G.U_de = lambda red, capa: 4 if red == "redfalsa" else _U0(red, capa)
cel = plan.celdas()
assert [c[2] for c in cel] == plan.metodos and all(c[0] == "redfalsa" for c in cel), cel
ctx = G.Contexto(plan)
# primer eslabon: se corta a mitad (quedan() se agota tras unos lotes) y se reanuda
cuenta = itertools.count()
cortado = G.celda_metodo(ctx, cel[0], lambda: 100.0 if next(cuenta) < 1 else -1.0)
assert cortado is False, "debia cortarse por tiempo"
rt = G.ruta_salida(plan.H, "redfalsa", "capaX", "gtx", 3, plan.k_de("redfalsa"))
hechas1 = G.lee_hechas(rt)
assert 0 < len(hechas1) < 4, hechas1
assert G.celda_metodo(ctx, cel[0], lambda: 100.0) is True
assert G.lee_hechas(rt) == {0, 1, 2, 3}
assert open(rt).read().count("\n") == 4, "sin duplicados tras reanudar"
for c in cel[1:]:
    assert G.celda_metodo(ctx, c, lambda: 100.0) is True
assert not any(plan.pendiente(c, G.U_de) for c in cel), "todo hecho"
print("celda completa con corte y reanudacion, sin filas duplicadas: OK")

def lee(m, C):
    return {r["neurona"]: r for r in map(json.loads, open(G.ruta_salida(plan.H, "redfalsa", "capaX", m, C, plan.k_de("redfalsa")), encoding="utf-8"))}

g3, g5, al, tk = lee("gtx", 3), lee("gtx", 5), lee("aleatorio", 3), lee("topk", 3)
r = g3[0]
assert set(r) >= {"neurona", "hipotesis", "puntaje", "longitud", "uid", "extra", "red", "capa", "explicador", "config"}
assert r["uid"] == "redfalsa/capaX/0/gtx/C3-k60" and r["config"] == "C3-k60" and r["longitud"] == 3 and len(r["extra"]["indices"]) == 3
assert r["extra"]["particion"] == "falsa" and r["extra"]["fuente"] == "imagenet" and r["puntaje"] == r["extra"]["corr_test"]
assert g3[0]["extra"]["indices"] == g5[0]["extra"]["indices"][:3], "la voraz es anidada: C=3 es prefijo de C=5"
idx = r["extra"]["indices"]
assert all(abs(A[i, 0].item() - x) < 5e-3 for i, x in zip(idx, r["extra"]["activaciones"])), "las activaciones guardadas son las del atlas en esos indices"
assert os.path.basename(G.ruta_salida(plan.H, "redfalsa", "capaX", "gtx", 5, 2000)) == "redfalsa__capaX__gtx__C5-k2000.jsonl"
assert G.dir_de(plan.H, "gtx") != G.dir_de(plan.H, "topk")
print(f"formato de fila, anidamiento C=3 prefijo de C=5, activaciones del atlas, nombre de archivo del sitio: OK")
print(f"neurona 0 (grupo 3): gtx test={g3[0]['puntaje']:.3f}  aleatorio={al[0]['puntaje']:.3f}  topk={tk[0]['puntaje']:.3f}")
print(f"neurona 1 (grupos 0,5,-7): gtx test={g3[1]['puntaje']:.3f}  aleatorio={al[1]['puntaje']:.3f}  topk={tk[1]['puntaje']:.3f}")
assert g3[0]["puntaje"] > 0.8 and g3[1]["puntaje"] > 0.7, "GTX debe capturar neuronas con estructura plantada"
assert g3[0]["puntaje"] > al[0]["puntaje"] and g3[1]["puntaje"] > tk[1]["puntaje"], "GTX debe superar al azar y a top-k en neuronas con estructura"
assert abs(g3[3]["puntaje"]) < 1e-9, "neurona constante: correlacion NaN -> 0"
assert abs(g3[2]["puntaje"]) < 0.3, "ruido puro: sin correlacion esperable"
for m in ("kmedoids", "kcenter", "dpp", "extremos", "gtxpropio"):
    assert len(lee(m, 3)) == 4
print("todos los metodos escriben las 4 neuronas; neurona constante -> 0; ruido -> ~0: OK")


# ---------------------------------------------------------------- 6. red externa (EXT) con conjunto falso: celdas sim -> act -> metodos, orden y particion de respaldo
import numpy as np
G.WORKERS = 0
G.REPO_LE = os.path.join(tmp, "sin_le")  # no hay particiones de LE -> semilla fija 70/10/20
NE = 120
gE = torch.randint(0, 8, (NE,), generator=torch.Generator().manual_seed(5))
cE = G.normaliza(torch.randn(8, 16, generator=torch.Generator().manual_seed(6)) * 3)
featE = G.normaliza(cE[gE] + 0.1 * torch.randn(NE, 16, generator=torch.Generator().manual_seed(7)))
actE = torch.stack([(gE == 3).float() * 2, (gE == 0).float() - (gE == 7).float(), torch.randn(NE, generator=torch.Generator().manual_seed(8))], 1)


class ImFalsa:
    def __init__(self, t):
        self.t = t

    def convert(self, modo):
        return self


class DsFalso:
    def __len__(self):
        return NE

    def __getitem__(self, i):
        return ImFalsa(torch.cat([featE[i], actE[i]])), int(gE[i])


G.abre_ds = lambda H, ds, descarga=False: DsFalso()
G.carga_red_ext = lambda red, H, dev: ((lambda im: im.t), (lambda x: {"c1": x[:, 16:19].half()}))
G.carga_modelo_sim = lambda nombre, H, dev: (None, (lambda im: im.t), (lambda m, x: x[:, :16]))
G.EXT["extfalsa"] = dict(ds="cifar100", sim="vitb16hf", k=15, n=NE, capas=[("c1", 3)])
G.SIM_DE["extfalsa"] = "vitb16hf"
planE = G.Plan(args(red="extfalsa", capas="c1", C="3", k=0, metodos="gtx,topk,kmedoids", sub=60, lote=3))
celE = planE.celdas()
assert [c[0] for c in celE] == ["sim", "act", "ext", "ext", "ext"][:0] or [c[0] for c in celE][:2] == ["sim", "act"], celE
assert celE[0] == ("sim", "vitb16hf", 0, "cifar100") and celE[1] == ("act", "extfalsa", 0, "cifar100") and [c[2] for c in celE[2:]] == ["gtx", "topk", "kmedoids"], celE
assert planE.k_de("extfalsa") == 15 and G.U_de("extfalsa", "c1") == 3
ctxE = G.Contexto(planE)
for c in celE:
    ok = G.celda_sim(ctxE, c, lambda: 99.0) if c[0] == "sim" else G.celda_act(ctxE, c, lambda: 99.0) if c[0] == "act" else G.celda_metodo(ctxE, c, lambda: 99.0)
    assert ok is True
assert not any(planE.pendiente(c, G.U_de) for c in celE)
aE = torch.load(G.ruta_act(planE.H, "extfalsa", 0))
assert aE["c1"].shape == (NE, 3) and torch.allclose(aE["c1"].float(), actE.half().float()), "las activaciones guardadas deben estar en el orden del conjunto"
sE = torch.load(G.ruta_sim(planE.H, "vitb16hf", 0, "cifar100"))
assert sE.shape == (NE, 16)
rE = [json.loads(l) for l in open(G.ruta_salida(planE.H, "extfalsa", "c1", "gtx", 3, 15), encoding="utf-8")]
assert len(rE) == 3 and rE[0]["extra"]["particion"] == "semilla0" and rE[0]["config"] == "C3-k15" and rE[0]["extra"]["fuente"] == "cifar100"
assert rE[0]["puntaje"] > 0.5, rE[0]["puntaje"]
print(f"red externa (conjunto falso): sim -> act -> gtx/topk/kmedoids, activaciones en orden del conjunto, particion de respaldo, k propio de la red: OK (neurona 0 gtx test={rE[0]['puntaje']:.2f})")


# ---------------------------------------------------------------- 7. fuente POOL (sitio publico): trozos, activaciones por trozo, particion guardada, salida en gtx_pool
tmp_pool = os.path.join(tmp, "pool_abierto")
os.makedirs(tmp_pool)
NP = 90
gP = torch.randint(0, 8, (NP,), generator=torch.Generator().manual_seed(11))
featP = G.normaliza(cE[gP] + 0.1 * torch.randn(NP, 16, generator=torch.Generator().manual_seed(12)))
actP = torch.stack([(gP == 3).float() * 2 + 0.05 * torch.randn(NP, generator=torch.Generator().manual_seed(13)), (gP == 0).float() - (gP == 7).float()], 1)
with open(os.path.join(tmp_pool, "pool_listo.jsonl"), "w") as f:
    for p in range(NP):
        f.write(json.dumps({"pid": p, "ruta": f"imagenes/{p // 1000:03d}/{p:06d}.jpg", "pool": "oi", "autor": f"autor{p}", "licencia": "https://creativecommons.org/licenses/by/2.0/"}) + chr(10))
G.POOL_DIR = tmp_pool
G._POOL.clear()
G.abre_imagen = lambda ruta: ImFalsa(torch.cat([featP[int(os.path.basename(ruta)[:6])], actP[int(os.path.basename(ruta)[:6])]]))
G.FUENTE = "pool"
G.carga_red_atlas = lambda red, H, dev: ((lambda im: im.t), (lambda x: {"cP": x[:, 16:18].half()}))
G.SIM_DE["redpool"] = "resnet101"
import bc_redes as _br
_br.capas = lambda red: [dict(id="cP", tipo="espacial", u=2)]
G.carga_modelo_sim = lambda nombre, H, dev: (None, (lambda im: im.t), (lambda m, x: x[:, :16]))
assert G.limites_pool(0) == (0, 45) and G.limites_pool(1) == (45, 90), "NSHARD=2: dos trozos"
planP = G.Plan(args(red="redpool", capas="cP", C="3", k=20, metodos="gtx", lote=2))
assert planP.H["gtx_dir"].endswith(os.path.join("cosy", "gtx_pool")), planP.H["gtx_dir"]
celP = planP.celdas()
assert [c[0] for c in celP] == ["sim", "act", "sim", "act", "redpool"], celP
assert celP[0] == ("sim", "resnet101", 0, "pool") and celP[1] == ("act", "redpool", 0, "pool")
ctxP = G.Contexto(planP)
for c in celP:
    ok = G.celda_sim(ctxP, c, lambda: 99.0) if c[0] == "sim" else G.celda_act(ctxP, c, lambda: 99.0) if c[0] == "act" else G.celda_metodo(ctxP, c, lambda: 99.0)
    assert ok is True
assert os.path.exists(G.ruta_act(planP.H, "redpool", 1)) and "pool_01" in G.ruta_act(planP.H, "redpool", 1)
partP = json.load(open(os.path.join(tmp_pool, "gtx_particion.json")))
assert sorted(sum(partP.values(), [])) == list(range(NP)), "la particion cubre todo el pool una sola vez"
rP = [json.loads(l) for l in open(G.ruta_salida(planP.H, "redpool", "cP", "gtx", 3, 20), encoding="utf-8")]
assert len(rP) == 2 and rP[0]["extra"]["fuente"] == "pool" and rP[0]["extra"]["particion"] == "pool_semilla0"
assert all(0 <= i < NP for r in rP for i in r["extra"]["indices"]), "los indices son pid del pool"
mu = float(actP[torch.tensor(partP["train"]), 0].mean())
assert abs(rP[0]["extra"]["media_train"] - mu) < 5e-3, (rP[0]["extra"]["media_train"], mu)
assert rP[0]["puntaje"] > 0.5, rP[0]["puntaje"]
# miniaturas y creditos del pool (sin PIL real): solo se comprueba la recoleccion de pid y los campos de credito
usados = sorted({i for r in rP for i in r["extra"]["indices"]})
assert len(usados) >= 3
print(f"fuente pool: trozos, act por trozo, particion guardada una vez, salida en cosy/gtx_pool, media_train, indices = pid: OK (neurona 0 gtx test={rP[0]['puntaje']:.2f})")

# ---------------------------------------------------------------- 8. mapas de calor de las tablas GTX (fuente pool): alineacion (neurona, imagen) y normalizacion
G.K_MAPA = 3  # la tabla de este ejemplo es de C=3
class RedFalsa:
    """Mapas espaciales conocidos: el mapa de la neurona u en la imagen x es su activacion (columnas 16:18 de x) repartida en 4x4, con un 'punto caliente' en (0, 0)."""
    def __init__(self, nombre, raiz, dev):
        self.transform = lambda im: im.t
        self.x = None

    def forward(self, x):
        self.x = x

    def espacial(self, cid):
        a = self.x[:, 16:18]
        m = a[:, :, None, None] * torch.ones(1, 1, 4, 4)
        m[:, :, 0, 0] *= 2
        return m


_br.Red = RedFalsa
_br.cobertura_mapa = lambda red: 0.875
planM = G.Plan(args(red="redpool", capas="cP", C="3", k=20, metodos="gtx", lote=2))
assert G.pares_mapas(planM.H, "redpool", 20)["cP"]["idx"].shape == (2, 3)
G.plan_mapas(planM, lambda: 999.0)
assert all(os.path.exists(G.ruta_mapa_trozo(planM.H, "pool", "redpool", i)) for i in (0, 1))
z = np.load(G.ruta_mapa_final(planM.H, "pool", "redpool", "cP"))
assert z["m"].shape == (2, 3, 4, 4) and z["m"].dtype == np.uint8 and z["indices"].shape == (2, 3) and int(z["C"]) == 3 and abs(float(z["cobertura"]) - 0.875) < 1e-9
idxs = {json.loads(l)["neurona"]: json.loads(l)["extra"]["indices"] for l in open(G.ruta_salida(planM.H, "redpool", "cP", "gtx", 3, 20))}
for u in (0, 1):
    assert z["indices"][u].tolist() == idxs[u], "la tabla guardada en el npz es la del jsonl"
    act = torch.tensor([float(actP[i, u]) for i in idxs[u]]).clamp_min(0)
    esperado = act / act.max().clamp_min(1e-8)
    obtenido = torch.tensor([float(z["m"][u, j, 0, 0]) / 255.0 for j in range(3)])  # el punto caliente (0, 0) es el maximo de cada mapa
    assert torch.allclose(obtenido, esperado.float(), atol=0.02) or float(act.max()) == 0.0, (u, obtenido, esperado)
    assert z["m"][u].max() == (255 if float(act.max()) > 0 else 0)
planM.a.existe = True
try:
    G.plan_mapas(planM, lambda: 999.0)
    raise AssertionError("--existe debia salir")
except SystemExit as e:
    assert e.code == 0, "con todo hecho --existe sale 0"
print("mapas de calor GTX: alineacion (neurona, imagen), punto caliente, normalizacion por neurona y --existe: OK")
G.FUENTE = "imagenet"

# ETA con datos sinteticos (solo comprueba que corre e imprime; en CPU los tiempos no sirven)
G.KFULL = 60
G.TAMS_ETA["redfalsa"] = [4]
G.eta(plan, 50.0, 1.0)
print("TODO OK")
