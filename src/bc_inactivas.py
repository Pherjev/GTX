#!/usr/bin/env python3
"""Estado de las neuronas segun su activacion sobre ImageNet-val (50 000 imagenes, con el agrupado de cada capa): VIVA, INHIBITORIA o MUERTA.

El criterio de Linear Explanations (Oikarinen y Weng, ICML 2024, arXiv 2405.06855, apendice C.2; leido 2026-10-16) llama MUERTA a la neurona cuya activacion MAXIMA sobre los datos de sondeo es < 0.01. Nuestro atlas lo reproduce exactamente (ViT-B/16 bloque11_mlp: 162 de 3072 = 5.27 %, la cifra del articulo).
PERO en capas con GELU (ViT, CLIP) ese criterio mezcla dos cosas distintas (hallazgo de Fernando, 2026-10-16, con el histograma de una neurona de rango -0.17 a -0.01): de las 16 078 neuronas con maximo < 0.01, solo 1 781 (11 %) son CONSTANTES (desviacion < 0.001); las otras 14 297 (89 %) VARIAN en la zona negativa de la GELU
(10 060 tienen minimo < -0.1) y llevan informacion (la tecnica GTX de Fernando explica con ejemplos negativos). Por eso hay tres estados:
  viva           : maximo >= umbral_max (0.01)
  inhibitoria  : maximo <  umbral_max y desviacion >= umbral_std (1e-3): varia, pero solo por debajo de cero (se analiza; NO se descarta)
  muerta         : maximo <  umbral_max y desviacion <  umbral_std: activacion practicamente constante (la unica realmente muerta)
Sin torch ni GPU. Uso:  python3 src/bc_inactivas.py --atlas-dir atlas_datos/resultados [--exporta neuronas_estado.json] [--csv neuronas_estado.csv]
"""
import argparse
import glob
import json
import os

RAIZ = os.path.expanduser(os.environ.get("GTX_RAIZ", "~/clusters/data/beyondcolor"))
ATLAS_DIR = os.path.join(RAIZ, "atlas", "resultados")
UMBRAL_LE = 0.01  # maximo < 0.01: criterio de Linear Explanations (apendice C.2)
UMBRAL_STD = 1e-3  # desviacion < 1e-3: activacion constante (la misma que 'casi constantes' del sitio)
ESTADOS = ("todas", "vivas", "muertas", "inhibitorias")
NOMBRE_ESTADO = {"vivas": "viva", "muertas": "muerta", "inhibitorias": "inhibitoria"}


def ruta_topk(red, capa, atlas_dir=ATLAS_DIR):
    return os.path.join(atlas_dir, f"{red}__{capa}__topk__k16.jsonl")


def clasifica(maximo, std, umbral_max=UMBRAL_LE, umbral_std=UMBRAL_STD):
    """'viva' | 'inhibitoria' | 'muerta' para una neurona con activacion maxima `maximo` y desviacion `std` (None = sin dato: se trata como variable)."""
    if maximo >= umbral_max:
        return "viva"
    return "muerta" if (std is not None and std < umbral_std) else "inhibitoria"


def carga_estadisticas(red, capa, atlas_dir=ATLAS_DIR):
    """{neurona: (maximo, desviacion)} de los resultados topk del atlas."""
    ruta = ruta_topk(red, capa, atlas_dir)
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"no hay {ruta}: el estado de las neuronas necesita los resultados topk del atlas")
    out = {}
    for l in open(ruta, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            e = r["extra"]
            out[r["neurona"]] = (float(e["max"]), (float(e["std"]) if e.get("std") is not None else None))
    return out


def carga_max(red, capa, atlas_dir=ATLAS_DIR):
    """{neurona: activacion maxima sobre ImageNet-val}."""
    return {u: m for u, (m, _) in carga_estadisticas(red, capa, atlas_dir).items()}


def carga_estados(red, capa, umbral=UMBRAL_LE, atlas_dir=ATLAS_DIR, umbral_std=UMBRAL_STD, est=None):
    """{neurona: 'viva' | 'inhibitoria' | 'muerta'}."""
    est = est if est is not None else carga_estadisticas(red, capa, atlas_dir)
    return {u: clasifica(m, s, umbral, umbral_std) for u, (m, s) in est.items()}


def inactivas(red, capa, umbral=UMBRAL_LE, atlas_dir=ATLAS_DIR, umbral_std=UMBRAL_STD):
    """Las MUERTAS (constantes): maximo < umbral Y desviacion < umbral_std. Las inhibitorias NO estan aqui: varian y se analizan."""
    return {u for u, e in carga_estados(red, capa, umbral, atlas_dir, umbral_std).items() if e == "muerta"}


def filtra(sel, red, capa, estado="todas", umbral=UMBRAL_LE, atlas_dir=ATLAS_DIR, mx=None, umbral_std=UMBRAL_STD):
    """Subconjunto de `sel` segun el estado: 'todas', 'vivas', 'muertas' (constantes) o 'inhibitorias' (inhibitorias). Conserva el orden de `sel`.
    `mx` (opcional) puede ser {neurona: maximo} o {neurona: (maximo, desviacion)}. Una neurona sin dato en el atlas cuenta como viva (no se descarta nada por falta de informacion)."""
    if estado not in ESTADOS:
        raise ValueError(f"estado {estado!r}: debe ser uno de {ESTADOS}")
    if estado == "todas":
        return list(sel)
    est = mx if mx is not None else carga_estadisticas(red, capa, atlas_dir)

    def estado_de(u):
        v = est.get(u)
        if v is None:
            return "viva"
        m, s = (v if isinstance(v, tuple) else (v, None))
        return clasifica(m, s, umbral, umbral_std)

    return [u for u in sel if estado_de(u) == NOMBRE_ESTADO[estado]]


def tabla(atlas_dir=ATLAS_DIR, umbral=UMBRAL_LE, umbral_std=UMBRAL_STD):
    """Lista de dict(red, capa, n, bajo_max, muertas, inhibitorias, fraccion_muertas, fraccion_inhibitorias) para todas las capas con resultados topk. bajo_max = maximo < umbral (criterio de LE)."""
    filas = []
    for f in sorted(glob.glob(os.path.join(atlas_dir, "*__topk__k16.jsonl"))):
        red, capa = os.path.basename(f).split("__")[:2]
        est = carga_estados(red, capa, umbral, atlas_dir, umbral_std)
        n = len(est)
        mu = sum(e == "muerta" for e in est.values())
        ne = sum(e == "inhibitoria" for e in est.values())
        filas.append(dict(red=red, capa=capa, n=n, bajo_max=mu + ne, muertas=mu, inhibitorias=ne, fraccion_muertas=mu / max(n, 1), fraccion_inhibitorias=ne / max(n, 1)))
    return filas


def exporta(atlas_dir=ATLAS_DIR, umbral=UMBRAL_LE, umbral_std=UMBRAL_STD):
    """{'umbral_max', 'umbral_std', 'criterio', 'capas': {'red/capa': {'n', 'n_muertas', 'muertas', 'n_inhibitorias', 'inhibitorias', 'estadisticas_no_vivas': {id: [max, std]}}}}."""
    capas = {}
    for f in tabla(atlas_dir, umbral, umbral_std):
        est = carga_estadisticas(f["red"], f["capa"], atlas_dir)
        estados = {u: clasifica(m, s, umbral, umbral_std) for u, (m, s) in est.items()}
        mu = sorted(u for u, e in estados.items() if e == "muerta")
        ne = sorted(u for u, e in estados.items() if e == "inhibitoria")
        capas[f"{f['red']}/{f['capa']}"] = dict(n=f["n"], n_muertas=len(mu), muertas=mu, n_inhibitorias=len(ne), inhibitorias=ne,
                                                estadisticas_no_vivas={str(u): [est[u][0], est[u][1]] for u in mu + ne})
    return dict(umbral_max=umbral, umbral_std=umbral_std,
                criterio=("muerta = maximo sobre ImageNet-val < umbral_max Y desviacion < umbral_std (activacion constante); inhibitoria = maximo < umbral_max pero desviacion >= umbral_std (varia por debajo de cero, GELU); "
                          "el criterio de Linear Explanations (apendice C.2) llama muerta a muertas + inhibitorias"),
                capas=capas)


def mal_inactivas(ruta, inactivas_ok):
    """Neuronas con una fila 'inactiva' en `ruta` que el criterio actual NO considera inactivas (marcadas con un criterio peor): hay que reintentarlas."""
    if not os.path.exists(ruta):
        return set()
    malas = set()
    for l in open(ruta, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r.get("extra", {}).get("inactiva") and r["neurona"] not in inactivas_ok:
                malas.add(r["neurona"])
    return malas


def purga(ruta, neuronas):
    """Reescribe `ruta` sin las filas de `neuronas` (escritura atomica). Devuelve cuantas filas quito."""
    if not neuronas or not os.path.exists(ruta):
        return 0
    quitadas, buenas = 0, []
    for l in open(ruta, encoding="utf-8"):
        if l.strip():
            if json.loads(l)["neurona"] in neuronas:
                quitadas += 1
            else:
                buenas.append(l if l.endswith("\n") else l + "\n")
    with open(ruta + ".tmp", "w", encoding="utf-8") as f:
        f.writelines(buenas)
    os.replace(ruta + ".tmp", ruta)
    return quitadas


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--atlas-dir", default=ATLAS_DIR)
    ap.add_argument("--umbral", type=float, default=UMBRAL_LE, help="umbral del maximo (criterio de LE: 0.01)")
    ap.add_argument("--umbral-std", type=float, default=UMBRAL_STD, dest="umbral_std")
    ap.add_argument("--exporta", default="", help="escribe el JSON con los IDs de las neuronas muertas y inhibitorias de todas las capas (ver exporta())")
    ap.add_argument("--csv", default="", help="escribe red,capa,neurona,estado,max,std de las neuronas no vivas")
    a = ap.parse_args()
    filas = tabla(a.atlas_dir, a.umbral, a.umbral_std)
    print(f"viva: maximo >= {a.umbral} | muerta: maximo < {a.umbral} y desviacion < {a.umbral_std} (constante) | inhibitoria: maximo < {a.umbral} y desviacion >= {a.umbral_std}")
    print(f"{'red':<11} {'capa':<22} {'n':>5} {'max<umbral':>10} {'muertas':>8} {'%':>6} {'inhib.':>9} {'%':>6}")
    for f in filas:
        print(f"{f['red']:<11} {f['capa']:<22} {f['n']:>5} {f['bajo_max']:>10} {f['muertas']:>8} {100 * f['fraccion_muertas']:>5.1f}% {f['inhibitorias']:>9} {100 * f['fraccion_inhibitorias']:>5.1f}%")
    print(f"TOTAL: {sum(f['n'] for f in filas)} neuronas | max<umbral {sum(f['bajo_max'] for f in filas)} | muertas {sum(f['muertas'] for f in filas)} | inhibitorias {sum(f['inhibitorias'] for f in filas)}")
    if a.exporta or a.csv:
        d = exporta(a.atlas_dir, a.umbral, a.umbral_std)
        if a.exporta:
            json.dump(d, open(a.exporta, "w", encoding="utf-8"))
            print("escrito", a.exporta)
        if a.csv:
            with open(a.csv, "w", encoding="utf-8") as f:
                f.write("red,capa,neurona,estado,max,std\n")
                for k, v in d["capas"].items():
                    red, capa = k.split("/")
                    for estado, ids in (("muerta", v["muertas"]), ("inhibitoria", v["inhibitorias"])):
                        for u in ids:
                            m, s = v["estadisticas_no_vivas"][str(u)]
                            f.write(f"{red},{capa},{u},{estado},{m:.6g},{'' if s is None else format(s, '.6g')}\n")
            print("escrito", a.csv)


if __name__ == "__main__":
    main()
