"""Resume los jsonl de bc_gtx.py (bajados del cluster) como la Tabla 2 del paper: media +- error estandar de la correlacion por red/capa/metodo/C.
Solo biblioteca estandar. Uso (PowerShell, desde esta carpeta):
    python tabla_resultados.py resultados            # carpeta donde se descomprimio gtx_res.tgz
Escribe resultados\\tabla_resultados.csv y resultados\\tabla_resultados.md, e imprime las capas INCOMPLETAS (menos neuronas de las que tiene la capa).
Columnas: test = correlacion en test (lo que LE llama puntaje); val = en validacion (sobre la que GTX optimiza: optimista para gtx, honesta para los baselines)."""
import csv
import glob
import json
import math
import os
import sys

TAMANOS = {"resnet50": {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048, "fc": 1000}, "vitb16": {"bloque11_mlp": 3072},
           "vitl32": {"bloque23_mlp": 4096}, "resnet18_places": {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512},
           "vgg16_cifar": {"features_43": 512, "classifier_4": 512}}
ORDEN_M = ["gtx", "gtxpropio", "topk", "extremos", "aleatorio", "kmedoids", "kcenter", "dpp"]


def media_sem(x):
    n = len(x)
    m = sum(x) / n
    if n < 2:
        return m, float("nan")
    v = sum((a - m) ** 2 for a in x) / (n - 1)
    return m, math.sqrt(v / n)


def main(raiz):
    archivos = glob.glob(os.path.join(raiz, "**", "*.jsonl"), recursive=True)
    archivos = [f for f in archivos if "__" in os.path.basename(f) and "ejemplos" not in f]
    filas = []
    for f in archivos:
        red, capa, metodo, cfg = os.path.basename(f)[:-6].split("__")[:4]
        rs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        if not rs:
            continue
        C = rs[0]["longitud"]
        t, st = media_sem([r["extra"]["corr_test"] for r in rs])
        v, sv = media_sem([r["extra"]["corr_val"] for r in rs])
        U = TAMANOS.get(red, {}).get(capa)
        vivas = [r["extra"]["corr_test"] for r in rs if r["extra"].get("estado_atlas", "viva") != "muerta"]
        filas.append(dict(red=red, capa=capa, metodo=metodo, C=C, n=len(rs), U=U or "", completa="si" if U == len(rs) else ("NO" if U else "?"), test=t, sem_test=st,
                          val=v, sem_val=sv, test_sin_muertas=(sum(vivas) / len(vivas) if vivas else float("nan")), particion=rs[0]["extra"].get("particion", ""), k=rs[0]["extra"].get("k", ""), config=cfg))
    filas.sort(key=lambda r: (r["red"], r["capa"], r["C"], ORDEN_M.index(r["metodo"]) if r["metodo"] in ORDEN_M else 99))
    campos = ["red", "capa", "metodo", "C", "n", "U", "completa", "test", "sem_test", "val", "sem_val", "test_sin_muertas", "particion", "k", "config"]
    with open(os.path.join(raiz, "tabla_resultados.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, campos)
        w.writeheader()
        w.writerows(filas)
    with open(os.path.join(raiz, "tabla_resultados.md"), "w", encoding="utf-8") as f:
        f.write("| red | capa | C | metodo | n/U | test (media ± sem) | val (media ± sem) | particion |\n|---|---|--:|---|---|---|---|---|\n")
        for r in filas:
            f.write(f"| {r['red']} | {r['capa']} | {r['C']} | {r['metodo']} | {r['n']}/{r['U']} | {r['test']:.4f} ± {r['sem_test']:.4f} | {r['val']:.4f} ± {r['sem_val']:.4f} | {r['particion']} |\n")
    inc = sorted({(r["red"], r["capa"], r["metodo"], r["n"], r["U"]) for r in filas if r["completa"] == "NO"})
    print(f"{len(filas)} combinaciones leidas de {len(archivos)} archivos -> {raiz}\\tabla_resultados.csv y .md")
    print("INCOMPLETAS (n < neuronas de la capa):" if inc else "Todas las capas estan completas.")
    for x in inc:
        print("  ", x)
    parts = sorted({r["particion"] for r in filas})
    print("particiones usadas:", parts, "  (semilla0 = NO es la particion de LE)" if "semilla0" in parts else "")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "resultados")
