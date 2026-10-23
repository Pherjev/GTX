#!/bin/bash
# GTX y baselines de seleccion de ejemplos (src/bc_gtx.py) sobre ImageNet-val, con las activaciones del atlas y las particiones de LE.
# PREPARACION (una vez, en el nodo de acceso, que es el unico con internet): bajar ResNet-101 (torchvision) y ViT-B/16 de Hugging Face (modelos de similitud):
#   cd ~/clusters && apptainer exec /opt/containers/pytorch_rocm.sif python3 src/bc_gtx.py --plan descarga
#   (si falta `transformers` en el contenedor: apptainer exec /opt/containers/pytorch_rocm.sif pip install --no-deps --target ~/clusters/pylibs transformers  y sus dependencias; avisar si falla)
# HUMO (shard 0, resnet50/layer1, 16 neuronas; imprime el ETA de la corrida completa y la correlacion frente a LE):
#   PLAN=humo bash hpc/noche/bc16_gtx.sh 1          # tras correrlo: grep -A40 "ETA" logs/b16_gtx_humo_e1_*.out
# EVALUACION (reanudable; repetir hasta "plan completo"):
#   PLAN=evalua ARGS="--red resnet50 --capas layer1,layer2,layer3,layer4" bash hpc/noche/bc16_gtx.sh 6
#   PLAN=evalua ARGS="--red vitb16 --capas bloque11_mlp --metodos gtx,aleatorio,topk,kmedoids,dpp" bash hpc/noche/bc16_gtx.sh 6
#   RESUMEN:  apptainer exec ... python3 src/bc_gtx.py --plan evalua --red resnet50 --capas layer1,layer4 --resumen
# Resultados: data/beyondcolor/cosy/gtx/ (gtx, lo lee el sitio) y data/beyondcolor/gtx/baselines/ (el resto). Humo: data/beyondcolor/gtx/humo/.
# comun.sh: copia hpc/comun.example.sh como hpc/comun.sh y ajustalo a tu cluster
source "$(dirname "$0")/comun.sh"
importa bc_gtx

PLAN="${PLAN:-evalua}"
NOMBRE="b16_gtx_${PLAN}${TAG:+_$TAG}"
R="$COMUN --cpus-per-task=${CPUS:-8}"
EXTRA="--exige-gpu $ARGS"

if squeue -u "$USER" -h -o "%j" | grep -q "^${NOMBRE}_"; then echo "  ya en cola"; exit 0; fi
if [ -z "$FORZAR" ] && [ "$PLAN" != "humo" ]; then
out=$($PY src/bc_gtx.py --plan "$PLAN" $ARGS --existe 2>&1); rc=$?
echo "$out" | grep -E "celdas:|Error|Traceback" | head -n 6
if [ $rc -eq 0 ]; then echo "  plan completo"; exit 0; fi
fi
ant=""
[ -n "$DESPUES" ] && ant=$(squeue -u "$USER" -h -o "%i %j" | grep -E " (${DESPUES})" | awk '{print $1}' | sort -n | tail -1)
[ -n "$DESPUES" ] && echo "  espera al trabajo ${ant:-ninguno}"
for e in $(seq 1 "${1:-2}"); do
  dep=""; [ -n "$ant" ] && dep="--dependency=afterany:$ant"
  ant=$(sbatch --parsable $dep --job-name="${NOMBRE}_e$e" $R --wrap="cd ~/clusters && $PY src/bc_gtx.py --plan $PLAN --minutos 25 $EXTRA")
done
echo "$NOMBRE: ${1:-2} eslabones encolados"
