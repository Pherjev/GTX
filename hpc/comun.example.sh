#!/bin/bash
# Definiciones compartidas por los lanzadores (bc16_gtx.sh). EJEMPLO: ajusta el contenedor, la particion y los limites a tu cluster. No se ejecuta solo.
# Tal como se uso en el paper: un cluster Slurm con una GPU AMD MI210 (ROCm), un contenedor con PyTorch, 30 min de pared por trabajo
# (los trabajos largos se parten en 'eslabones' de 25 min encadenados con --dependency=afterany y son reanudables).
cd ~/clusters || exit 1
mkdir -p logs
SIF="/opt/containers/pytorch_rocm.sif"            # contenedor con torch, torchvision, pyarrow, open_clip y transformers
COMUN="--partition=gpu --qos=beginner --gres=gpu:1 --time=00:28:00 --mem=16G --output=logs/%x_%j.out"
# En ROCm, Slurm define a la vez ROCR_VISIBLE_DEVICES y CUDA_VISIBLE_DEVICES y HIP aplica los dos filtros: se quita el de CUDA/HIP.
PY="env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES apptainer exec --rocm $SIF python3"
# En una GPU NVIDIA con torch instalado, bastaria:  PY="python3"

# Comprobacion previa: si el modulo no importa, no se encola nada.
importa () {
  $PY -c "import sys; sys.path.insert(0,'src'); import $1" \
    || { echo "ABORTADO: $1 no importa"; exit 1; }
}
lanza () { sbatch --job-name="$1" $COMUN --wrap="cd ~/clusters && $2" > /dev/null; }
