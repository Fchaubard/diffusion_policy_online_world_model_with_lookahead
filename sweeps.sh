#!/bin/bash
echo "[INFO] Killing existing screen sessions named '*'..."
screen -ls | grep '\.' | awk '{print $1}' | xargs -I{} screen -S {} -X quit 2>/dev/null


# Sweep values
ITERS_LIST=(1 10 100) # for diffusion time steps
LAYERS_LIST=(2 4 8) # for depth of the U-Net world model

# Base script to run (edit this!)
SCRIPT="train_diffusion_world_model_pusht.py"

# Counter for GPU ID / screen index
gpu=0

for iters in "${ITERS_LIST[@]}"; do
  for layers in "${LAYERS_LIST[@]}"; do
    if [ $gpu -ge 10 ]; then
      echo "❌ Only 10 GPUs available — skipping extra configs"
      break
    fi

    screen_name="sweep_gpu${gpu}_i${iters}_l${layers}"

    echo "🟢 Launching $screen_name on cuda:$gpu..."

    screen -dmS "$screen_name" bash -c "
      source ~/miniconda3/etc/profile.d/conda.sh && \
      conda init && \
      conda activate robodiff && \
      python $SCRIPT \
        --num_diffusion_iters_worldmodel $iters \
        --unet_num_layers $layers \
        --device cuda:$gpu
     echo '[INFO] Finished run $screen_name';
     exec bash
    "

    ((gpu+=1))
  done
done
