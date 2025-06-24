#!/usr/bin/env bash
# Sweep launcher for 10 × A40 node
# ────────────────────────────────────────────────────────────────
# 1. kill previous screens so the slate is clean
screen -ls | awk '/\t/{print $1}' | xargs -r -I{} screen -S {} -X quit

SCRIPT="train_video_world_model.py"
ROOT_CKP="./all_runs"          # parent folder for every run
mkdir -p "$ROOT_CKP"

# ── hyper-parameter grids ───────────────────────────────────────
BASE_LIST=(8 64)              # UNet base channels
T_LIST=(200 1000)               # diffusion steps for the world model
LR_LIST=(2e-4 2e-3)            # learning rates 

# (random-policy pct left constant – edit if desired)
RAND_PCT=0.1

gpu=0
for base in "${BASE_LIST[@]}"; do
  for T in "${T_LIST[@]}";     do
    for lr in "${LR_LIST[@]}"; do

      if [ $gpu -ge 10 ]; then
        echo "❌ 10 GPUs max — remaining configs skipped."
        break 3
      fi

      run_id="b${base}_T${T}_lr${lr}"
      screen_name="GPU${gpu}_${run_id}"
      ckpt_dir="${ROOT_CKP}/${run_id}"

      echo "🟢 Launching $screen_name → cuda:$gpu"

      # one detached screen per GPU / config
      screen -dmS "$screen_name" bash -c "
        # --- set visible GPU & load conda env ---------------------
        export CUDA_VISIBLE_DEVICES=$gpu
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate robodiff

        # --- launch training -------------------------------------
        python $SCRIPT \
          --base $base \
          --layers 4 \
          --T $T \
          --lr $lr \
          --microbatch 16 \
          --macrobatch 4 \
          --random_policy_pct $RAND_PCT \
          --device cuda:0 \
          --dataset_device cuda:0 \
          --ckpt_dir $ckpt_dir \
          --val_every 10 \
          --episodes 100000000 \
          --num_diffusion_iters_action_policy 100 \
          --action_horizon 8 \
          --context_frames 3 \
          --pred_horizon 16 \
          --max_steps_env 200 \
          --heads 4 \
          --beta0 1e-4 \
          --betaT 2e-2 \
          --img_hw 96 96 
        echo '[INFO] run $run_id finished';
        exec bash
      "

      ((gpu+=1))
    done
  done
done

echo "All $gpu screen sessions started ✓"
