DATASET="pusht_cchi_v1_lerobotv3"

lerobot-train \
  --dataset.repo_id=local/${DATASET} \
  --dataset.root=/root/.cache/huggingface/lerobot/local/${DATASET} \
  --policy.type=diffusion \
  --policy.repo_id=local/diffusion-${DATASET} \
  --policy.push_to_hub=false \
  --policy.device="cuda" \
  --policy.config.robot_state_feature=null \
  --env.type=pusht \
  --output_dir=outputs/train/${DATASET}/diffusion_pusht \
  --steps=200000 \
  --batch_size=64 \
  --eval_freq=25000 \
  --wandb.enable=false
