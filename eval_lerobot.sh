DATASET="pusht_cchi_v1_lerobotv3"

lerobot-eval \
  --policy.path=outputs/train/${DATASET}/diffusion_pusht/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --env.type=pusht \
  --eval.n_episodes=100 \
  --eval.batch_size=10 \
  --rename_map='{"observation.image":"observation.images.img", "agent_pos":"observation.state"}' \
  --output_dir=outputs/eval/${DATASET}/diffusion_pusht/last
