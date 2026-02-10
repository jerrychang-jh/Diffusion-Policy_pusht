DATE=$(date +"%Y.%m.%d")
TIME=$(date +"%H.%M.%S")
DATASET="pusht_cchi_v2"

python diffusion_policy/train.py \
  --config-dir=. \
  --config-name=stanford_custom_config.yaml \
  task.dataset.zarr_path="Dataset/zarr/${DATASET}.zarr" \
  dataloader.num_workers=0 \
  training.max_train_steps=200000 \
  training.device="cuda:0" \
  training.seed=42 \
  training.val_every=100 \
  val_dataloader.batch_size=32 \
  val_dataloader.num_workers=0 \
  logging.name="${DATE}-${TIME}-${DATASET}_train_diffusion_unet_hybrid_pusht_image" \
  logging.project="diffusion_policy_debug" \
  hydra.run.dir="outputs/train/stanford/${DATE}/${TIME}_${DATASET}"