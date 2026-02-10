DATASET="pusht_cchi_v2"
DATE=2026.02.09
TIME=13.53.18

python diffusion_policy/eval.py \
    --checkpoint outputs/train/stanford/${DATE}/${TIME}_${DATASET}/checkpoints/latest.ckpt \
    --output_dir outputs/eval/stanford/${DATE}/${TIME}_${DATASET} \
    --device cuda:0