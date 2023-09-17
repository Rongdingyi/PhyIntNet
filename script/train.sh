python3 ./train/train.py --arch Uformer_E --batch_size 8 --gpu 0,1 --env look\
    --qk_mode dist-2\
    --warmup  --use_LENmu --time_encode  --train_mask  --w1 1.0 --w2 0.0