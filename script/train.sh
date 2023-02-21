python3 ./train/train.py --arch Uformer_E --batch_size 16 --gpu 4,5,6,7 --env predict\
    --qk_mode dist-2\
    --warmup  --use_LENmu --time_encode --train_mask  --w1 1.0 --w2 0.0\
    --resume  --pretrain_weights /wight_path
