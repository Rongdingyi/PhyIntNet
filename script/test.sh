# python3 ./test/test_eye.py  --arch Uformer_E --gpu 3 --batch_size 1\
#         --qk_mode dist-2\
#         --warmup   --train_mask  --use_LENmu  --time_encode
        # 
# python3 ./test/test_eye.py --arch Uformer_E  --batch_size 1 --gpu 3 --train_mask --w1 1.0 --w2 1.0
# python3 ./test/test_eye.py --arch Full-Convolution  --batch_size 1 --gpu 3 --train_mask --w1 1.0 --w2 1.0
# python3 ./test/test_eye.py --arch Stacked-hourglass  --batch_size 1 --gpu 3 --train_mask --w1 1.0 --w2 1.0

python3 ./test/test_eye.py --arch DehazeFormer --batch_size 1 --gpu 1\
        --warmup --use_LENmu --time_encode  --train_mask  \
        --resume  --pretrain_weights /data1/rong/code/Eyes/exps_new/DehazeFormer/2023-06-06T00:34:39.678885/models/model_best.pth