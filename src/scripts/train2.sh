cd ..
python ./train.py \
    --model-config "../configs/$1/config.json" \
    --phase "finetune" \
    --gpus "$2," \
    --epochs 10 \
    --batch-size 32 \
    --mixed-precision \
    --data-dir ../_datasets \
    --seed 42