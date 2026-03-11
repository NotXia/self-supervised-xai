cd ..
python ./train.py \
    --model-config "../configs/$1/config.json" \
    --checkpoint "$3" \
    --phase "masker" \
    --gpus "$2," \
    --epochs 5 \
    --batch-size 32 \
    --data-dir ../_datasets \
    --seed 42