cd "$(dirname $0)"
if [ ! -n "$1" ] ; then
    gpu="0"
else
    gpu="$1"
fi
echo "gpu: $gpu"
python -u ./main.py \
    --dataset mag \
    --num-runs 1 \
    --K 5 \
    --num-hidden 512 \
    --lr 0.001 \
    --dropout 0.5 \
    --batch-size 10000  \
    --gpu $gpu \
    --aggr-gpu -1 \
    --eval-every 10 \
    --root '/data4/zwt/' \
    --epoch-setting 1000\
    --attn-drop 0. \
    --input-drop 0.1 \
    --sample-size 8 \
    --threshold 0.4 \
    --model NARS_gmlp \
    --emb-path  '/data4/zwt/NARS-main/' \
    --mlp-layer 2 \
    --label-K 3 \
    --use-labels \
    --warm_start 0 \
    --fixed-subsets


