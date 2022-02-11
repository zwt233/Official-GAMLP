# GAMLP: Graph Attention Multi-Layer Perceptron

This repository is the official implementation of GAMLP.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

### PPI

```
cd ./ppi
python ppi.py --dataset ppi --method JK_GAMLP --dropout 0.1 --input-drop 0.1 --att-drop 0 --act sigmoid --label-num-hops 3 --num-hops 10 --epochs 800  --n-layers-1 4 --n-layers-2 6 --lr 0.001 --weight-decay 0 --hidden 2048 --seed 42 --label-num-hops 9 --use-label --patience 100
```

````
cd ./ppi
python ppi.py --dataset ppi --method R_GAMLP --dropout 0.1 --input-drop 0.1 --att-drop 0 --act sigmoid --label-num-hops 3 --num-hops 10 --epochs 600  --n-layers-1 4 --n-layers-2 6 --lr 0.001 --weight-decay 0 --hidden 2048 --seed 42 --label-num-hops 9 --use-label --patience 100
````

### Reddit

```
python main.py --dataset reddit --method JK_GAMLP --dropout 0.7 --input-drop 0 --att-drop 0 --act leaky_relu --batch 1000 --num-hops 10 --epochs 350 --n-layers-1 2 --n-layers-2 2 --n-layers-3 2 --n-layers-4 2 --lr 0.0001 --weight-decay 0 --hidden 512 --patience 100 --seed 42 --pre-process --use-label --label-drop 0.3 --label-num-hops 4 --weight 1e-7 --num-runs 1 
```

```
python main.py --dataset reddit --method R_GAMLP --dropout 0.7 --input-drop 0 --att-drop 0 --act leaky_relu --batch 1000 --num-hops 10 --epochs 350 --n-layers-1 2 --n-layers-2 2 --n-layers-3 2 --n-layers-4 2 --lr 0.0001 --weight-decay 0 --hidden 512 --patience 100 --seed 42 --pre-process --use-label --label-drop 0.5 --label-num-hops 4 --weight 1e-7  --num-runs 1 
```

### Flickr

```
python main.py --dataset flickr --method JK_GAMLP --dropout 0.7 --input-drop 0 --att-drop 0 --act leaky_relu --batch 250 --num-hops 10 --epochs 50 --n-layers-1 2 --n-layers-2 2 --n-layers-3 2 --n-layers-4 2 --lr 0.001 --weight-decay 0 --hidden 512 --patience 100 --seed 42 --pre-process --use-label --label-drop 0.5 --label-num-hops 5 --num-runs 1
```

```
python main.py --dataset flickr --method R_GAMLP --dropout 0.5 --input-drop 0 --att-drop 0 --act leaky_relu --batch 250 --num-hops 10 --epochs 50 --n-layers-1 2 --n-layers-2 2 --n-layers-3 2 --n-layers-4 2 --lr 0.0001 --weight-decay 0 --hidden 512 --patience 100 --seed 42 --pre-process --use-label --label-drop 0.5 --label-num-hops 5 --num-runs 1
```
### ogbn-mag

```
python main.py --method JK_GAMLP --stages 500 --train-num-epochs 0 --input-drop 0.3 --att-drop 0 --label-drop 0 --pre-process --dataset ogbn-mag --num-runs 10 --eval 10 --act leaky_relu --batch 10000 --patience 300 --n-layers-1 4 --n-layers-2 6 --label-num-hops 3 --bns --use-relation-subsets ./mag --emb_path ./ --root ./
```

### ogbn-products

```
python main.py --method R_GAMLP --epochs 1000 --input-drop 0.5 --att-drop 0.4 --dropout 0.2  --label-drop 0 --pre-process --dataset ogbn-products --num-runs 10 --gpu 0 --eval-every 1 --eval-batch 500000 --act leaky_relu --batch 50000 --patience 300 --n-layers-1 4 --n-layers-2 2 --root  ./ --seed 0 --hidden 1024
```

```
python main.py --method JK_GAMLP --epochs 700 --train-epochs 200 --input-drop 0.5 --att-drop 0.4 --dropout 0.2  --label-drop 0 --pre-process --dataset ogbn-products --num-runs 10 --eval-every 1 --eval-batch 50000 --act leaky_relu --batch 50000 --patience 300 --n-layers-1 2 --n-layers-2 2 --n-layers-3 2 --n-layers-4 2 --root  ./ --gpu 0 --seed 0 --hidden 1024
```

```
python main.py --use-label --method R_GAMLP --epochs 400 --train-epochs 0 --dataset ogbn-products --eval-every 10 --act leaky_relu --batch 50000 --eval-batch 500000 --patience 300 --n-layers-1 4 --n-layers-2 4 --n-layers-3 4 --n-layers-4 4 --num-hops 5 --label-num-hops 10 --input-drop 0.2 --att-drop 0.5 --label-drop 0 --pre-process --residual --gpu 0
```
```
python main.py --use-label --method JK_GAMLP --epochs 400 --train-epochs 0 --dataset ogbn-products --eval-every 10 --act leaky_relu --batch 50000 --eval-batch 500000 --patience 300 --n-layers-1 2 --n-layers-2 4 --n-layers-3 2 --n-layers-4 4 --num-hops 5 --label-num-hops 10 --input-drop 0.3 --att-drop 0.5 --label-drop 0 --pre-process --residual
```

### ogbn-papers100M

```
python preprocess_papers100m.py --num_hops 12 --root ./
```



```
python main_papers.py --method JK_GAMLP --epochs 400 --train-epochs 0 --dataset ogbn-papers100M --eval-every 1 --act sigmoid --batch 5000 --eval-batch 50000 --patience 60 --n-layers-1 4 --n-layers-2 6  --num-hops 12 --input-drop 0 --att-drop 0.5 --pre-process --hidden 1280 --lr 0.001 --root ./ --use-label --label-num-hops 9 --label-drop 0.3 --gpu 0
```

```
python main_papers.py --method R_GAMLP --epochs 400 --train-epochs 0 --dataset ogbn-papers100M --eval-every 1 --act sigmoid --batch 5000 --eval-batch 50000 --patience 60 --n-layers-1 4 --n-layers-2 6  --num-hops 12 --input-drop 0 --att-drop 0.5 --pre-process --hidden 1280 --lr 0.001 --root ./ --use-label --label-num-hops 9 --label-drop 0.3 --gpu 0
```

### Dataset resources

##### The PPI dataset can be fetched from http://snap.stanford.edu/graphsage/#datasets.

##### The OGB datasets can be fetched from https://ogb.stanford.edu/.

##### The other datasets can be fetched from https://github.com/GraphSAINT/GraphSAINT.

### Results

- Accuracy comparison:

  <img src="Table1.png" width="80%" height="80%">
  <img src="Table2.png" width="80%" height="80%">
  <img src="Table3.png" width="80%" height="80%">
  <img src="Table4.png" width="80%" height="80%">
  <img src="Table7.png" width="80%" height="80%">
- Efficiency comparision

  <img src="Table5.png" width="100%" height="80%">
- Ablation study

  <img src="Table6.png" width="80%" height="100%">
