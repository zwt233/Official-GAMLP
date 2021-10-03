# GAMLP-V2

##### PPI

```
cd ./ppi
python ppi.py --dataset ppi --method JK_GAMLP --dropout 0.1 --input-drop 0.1 --att-drop 0 --act sigmoid --label-num-hops 3 --num-hops 10 --epochs 800  --n-layers-1 4 --n-layers-2 6 --lr 0.001 --weight-decay 0 --hidden 2048 --seed 42 --att-drop 0 --label-num-hops 9 --use-label --patience 100
```

##### Validation 9840

##### Test 9982

````
cd ./ppi
python ppi.py --dataset ppi --method R_GAMLP --dropout 0.1 --input-drop 0.1 --att-drop 0 --act sigmoid --label-num-hops 3 --num-hops 10 --epochs 600  --n-layers-1 4 --n-layers-2 6 --lr 0.001 --weight-decay 0 --hidden 2048 --seed 42 --att-drop 0 --label-num-hops 9 --use-label --patience 100
````

##### Validation 9827

##### Test 9970

### Reddit

```
python main.py --dataset reddit --method JK_GAMLP --dropout 0.7 --input-drop 0 --att-drop 0 --act leaky_relu --batch 1000 --label-num-hops 10 --num-hops 10 --epochs 350 --n-layers-1 2 --n-layers-2 2 --n-layers-3 2 --n-layers-4 2 --lr 0.0001 --weight-decay 0 --hidden 512 --patience 100 --seed 42 --pre-process --use-label --label-drop 0.3 --label-num-hops 4 --weight 1e-7 --num-runs 1 
```

##### Validation 9714

#####  Test 97.04

```
python main.py --dataset reddit --method R_GAMLP --dropout 0.7 --input-drop 0 --att-drop 0 --act leaky_relu --batch 1000 --label-num-hops 10 --num-hops 10 --epochs 350 --n-layers-1 2 --n-layers-2 2 --n-layers-3 2 --n-layers-4 2 --lr 0.0001 --weight-decay 0 --hidden 512 --patience 100 --seed 42 --pre-process --use-label --label-drop 0.5 --label-num-hops 4 --weight 1e-7  --num-runs 1 
```

##### Validation 96.81

##### Test 96.62

### Flickr

```
python main.py --dataset flickr --method JK_GAMLP --dropout 0.7 --input-drop 0 --att-drop 0 --act leaky_relu --batch 250 --label-num-hops 10 --num-hops 10 --epochs 50 --n-layers-1 2 --n-layers-2 2 --n-layers-3 2 --n-layers-4 2 --lr 0.001 --weight-decay 0 --hidden 512 --patience 100 --seed 42 --pre-process --use-label --label-drop 0.5 --label-num-hops 5 --num-runs 1
```

##### Validation 5412

##### Test 5403

```
python main.py --dataset flickr --method R_GAMLP --dropout 0.7 --input-drop 0 --att-drop 0 --act leaky_relu --batch 250 --label-num-hops 10 --num-hops 10 --epochs 50 --n-layers-1 2 --n-layers-2 2 --n-layers-3 2 --n-layers-4 2 --lr 0.0001 --weight-decay 0 --hidden 512 --patience 100 --seed 42 --pre-process --use-label --label-drop 0.5 --label-num-hops 5 --num-runs 1
```



### ogbn-products

```
python main.py --method R_GAMLP --epochs 1000 --input-drop 0.5 --att-drop 0.4 --dropout 0.2  --label-drop 0 --pre-process --dataset ogbn-products --num-runs 10 --gpu 6 --eval-every 1 --eval-batch 500000 --act leaky_relu --batch 50000 --patience 300 --n-layers-1 4 --n-layers-2 2 --root  /data1/zwt/ --gpu 0 --seed 0 --hidden 1024
```

##### Average val accuracy: 0.9231, std: 0.0006
##### Average test accuracy: 0.8141, std: 0.0025

```
python main.py --method JK_GAMLP --epochs 700 --train-epochs 200 --input-drop 0.5 --att-drop 0.4 --dropout 0.2  --label-drop 0 --pre-process --dataset ogbn-products --num-runs 10 --gpu 6 --eval-every 1 --eval-batch 50000 --act leaky_relu --batch 50000 --patience 300 --n-layers-1 2 --n-layers-2 2 --n-layers-3 2 --n-layers-4 2 --root  /data2/zwt/ --gpu 0 --seed 0 --hidden 1024
```

#####  Params: 15477863

##### Average val accuracy: 0.9229, std: 0.0006
##### Average test accuracy: 0.8143, std: 0.0018



```
python main.py --use-label --method R_GAMLP --epochs 400 --train-epochs 0 --dataset ogbn-products --eval-every 10 --act leaky_relu --batch 50000 --eval-batch 500000 --patience 300 --n-layers-1 4 --n-layers-2 4 --n-layers-3 4 --n-layers-4 4 --num-hops 5 --label-num-hops 10 --input-drop 0.2 --att-drop 0.5 --label-drop 0 --pre-process --residual
```
##### Params: 3338135

##### Average val accuracy: 0.9314, std: 0.0006
##### Average test accuracy: 0.8360, std: 0.0004

```
python main.py --use-label --method JK_GAMLP --epochs 400 --train-epochs 0 --dataset ogbn-products --eval-every 10 --act leaky_relu --batch 50000 --eval-batch 500000 --patience 300 --n-layers-1 2 --n-layers-2 4 --n-layers-3 2 --n-layers-4 4 --num-hops 5 --label-num-hops 10 --input-drop 0.3 --att-drop 0.5 --label-drop 0 --pre-process --residual
```

##### Params: 5440967

##### Average val accuracy: 0.9319, std: 0.0003
##### Average test accuracy: 0.8332, std: 0.0025

#### ogbn-100Mpapers

```
python main_papers.py --method JK_GAMLP --epochs 400 --train-epochs 0 --dataset ogbn-papers100M --eval-every 1 --act sigmoid --batch 5000 --eval-batch 50000 --patience 60 --n-layers-1 4 --n-layers-2 6  --num-hops 12 --input-drop 0 --att-drop 0.5 --pre-process --hidden 1280 --lr 0.001 --root /data2/zwt/ --use-label --label-num-hops 9 --label-drop 0.3
```

##### Params: 67560875

###### seed 0 7188 6815

###### seed 1 7190 6792
###### seed 2 7197 6807

###### seed 3 7176 6795	

###### seed 4 7188 6810

```
python main_papers.py --method R_GAMLP --epochs 400 --train-epochs 0 --dataset ogbn-papers100M --eval-every 1 --act sigmoid --batch 5000 --eval-batch 50000 --patience 60 --n-layers-1 4 --n-layers-2 6  --num-hops 12 --input-drop 0 --att-drop 0.5 --pre-process --hidden 1280 --lr 0.001 --root /data2/zwt/ --use-label --label-num-hops 9 --label-drop 0.3
```



### Dataset resources

##### the PPI dataset can get from http://snap.stanford.edu/graphsage/#datasets

##### the other datasets can get from https://github.com/GraphSAINT/GraphSAINT

