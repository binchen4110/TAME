# NEDA

This is the code of NEDA.


### Installation
```
conda create -n NEDA python=3.9

conda activate NEDA

pip install -r requirement.txt
```



## How to run

#### Process data

For all the datasets, the following command can be used to get the history of their entities and relations.
```
cd src
python get_history.py --dataset ICEWS14
```


#### Train models

Then the following commands can be used to train TiRGN.

Train models

```
python ../src/main.py -d ICEWS14s --train-use-ratio 1 --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint

```

###### Test models

```
python ../src/main.py -d ICEWS14s --train-use-ratio 1 --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint --test 
```





