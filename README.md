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

The following command can be used to get the history and natural division of the queries.
```
cd src
python get_division_history.py --dataset GDELT
```


#### Train model

```
python ../src/main.py -d GDELT --train-use-ratio 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 


```

#### Test models

```
python ../src/main.py -d GDELT --train-use-ratio 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 

```





