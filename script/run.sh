# # get history
# python ../src/get_history.py --dataset ICEWS18
# python ../src/get_history.py --dataset ICEWS14s
# python ../src/get_history.py --dataset ICEWS05-15
# python ../src/get_history.py --dataset GDELT
# python ../src/get_history_ICEWS15.py --dataset YAGO
# python ../src/get_history_ICEWS15.py --dataset WIKI


# test
# # with static graph 


# # icews23
# python ../src/main.py -d ICEWS23 --history-rate 0.3 --identity-len 9 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint

# # icews22
# python ../src/main.py -d ICEWS22 --history-rate 0.3 --identity-len 9 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint

# # icews14s
# python ../src/main.py -d ICEWS14s --history-rate 0.3 --identity-len 9 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint

# # icews18
# python ../src/main.py -d ICEWS18 --history-rate 0.3 --identity-len 10 --train-history-len 10 --test-history-len 10 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 

# # WIKI
# python ../src/main.py -d WIKI --history-rate 0.3 --identity-len 2 --train-history-len 2 --test-history-len 2 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 

# # YAGO
# python ../src/main.py -d YAGO --history-rate 0.3 --identity-len 1 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 1 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 

# # icews05-15
# python ../src/main.py -d ICEWS05-15 --history-rate 0.3 --identity-len 15 --train-history-len 15 --test-history-len 15 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 

# # GDELT
# python ../src/main.py -d GDELT --history-rate 0.3 --identity-len 7 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 

# new abalation 3/17

# # icews23
# python ../src/main.py -d ICEWS23 --history-rate 0.3 --identity-len 1 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint

# # icews22
# python ../src/main.py -d ICEWS22 --history-rate 0.3 --identity-len 1 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint

# # icews14s
# python ../src/main.py -d ICEWS14s --history-rate 0.3 --identity-len 1 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint

# # icews18
# python ../src/main.py -d ICEWS18 --history-rate 0.3 --identity-len 1 --train-history-len 10 --test-history-len 10 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 

# # icews05-15
# python ../src/main.py -d ICEWS05-15 --history-rate 0.3 --identity-len 1 --train-history-len 15 --test-history-len 15 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 

# # GDELT
# python ../src/main.py -d GDELT --history-rate 0.3 --identity-len 1 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 

# # WIKI
# python ../src/main.py -d WIKI --history-rate 0.3 --identity-len 1 --train-history-len 2 --test-history-len 2 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 

# # YAGO
# python ../src/main.py -d YAGO --history-rate 0.3 --identity-len 1 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 1 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 


# # new abalation 3/18 (his_len=1, N_v取全部历史)
# # icews14s
# python ../src/main.py -d ICEWS14s --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint

# # icews18
# python ../src/main.py -d ICEWS18 --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 

# # icews05-15
# python ../src/main.py -d ICEWS05-15 --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 


# new abalation 3/24 (无图学习, N_v取全部历史)
# # icews23
# python ../src/main.py -d ICEWS23 --history-rate 0.3 --identity-len 10000 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint

# # icews22
# python ../src/main.py -d ICEWS22 --history-rate 0.3 --identity-len 10000 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint

# # icews14s
# python ../src/main.py -d ICEWS14s --history-rate 0.3 --identity-len 10000 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint

# # icews18
# python ../src/main.py -d ICEWS18 --history-rate 0.3 --identity-len 10000 --train-history-len 10 --test-history-len 10 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 

# # icews05-15
# python ../src/main.py -d ICEWS05-15 --history-rate 0.3 --identity-len 10000 --train-history-len 15 --test-history-len 15 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 

# # GDELT
# python ../src/main.py -d GDELT --history-rate 0.3 --identity-len 10000 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 

# # WIKI
# python ../src/main.py -d WIKI --history-rate 0.3 --identity-len 2 --train-history-len 2 --test-history-len 2 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 

# # YAGO
# python ../src/main.py -d YAGO --history-rate 0.3 --identity-len 1 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 1 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 

# # 2024/06/03 with different training set use ratio
# # icews14s
# # python ../src/main.py -d ICEWS14s --train-use-ratio 0.1 --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint
# python ../src/main.py -d ICEWS14s --train-use-ratio 0.2 --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint
# python ../src/main.py -d ICEWS14s --train-use-ratio 0.3 --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint
# python ../src/main.py -d ICEWS14s --train-use-ratio 0.4 --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint
# python ../src/main.py -d ICEWS14s --train-use-ratio 0.5 --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint
# python ../src/main.py -d ICEWS14s --train-use-ratio 0.6 --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint
# python ../src/main.py -d ICEWS14s --train-use-ratio 0.7 --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint
# python ../src/main.py -d ICEWS14s --train-use-ratio 0.8 --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint
# python ../src/main.py -d ICEWS14s --train-use-ratio 0.9 --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint

# # icews18
# python ../src/main.py -d ICEWS18  --train-use-ratio 0.1 --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d ICEWS18  --train-use-ratio 0.2 --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d ICEWS18  --train-use-ratio 0.3 --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d ICEWS18  --train-use-ratio 0.4 --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d ICEWS18  --train-use-ratio 0.5 --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d ICEWS18  --train-use-ratio 0.6 --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d ICEWS18  --train-use-ratio 0.7 --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d ICEWS18  --train-use-ratio 0.8 --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d ICEWS18  --train-use-ratio 0.9 --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 


# # GDELT
# python ../src/main.py -d GDELT --train-use-ratio 0.1 --history-rate 0.3 --identity-len 10000 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d GDELT --train-use-ratio 0.2 --history-rate 0.3 --identity-len 10000 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d GDELT --train-use-ratio 0.3 --history-rate 0.3 --identity-len 10000 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d GDELT --train-use-ratio 0.4 --history-rate 0.3 --identity-len 10000 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d GDELT --train-use-ratio 0.5 --history-rate 0.3 --identity-len 10000 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d GDELT --train-use-ratio 0.6 --history-rate 0.3 --identity-len 10000 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d GDELT --train-use-ratio 0.7 --history-rate 0.3 --identity-len 10000 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d GDELT --train-use-ratio 0.8 --history-rate 0.3 --identity-len 10000 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d GDELT --train-use-ratio 0.9 --history-rate 0.3 --identity-len 10000 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 



# # icews05-15
# python ../src/main.py -d ICEWS05-15 --history-rate 0.3 --identity-len 10000 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 

# 2024/06/14 with different training set use ratio
# icews14s
python ../src/main.py -d ICEWS14s --train-use-ratio 0.1 --history-rate 0.3 --identity-len 9 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint
python ../src/main.py -d ICEWS14s --train-use-ratio 0.2 --history-rate 0.3 --identity-len 9 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint
python ../src/main.py -d ICEWS14s --train-use-ratio 0.3 --history-rate 0.3 --identity-len 9 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint
python ../src/main.py -d ICEWS14s --train-use-ratio 0.4 --history-rate 0.3 --identity-len 9 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint
python ../src/main.py -d ICEWS14s --train-use-ratio 0.5 --history-rate 0.3 --identity-len 9 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint
python ../src/main.py -d ICEWS14s --train-use-ratio 0.6 --history-rate 0.3 --identity-len 9 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint
python ../src/main.py -d ICEWS14s --train-use-ratio 0.7 --history-rate 0.3 --identity-len 9 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint
python ../src/main.py -d ICEWS14s --train-use-ratio 0.8 --history-rate 0.3 --identity-len 9 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint
python ../src/main.py -d ICEWS14s --train-use-ratio 0.9 --history-rate 0.3 --identity-len 9 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu 0 --save checkpoint

# icews18
python ../src/main.py -d ICEWS18  --train-use-ratio 0.1 --history-rate 0.3 --identity-len 10 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
python ../src/main.py -d ICEWS18  --train-use-ratio 0.2 --history-rate 0.3 --identity-len 10 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
python ../src/main.py -d ICEWS18  --train-use-ratio 0.3 --history-rate 0.3 --identity-len 10 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
python ../src/main.py -d ICEWS18  --train-use-ratio 0.4 --history-rate 0.3 --identity-len 10 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
python ../src/main.py -d ICEWS18  --train-use-ratio 0.5 --history-rate 0.3 --identity-len 10 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
python ../src/main.py -d ICEWS18  --train-use-ratio 0.6 --history-rate 0.3 --identity-len 10 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
python ../src/main.py -d ICEWS18  --train-use-ratio 0.7 --history-rate 0.3 --identity-len 10 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
python ../src/main.py -d ICEWS18  --train-use-ratio 0.8 --history-rate 0.3 --identity-len 10 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
python ../src/main.py -d ICEWS18  --train-use-ratio 0.9 --history-rate 0.3 --identity-len 10 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --add-static-graph --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 


# # GDELT
# python ../src/main.py -d GDELT --train-use-ratio 0.1 --history-rate 0.3 --identity-len 7 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d GDELT --train-use-ratio 0.2 --history-rate 0.3 --identity-len 7 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d GDELT --train-use-ratio 0.3 --history-rate 0.3 --identity-len 7 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d GDELT --train-use-ratio 0.4 --history-rate 0.3 --identity-len 7 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d GDELT --train-use-ratio 0.5 --history-rate 0.3 --identity-len 7 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d GDELT --train-use-ratio 0.6 --history-rate 0.3 --identity-len 7 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d GDELT --train-use-ratio 0.7 --history-rate 0.3 --identity-len 10000 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d GDELT --train-use-ratio 0.8 --history-rate 0.3 --identity-len 10000 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 
# python ../src/main.py -d GDELT --train-use-ratio 0.9 --history-rate 0.3 --identity-len 10000 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --task-weight 1 --gpu 0 --save checkpoint 

