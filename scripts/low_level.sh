
nohup python -u RL/agent/low_level.py --alpha 4 --exp 'short_action' --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0005 --action_mode short --action_size 1 --label label_2 >./logs/low_level/ETHUSDT/vol_2_short.log 2>&1 &

