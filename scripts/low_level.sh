nohup python -u RL/agent/low_level.py --alpha 0.5 --exp 'long_action' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0005 --action_mode long --action_size 1 --label label_1 >./logs/low_level/ETHUSDT/slope_1_long.log 2>&1 &
nohup python -u RL/agent/low_level.py --alpha 4 --exp 'long_action' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0005 --action_mode long --action_size 1 --label label_2 >./logs/low_level/ETHUSDT/slope_2_long.log 2>&1 &
nohup python -u RL/agent/low_level.py --alpha 0.5 --exp 'long_action' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0005 --action_mode long --action_size 1 --label label_3 >./logs/low_level/ETHUSDT/slope_3_long.log 2>&1 &

