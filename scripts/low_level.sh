
#nohup python -u RL/agent/low_level.py --alpha 1 --exp 'exp4_1' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --label label_2 >./logs/low_level/ETHUSDT/slope_1.log 2>&1 &
# nohup python -u RL/agent/low_level.py --alpha 1 --exp 'exp4_1' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0002 --action_mode both --label label_2 >./logs/low_level/ETHUSDT/slope_4_1.log 2>&1 &
nohup python -u RL/agent/low_level.py --alpha 1 --exp 'both_action' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0005 --action_mode both --action_size 2 --label label_2 >./logs/low_level/ETHUSDT/1.log 2>&1 &
#nohup python -u RL/agent/low_level.py --alpha 1 --exp 'long_action' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0005 --action_mode long --action_size 1 --label label_2 >./logs/low_level/ETHUSDT/2.log 2>&1 &
#nohup python -u RL/agent/low_level.py --alpha 1 --exp 'short_action' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0005 --action_mode short --action_size 1 --label label_2 >./logs/low_level/ETHUSDT/3.log 2>&1 &
#nohup python -u RL/agent/low_level.py --alpha 3 --exp 'exp4_3' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --label label_2 >./logs/low_level/ETHUSDT/slope_3.log 2>&1 &
# nohup python -u RL/agent/low_level.py --alpha 4 --exp 'exp4_4' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --label label_1 >./logs/low_level/ETHUSDT/slope_4.log 2>&1 &
# nohup python -u RL/agent/low_level.py --alpha 0 --exp 'exp4_0' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --label label_1 >./logs/low_level/ETHUSDT/slope_0.log 2>&1 &
