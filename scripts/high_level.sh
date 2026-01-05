# nohup python -u RL/agent/high_level.py --buffer_size 1500000 --exp 'both_action_3' --dataset 'ETHUSDT' --transcation_cost 0.0005 --action_mode both --action_size 2 --reward_no_action False >./logs/high_level/ETHUSDT/both_action_1.log 2>&1 &

#nohup python -u RL/agent/high_level.py --exp 'long_action' --dataset 'ETHUSDT' --transcation_cost 0.0005 --action_mode long --action_size 1 >./logs/high_level/ETHUSDT/long_action_1.log 2>&1 &
#nohup python -u RL/agent/high_level.py --exp 'exp4_1' --dataset 'ETHUSDT' --transcation_cost 0.0004 --action_mode long --action_size 1 --reward_no_action True >./logs/high_level/ETHUSDT/exp4_1.log 2>&1 &
#nohup python -u RL/agent/high_level.py --exp 'exp4_2' --dataset 'ETHUSDT' --transcation_cost 0.0004 --action_mode long --action_size 1 --reward_no_action False >./logs/high_level/ETHUSDT/exp4_2.log 2>&1 &
nohup python -u RL/agent/high_level.py --buffer_size 1000000 --exp 'short_action_n' --dataset 'ETHUSDT' --transcation_cost 0.0005 --action_mode short --action_size 1 --reward_no_action False >./logs/high_level/ETHUSDT/short_action_n.log 2>&1 &

#nohup python -u RL/agent/high_level.py --exp 'short_action' --dataset 'ETHUSDT' --transcation_cost 0.0005 --action_mode short --action_size 1 >./logs/high_level/ETHUSDT/short_action_1.log 2>&1 &
