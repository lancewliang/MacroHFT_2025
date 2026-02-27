 

nohup python -u RL/agent/high_level.py --buffer_size 3000000 --exp 'both_action_2023-2025_5_20260130' --dataset 'ETHUSDT' --transcation_cost 0.0005 --action_mode both --action_size 2 --reward_no_action False >./logs/high_level/ETHUSDT/both_action_2023-2025_5_20260130.log 2>&1 &
