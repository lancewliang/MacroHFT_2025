nohup python -u RL/agent/high_level.py --subagent_hidden_size=512 --buffer_size 2200000 --exp 'short_action_2023-2025_5_20260202' --dataset 'ETHUSDT' --transcation_cost 0.0005 --action_mode short --action_size 1 --reward_no_action False >./logs/high_level/ETHUSDT/short_action_2023-2025_5_20260202.log 2>&1 &


# nohup python -u RL/agent/high_level.py --buffer_size 1000000 --exp 'short_action_n' --dataset 'ETHUSDT' --transcation_cost 0.0005 --action_mode short --action_size 1 --reward_no_action False >./logs/high_level/ETHUSDT/short_action_n.log 2>&1 &
