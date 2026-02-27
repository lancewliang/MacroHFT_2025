#nohup python -u RL/agent/high_level.py --subagent_hidden_size=512 --buffer_size 2200000 --exp 'both_action_2023-2025_2_20260201' --dataset 'ETHUSDT' --transcation_cost 0.0005 --action_mode both --action_size 2 --reward_no_action False >./logs/high_level/ETHUSDT/both_action_2023-2025_2_20260201.log 2>&1 &

nohup python -u RL/agent/high_level.py --subagent_hidden_size=512 --buffer_size 3000000 --exp 'both_action_2023-2025_2_20260201' --dataset 'ETHUSDT' --transcation_cost 0.0002 --action_mode both --action_size 2 --reward_no_action False >./logs/high_level/ETHUSDT/both_action_2023-2025_2_20260201.log 2>&1 &

