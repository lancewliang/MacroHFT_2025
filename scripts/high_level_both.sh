 

nohup python -u RL/agent/high_level.py --buffer_size 3000000 --exp 'both_action_stage_2' --dataset 'ETHUSDT' --transcation_cost 0.0005 --action_mode both --action_size 2 --reward_no_action False >./logs/high_level/ETHUSDT/both_action_stage_2.log 2>&1 &
