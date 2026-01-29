 

nohup python -u RL/agent/high_level.py --buffer_size 2200000 --exp 'short_action_stage2' --dataset 'ETHUSDT' --transcation_cost 0.0005 --action_mode short --action_size 1 --reward_no_action False >./logs/high_level/ETHUSDT/short_action_stage2.log 2>&1 &
