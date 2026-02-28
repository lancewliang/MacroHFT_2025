
nohup python -u RL/agent/high_level.py --beta 2 --alpha 1 --lr 0.0001 --subagent_hidden_size=128 --buffer_size 1000000 --exp 'both_action_2_30s' --dataset 'MRMB' --transcation_cost 0.0002 --action_mode both --action_size 2 --reward_no_action False >./logs/high_level/MRMB/both_action_2_30s.log 2>&1 &



nohup python -u RL/agent/high_level.py --beta 2 --alpha 1 --lr 0.0001 --subagent_hidden_size=128 --buffer_size 1000000 --exp 'both_action_5_30s' --dataset 'MRMB' --transcation_cost 0.0005 --action_mode both --action_size 2 --reward_no_action False >./logs/high_level/MRMB/both_action_5_30s.log 2>&1 &
