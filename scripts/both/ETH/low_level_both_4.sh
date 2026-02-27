
nohup python -u RL/agent/low_level.py --lr 0.0001 --alpha 1 --exp 'both_action_stage_2' --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0004 --action_mode both --action_size 2 --label label_1 --reward_no_action False >./logs/low_level/ETHUSDT/vol_1_both_stage_2.log 2>&1 &
nohup python -u RL/agent/low_level.py --lr 0.0001 --alpha 1 --exp 'both_action_stage_2' --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0004 --action_mode both --action_size 2 --label label_2 --reward_no_action False >./logs/low_level/ETHUSDT/vol_2_both_stage_2.log 2>&1 &
nohup python -u RL/agent/low_level.py --lr 0.0001 --alpha 1 --exp 'both_action_stage_2' --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0004 --action_mode both --action_size 2 --label label_3 --reward_no_action False >./logs/low_level/ETHUSDT/vol_3_both_stage_2.log 2>&1 &

nohup python -u RL/agent/low_level.py --lr 0.0001 --alpha 1 --exp 'both_action_stage_2' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0004 --action_mode both --action_size 2 --label label_1 --reward_no_action False >./logs/low_level/ETHUSDT/slope_1_both_stage_2.log 2>&1 &
nohup python -u RL/agent/low_level.py --lr 0.0001 --alpha 1 --exp 'both_action_stage_2' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0004 --action_mode both --action_size 2 --label label_2 --reward_no_action False >./logs/low_level/ETHUSDT/slope_2_both_stage_2.log 2>&1 &
nohup python -u RL/agent/low_level.py --lr 0.0001 --alpha 1 --exp 'both_action_stage_2' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0004 --action_mode both --action_size 2 --label label_3 --reward_no_action False >./logs/low_level/ETHUSDT/slope_3_both_stage_2.log 2>&1 &

