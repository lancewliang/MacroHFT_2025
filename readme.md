# MacroHFT
This is the official implementation of the KDD 2024 "MacroHFT: Memory Augmented Context-aware Reinforcement Learning On High Frequency Trading".
https://arxiv.org/abs/2406.14537

To run the demo code:

You may first download the dataset from Google Drive:

https://drive.google.com/drive/folders/1AYHy-wUV0IwPoA7E1zvMRPL3wK0tPNiY?usp=drive_link

and put the folder under data folder.

## Step 1
Run scripts/decomposition.sh for data decomposition and labeling. 
## Step 2
Run scripts/low_level.sh for low-level policy optimization. 

Update: We now provide trained model checkpoints for sub-agents, which can be directly used to train meta-policy.
## Step 3
Run scripts/high_level.sh for meta-policy optimization. 

After 4 months of unremitting efforts, mainly due to the lack of computing power equipment, I finally successfully reproduced the test results of low-level and high-level in your paper. Thank you for releasing all the code, algorithms, and data. Sincere tribute.
I will continue to try to improve some parts of this framework, including:
--More actions, more position levels  （已经测试过了， 不太可能成功，由于先验经验一定是满仓最大优势，导致学习不到半仓的好处，除非更换奖励函数为夏普之类的长期奖励，暂时不作研究）
--Due to the low trading frequency in my testing results, （在没有持仓的时候增加了惩罚，交易频率上升了）
--I tend to remove order data from the micro market for testing purposes
--Use LSTM or Transformer to observe the effect at the model level 
--Attempt to investigate the impact of ranbowdqn on the model
--Incorporate spot prices into the factor,
--Short selling strategy, one-way short selling or as protection for long positions （可行）
--Test buying spot goods （放弃，由于现货交易的手续费较高0.3%，而我在测试中只关注了期货交易的收益，所以放弃了测试现货交易的效果，之后会考虑）