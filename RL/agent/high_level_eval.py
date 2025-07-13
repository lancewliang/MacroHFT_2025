
import logging as log
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from env.high_level_env import Testing_Env, Training_Env
import numpy as np
from model.net import *
import matplotlib.pyplot as plt


class DQN_EVAL(object):
    def __init__(self, n_state_1,n_state_2,n_action,
                 device,
                 clf_list,
                 test_data_path,
                 val_data_path,
                 tech_indicator_list,
                 tech_indicator_list_trend,
                 transcation_cost,
                 back_time_length,
                 max_holding_number,
                 slope_agents,
                 vol_agents
                 ):  
        self.device = device
        self.vol_agents = vol_agents
        self.slope_agents = slope_agents
        self.clf_list = clf_list
        self.test_data_path = test_data_path
        self.val_data_path = val_data_path
        self.hyperagent = hyperagent(n_state_1, n_state_2, n_action, 32).to(self.device)
        self.tech_indicator_list=tech_indicator_list
        self.tech_indicator_list_trend=tech_indicator_list_trend
        self.transcation_cost=transcation_cost
        self.back_time_length=back_time_length
        self.max_holding_number=max_holding_number
        
    
    def act_test(self, state, state_trend, state_clf, info):
        """
        测试模式下的动作选择（无随机性）
        
        Args:
            state: 当前状态
            state_trend: 状态趋势
            state_clf: 状态分类特征
            info: 包含历史动作等信息的字典
            
        Returns:
            int: 确定性选择的最优动作
        """
        with torch.no_grad():
            x1 = torch.FloatTensor(state).to(self.device)
            x2 = torch.FloatTensor(state_trend).to(self.device)
            x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device)
            previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long().to(self.device), 0).to(self.device)
            qs = [
                    self.slope_agents[0](x1, x2, previous_action),
                    self.slope_agents[1](x1, x2, previous_action),
                    self.slope_agents[2](x1, x2, previous_action),
                    self.vol_agents[0](x1, x2, previous_action),
                    self.vol_agents[1](x1, x2, previous_action),
                    self.vol_agents[2](x1, x2, previous_action)
            ]
            w = self.hyperagent(x1, x2, x3, previous_action)
            actions_value = self.calculate_q(w, qs)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
            return action
    def calculate_q(self, w, qs):
        q_tensor = torch.stack(qs)
        q_tensor = q_tensor.permute(1, 0, 2)
        weights_reshaped = w.view(-1, 1, 6)
        combined_q = torch.bmm(weights_reshaped, q_tensor).squeeze(1)
        
        return combined_q 
    def val_cluster(self, epoch_path, save_path):
        self.hyperagent.load_state_dict(torch.load(os.path.join(epoch_path, "trained_model.pkl")))
        self.hyperagent.eval()
        counter = False
        action_list = []
        reward_list = []
        final_balance_list = []
        required_money_list = []
        commission_fee_list = []
        self.df = pd.read_feather(os.path.join(self.val_data_path, "val.feather"))
        log.info(f"validating on df")
        val_env = Testing_Env(
                df=self.df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                clf_list=self.clf_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                initial_action=0)
        s, s2, s3, info = val_env.reset()
        log.info(f"validating val_env reset")
        done = False
        action_list_episode = []
        reward_list_episode = []
        while not done:            
            a = self.act_test(s, s2, s3, info)
            s_, s2_, s3_, r, done, info_ = val_env.step(a)
            reward_list_episode.append(r)
            s, s2, s3, info = s_, s2_, s3_, info_
            action_list_episode.append(a)
        portfit_magine, final_balance, required_money, commission_fee = val_env.get_final_return_rate(slient=True)
        final_balance = val_env.final_balance
        action_list.append(action_list_episode)
        reward_list.append(reward_list_episode)
        final_balance_list.append(final_balance)
        required_money_list.append(required_money)
        commission_fee_list.append(commission_fee)
        action_list = np.array(action_list)
        reward_list = np.array(reward_list)
        final_balance_list = np.array(final_balance_list)
        required_money_list = np.array(required_money_list)
        commission_fee_list = np.array(commission_fee_list)
        np.save(os.path.join(save_path, "action_val.npy"), action_list)
        np.save(os.path.join(save_path, "reward_val.npy"), reward_list)
        np.save(os.path.join(save_path, "final_balance_val.npy"), final_balance_list)
        np.save(os.path.join(save_path, "require_money_val.npy"), required_money_list)
        np.save(os.path.join(save_path, "commission_fee_history_val.npy"), commission_fee_list)
        return_rate = final_balance / required_money
        return return_rate

    def test_cluster(self, epoch_path, save_path):
        self.hyperagent.load_state_dict(torch.load(epoch_path))
        self.hyperagent.eval()
        counter = False
        action_list = []
        reward_list = []
        final_balance_list = []
        required_money_list = []
        commission_fee_list = []
        price_list = []
        self.df = pd.read_feather(os.path.join(self.test_data_path, "test.feather"))
        
        test_env = Testing_Env(
                df=self.df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                clf_list=self.clf_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                initial_action=0)
        s, s2, s3, info = test_env.reset()
        done = False
        total_reward = 0
        # action_list_episode = []
        # reward_list_episode = []
        while not done:
            a = self.act_test(s, s2, s3, info)
            s_, s2_, s3_, r, done, info_ = test_env.step(a)
            # reward_list_episode.append(r)
            s, s2, s3, info = s_, s2_, s3_, info_
            # action_list_episode.append(a)
            portfit_magine, final_balance, required_money, commission_fee = test_env.get_final_return_rate(slient=True)
            # final_balance = test_env.final_balance
            
            price = info_['close']
            position = info_['position']
            action_list.append(a)
            reward_list.append(r)
            total_reward = total_reward+r
            final_balance_list.append(final_balance)
            required_money_list.append(required_money)
            commission_fee_list.append(commission_fee)
            price_list.append(price)
            log.info (f"price:{price},action:{a},position:{position},reward:{r},total_reward:{total_reward},balance:{final_balance},commission_fee:{commission_fee},required_money:{required_money}")

        action_list = np.array(action_list)
        reward_list = np.array(reward_list)
        final_balance_list = np.array(final_balance_list)
        required_money_list = np.array(required_money_list)
        commission_fee_list = np.array(commission_fee_list)
        price_list = np.array(price_list)
        
        log.info (f"{portfit_magine},{final_balance},{required_money},{commission_fee}")
        np.save(os.path.join(save_path, "action.npy"), action_list)
        np.save(os.path.join(save_path, "reward.npy"), reward_list)
        np.save(os.path.join(save_path, "final_balance.npy"), final_balance_list)
        np.save(os.path.join(save_path, "require_money.npy"), required_money_list)
        np.save(os.path.join(save_path, "commission_fee_history.npy"), commission_fee_list)
        
   
        plt.figure(figsize=(14, 8))

        plt.subplot(2, 3, 1)
        plt.plot(action_list)
        plt.title('Action List')
        plt.xlabel('Step')
        plt.ylabel('Action')

        plt.subplot(2, 3, 2)
        plt.plot(reward_list)
        plt.title('Reward List')
        plt.xlabel('Step')
        plt.ylabel('Reward')

        plt.subplot(2, 3, 3)
        plt.plot(final_balance_list)
        plt.title('Final Balance List')
        plt.xlabel('Step')
        plt.ylabel('Balance')

        plt.subplot(2, 3, 4)
        plt.plot(required_money_list)
        plt.title('Required Money List')
        plt.xlabel('Step')
        plt.ylabel('Required Money')

        plt.subplot(2, 3, 5)
        plt.plot(commission_fee_list)
        plt.title('Commission Fee List')
        plt.xlabel('Step')
        plt.ylabel('Fee')
        
        # 新增price_list可视化
        plt.subplot(2, 3, 6)
        plt.plot(price_list)
        plt.title('Price List')
        plt.xlabel('Step')
        plt.ylabel('Price')
        
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'evaluation_metrics.png') )
        plt.close()