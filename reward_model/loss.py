import torch
import torch.nn.functional as F

def log_sigmoid_loss(chosen_reward, reject_reward) : 
    return -F.logsigmoid(chosen_reward - reject_reward).mean()