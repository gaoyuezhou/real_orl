import numpy as np
import pickle
import torch

class D3Agent():
    def __init__(self, policy, config, device):
        self.policy = policy
        self.config = config
        self.device = device

    # For 1-batch query only!
    def predict(self, sample):
        with torch.no_grad():
            input = torch.from_numpy(sample['inputs']).float().unsqueeze(0).to(self.device)
            at = self.policy(input)[0].to('cpu').detach().numpy()
        return at

def _init_agent_from_config(config, device):
    device='cpu'
    policy = torch.jit.load(config.agent.policy_pt)
    policy.to(device)

    return D3Agent(policy, config, device)