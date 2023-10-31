import d3rlpy
import pickle
import numpy as np
import argparse
import os
import collections

def get_dataset(dataset_fn):
    raw_dataset = pickle.load(open(dataset_fn, 'rb'))
    observations = np.concatenate([traj['observations'] for traj in raw_dataset])
    actions = np.concatenate([traj['actions'] for traj in raw_dataset])
    rewards = np.concatenate([traj['rewards'] for traj in raw_dataset])
    terminals = np.zeros(observations.shape[0])

    episode_terminals = np.zeros_like(terminals)
    idx = 0
    for traj in raw_dataset:
        idx += traj['rewards'].shape[0]
        episode_terminals[idx-1] = 1.0

    return d3rlpy.dataset.MDPDataset(
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=np.array(terminals, dtype=np.float32),
        episode_terminals=np.array(episode_terminals, dtype=np.float32),
    )

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        help="Config file (yaml)",
                        default="bc.yaml")
    parser.add_argument("-d", "--dataset",
                        default='')
    parser.add_argument("-o", "--output", default="tmp")
    parser.add_argument("-s", "--seed", type=int, default=123)
    return parser.parse_args()

class Namespace(collections.MutableMapping):
    def __init__(self, data):
        self._data = data

    def __getitem__(self, k):
        return self._data[k]

    def __setitem__(self, k, v):
        self._data[k] = v

    def __delitem__(self, k):
        del self._data[k]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getattr__(self, k):
        if not k.startswith('_'):
            if k not in self._data:
                return Namespace({})
            v = self._data[k]
            if isinstance(v, dict):
                v = Namespace(v)
            return v

        if k not in self.__dict__:
            raise AttributeError("'Namespace' object has no attribute '{}'".format(k))

        return self.__dict__[k]

    def __repr__(self):
        return repr(self._data)

args = get_args()
with open(args.config) as f:
    import yaml
    config = yaml.load(f, Loader=yaml.FullLoader)
    config = Namespace(config)
    if args.dataset != '':
        config.dataset = args.dataset
output_dir = args.dataset.split('/')[-1].split('.')[0] + '-' + str(args.seed)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

dataset = get_dataset(config.dataset)

import torch
SEED = args.seed
np.random.seed(SEED)
torch.random.manual_seed(SEED)

agent_class_name = f'd3rlpy.algos.{config.agent}'
agent = eval(agent_class_name)( **dict(config.learner))

agent.fit(dataset, n_steps=config.training.n_steps, logdir=output_dir)
agent.save_policy(os.path.join(output_dir, 'policy.pt'))