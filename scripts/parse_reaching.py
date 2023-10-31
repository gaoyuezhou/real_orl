import os
import pickle
from glob import glob
import pandas as pd
import numpy as np
import argparse
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--logs_folder",
                        type=str,
                        help="The folder that contains all logs to parse",
                        default="/home/franka/dev/franka_demo/logs/")
    parser.add_argument("-o", "--output_pickle_name",
                        type=str,
                        default="parsed_data.pkl")
    parser.add_argument("-c", "--ignore_camera",
                        help="When present, expect to NOT parse camera images",
                        action="store_true")
    parser.add_argument("-d", "--include_depth",
                        help="When expect, include depth camera images",
                        action="store_true")
    parser.add_argument("-t", "--ignore_tracking",
                        help="When present, expect to NOT get marker positions from images",
                        action="store_true")
    #parser.add_argument('-m','--marker', nargs='*', help='marker ids', type=int)
    return parser.parse_args()

def remove_outliers(data,thresh=2.0):
    m = np.median(data, axis=0)
    s = np.abs(data - m).sum(axis=1)
    return data[(s < (np.median(s)+1e-6) * thresh)]


if __name__ == "__main__":
    args = get_args()
    logs_folder = args.logs_folder + ('/' if args.logs_folder[-1] != '/' else '')

    args.output_pickle_name = logs_folder[:-1].split('/')[-1] + '.pkl'
    print("output name: ", args.output_pickle_name)

    logs_folder += '[0-9]*[0-9]/'
    name_of_demos = glob(logs_folder)
    print(f"Found {len(name_of_demos)} recordings in the {logs_folder}")

    if len(name_of_demos) == 0:
        exit()
    if os.path.isfile(args.output_pickle_name):
        print(f"Output file existing. Press to confirm overwriting!")
        input()

    list_of_demos = []
    count_dp = 0
    rewards = []
    for demo in sorted(name_of_demos):
        try:
            csv_log = os.path.join(demo, 'log.csv')
            csv_data = pd.read_csv(csv_log, names=['timestamp','robostate','robocmd','exec_cmd' ,'goal', 'cam'])
        except Exception as e:
            print(f"Exception: cannot read log.csv at {demo}")
            continue

        initial_len = len(csv_data)
        valid_data = csv_data[csv_data['robocmd'] != 'None']
        print(f"Found {initial_len} datapoints. " +
            (f"{len(csv_data)} has all images. " if not args.ignore_camera else "") +
            f"{len(valid_data)} usable. ")

        # Extract info
        jointstates = np.array([np.fromstring(_data, dtype=np.float64, sep=' ') for _data in valid_data['robostate']])
        commands = np.array([np.fromstring(_data, dtype=np.float64, sep=' ') for _data in valid_data['robocmd']])
        goals = np.array([np.fromstring(_data, dtype=np.float64, sep=' ') for _data in valid_data['goal']])
        info = {
            'traj_id': os.path.basename(demo[:-1]),  # removing trailing /
            'jointstates': jointstates,
            'commands': commands,
            'goals': goals,
        }

        ######################### FOR MJRL #############################
        # Step 1. goal relabel
        goal = np.copy(info["goals"])
        segment_ids = np.nonzero([np.linalg.norm(goal[i] - goal[i - 1]) for i in range(1, len(goal))])[0]
        cur_infos = []

        if len(segment_ids) == 0:
            cur_infos.append(info)
        else:
            segment_ids = [-1] + list(segment_ids)
            for s in range(1, len(segment_ids)):
                cur_info = {
                    'traj_id': info['traj_id'] + "_" + str(s),
                    'jointstates': info['jointstates'][segment_ids[s - 1]+1:segment_ids[s]+1],
                    'commands': info['commands'][segment_ids[s - 1]+1:segment_ids[s]+1],
                    'goals': info['goals'][segment_ids[s - 1]+1:segment_ids[s]+1],
                }
                cur_infos.append(cur_info)

        for info in cur_infos:
            info['observations'] = np.concatenate([info['jointstates'], info['goals']], axis=1)
            info['actions'] = info['commands']
            termination = np.zeros(len(info['commands']))
            termination[-1] = 1
            info['terminated'] = termination

            from rewards.reaching import reward_function # CHOOSE THE TASK
            tmp_info = {}
            tmp_info["observations"] = np.expand_dims(info["observations"], axis=0)
            tmp_info["actions"] = np.expand_dims(info["actions"], axis=0)
            reward_function(tmp_info)
            info['rewards'] = tmp_info['rewards'][0]
            print(max(info['rewards']))
            rewards.append(max(info['rewards']))
            ######################################################

            list_of_demos.append(info)
            count_dp += len(valid_data)
    print("rewards: ", rewards, np.mean(rewards),  np.std(rewards))
    with open(args.output_pickle_name, 'wb') as f:
        pickle.dump(list_of_demos, f)