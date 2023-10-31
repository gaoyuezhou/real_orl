import os
import pickle
from glob import glob
import pandas as pd
import numpy as np
import argparse
from PIL import Image
from tracking import get_markers_pos
from camera_params import INTRINSICS_DICT, EXTRINSICS_DICT, CAMERA_IDS_DICT
from gym import utils
from gym.envs.mujoco import mujoco_env

current_cameras = ['103422071965', '815412070238']
MARKER_IDS = [4, 5, 10, 11, 2]
MARKER_SIZES = [0.05 for i in range(len(MARKER_IDS))]
extrinsics = [EXTRINSICS_DICT[i] for i in current_cameras]
intrinsics = [INTRINSICS_DICT[i] for i in current_cameras]
# frame between robot and tag
offset = np.array([1.09826201, -0.05144846999, 0.2276264]) 

def coor_transform(tag_frame_pos):
    xt, yt, zt = tag_frame_pos
    x, y, z = -zt, -xt, yt
    franka_frame_pos = np.array([x, y, z]) + offset
    return franka_frame_pos

def get_mid_pos(markers_pos):
    pos_4 = markers_pos[4]
    pos_5 = markers_pos[5]
    pos_10 = markers_pos[10]
    pos_11 = markers_pos[11]
    positions = [pos_4, pos_5, pos_10, pos_11]
    detected = [np.linalg.norm(i) != 0 for i in positions]
    if not any(detected):
        return np.array([0, 0, 0])

    if all(detected):
        mid_pos = np.mean(positions, axis=0)
    elif all(detected[:2]):
        mid_pos = np.mean(positions[:2], axis=0)
    elif all(detected[2:]):
        mid_pos = np.mean(positions[2:], axis=0)
    else:
        center_pos_4 = pos_4 + np.array([-0.09, 0, 0])
        center_pos_5 = pos_5 + np.array([0.09, 0, 0])
        center_pos_10 = pos_10 + np.array([0, 0, 0.09])
        center_pos_11 = pos_11 + np.array([0, 0, -0.09])
        center_positions = [center_pos_4, center_pos_5, center_pos_10, center_pos_11]
        choice = np.where(detected)[0][0]
        mid_pos = center_positions[choice]
    return mid_pos

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

def remove_datapoints_by_idx(info, ids):
    for key in info.keys():
        if type(info[key]) is list:
            for idx in sorted(ids, reverse=True):
                print("cur pop idx: ", idx)
                info[key].pop(idx)
        elif type(info[key]) is np.ndarray:
            info[key] = np.delete(info[key], ids, axis=0)


if __name__ == "__main__":
    args = get_args()
    logs_folder = args.logs_folder + ('/' if args.logs_folder[-1] != '/' else '')

    args.output_pickle_name = logs_folder[:-1].split('/')[-1] + '.pkl'
    print("output name: ", args.output_pickle_name)

    logs_folder += '*/'
    name_of_demos = glob(logs_folder)
    print(f"Found {len(name_of_demos)} recordings in the {logs_folder}")


    if len(name_of_demos) == 0:
        exit()
    if os.path.isfile(args.output_pickle_name):
        print(f"Output file existing. Press to confirm overwriting!")
        input()

    list_of_demos = []
    count_dp = 0
    best_rewards = []
    for demo in sorted(name_of_demos):
        print("demo name: ", demo)
        try:
            csv_log = os.path.join(demo, 'log.csv')
            csv_data = pd.read_csv(csv_log, names=['timestamp','robostate','robocmd', 'exec_cmd', 'goal', 'cam'])
        except Exception as e:
            print(f"Exception: cannot read log.csv at {demo}")
            continue

        initial_len = len(csv_data)

        # Validates Camera Images
        if not args.ignore_camera:
            if type(csv_data['cam'][0]) == float:
                import pdb; pdb.set_trace()
            cam_ts = [_timestamps.split("-") for _timestamps in csv_data['cam']]
            color_cam_fn = {CAMERA_IDS_DICT[i]:[] for i in current_cameras}
            depth_cam_fn = {CAMERA_IDS_DICT[i]:[] for i in current_cameras}
            for dp_idx, datapoint in enumerate(cam_ts):
                my_image_fn = []
                for j, cam_id in enumerate(current_cameras):
                    i = CAMERA_IDS_DICT[cam_id]
                    cimage_fn = f"c{i}-{cam_id}-{datapoint[j*2]}-color.jpeg"
                    color_cam_fn[i].append(cimage_fn
                        if os.path.isfile(os.path.join(demo, cimage_fn))
                        else np.nan
                    )
                    if args.include_depth:
                        dimage_fn = f"c{i}-{cam_id}-{datapoint[j*2+1]}-depth.jpeg"
                        depth_cam_fn[i].append(dimage_fn
                            if os.path.isfile(os.path.join(demo, dimage_fn))
                            else np.nan
                        )
            for cam_id in current_cameras:
                i = CAMERA_IDS_DICT[cam_id]
                csv_data[f"cam{i}c"] = color_cam_fn[i]
                if args.include_depth:
                    csv_data[f"cam{i}d"] = depth_cam_fn[i]

            # Remove datapoints missing any image
            csv_data.dropna(inplace=True)

        valid_data = csv_data[csv_data['robocmd'] != 'None']
        print(f"Found {initial_len} datapoints. " +
            (f"{len(csv_data)} has all images. " if not args.ignore_camera else "") +
            f"{len(valid_data)} usable. ")

        # Extract info
        jointstates = np.array([np.fromstring(_data, dtype=np.float64, sep=' ') for _data in valid_data['robostate']])
        commands = np.array([np.fromstring(_data, dtype=np.float64, sep=' ') for _data in valid_data['exec_cmd']]) ### saving exec_cmd for some datasets!
        goals = np.array([np.fromstring(_data, dtype=np.float64, sep=' ') for _data in valid_data['goal']])
        info = {
            'traj_id': os.path.basename(demo[:-1]),  # removing trailing /
            'jointstates': jointstates,
            'commands': commands,
            'goals': goals,
        }

        if not args.ignore_camera:
            for cam_id in current_cameras:
                i = CAMERA_IDS_DICT[cam_id]
                info[f"cam{i}c"] = list(valid_data[f"cam{i}c"])
                if args.include_depth:
                    info[f"cam{i}d"] = list(valid_data[f"cam{i}d"])
        
        no_marker_idx = []

        mid_marker_positions = []
        board_marker_positions = []
        if not args.ignore_tracking:
            for idx in MARKER_IDS:
                info[f"marker{idx}"] = []

            for i in range(len(info['jointstates'])):
                imgs_ti = []
                for cam_id in current_cameras:
                    _c = CAMERA_IDS_DICT[cam_id]
                    img_fn = "{}/{}".format(info['traj_id'], info[f"cam{_c}c"][i])
                    img = Image.open(os.path.join(args.logs_folder, img_fn)).copy()
                    imgs_ti.append(np.array(img))

                markers_pos_ti = get_markers_pos(imgs_ti, intrinsics=intrinsics, extrinsics=extrinsics, marker_ids=MARKER_IDS, marker_sizes=MARKER_SIZES)
                for idx in MARKER_IDS:
                    info[f"marker{idx}"].append(markers_pos_ti[idx])

                mid_marker_pos = get_mid_pos(markers_pos_ti)
                if np.linalg.norm(mid_marker_pos) == 0:
                    no_marker_idx.append(i)
                mid_marker_pos = coor_transform(mid_marker_pos)
                board_marker_pos = markers_pos_ti[2] # not transformed

                if mid_marker_pos[1] < -0.5:
                    import pdb; pdb.set_trace()

                mid_marker_positions.append(mid_marker_pos)
                board_marker_positions.append(board_marker_pos)
                cur_obs = np.concatenate([info['jointstates'][i], mid_marker_pos, info['goals'][i]])

        mid_marker_positions = np.vstack(mid_marker_positions)

        ############################ FOR MJRL ########################## 
        #### not post processing goals, use the original scripted goals
        print(mid_marker_positions[0], info['goals'][0])
        info['observations'] = np.concatenate([info['jointstates'], mid_marker_positions, info['goals']], axis=1)
        info['actions'] = info['commands']
        termination = np.zeros(len(info['commands']))
        termination[-1] = 1
        info['terminated'] = termination
        print("no marker idx: ", no_marker_idx)
        remove_datapoints_by_idx(info, no_marker_idx)

        from rewards.lifting import reward_function # CHOOSE THE TASK
        tmp_info = {}
        tmp_info["observations"] = np.expand_dims(info["observations"], axis=0)
        tmp_info["actions"] = np.expand_dims(info["actions"], axis=0)
        reward_function(tmp_info)
        info['rewards'] = tmp_info['rewards'][0]
        success = np.linalg.norm(board_marker_positions[-1]) == 0
        success = False
        if success:
            best_rewards.append(1)
        else:
            best_rewards.append(max(info['rewards']))
        import matplotlib.pyplot as plt
        ######################################################
        list_of_demos.append(info)
        count_dp += len(valid_data)
    print(best_rewards, np.mean(best_rewards), np.std(best_rewards))
    with open(args.output_pickle_name, 'wb') as f:
        pickle.dump(list_of_demos, f)