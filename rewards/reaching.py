import numpy as np
from franka_wrapper import FrankaWrapper

# frame between robot and tag
offset = np.array([0.6, 0.07, 0.46])

JOINT_LIMIT_MIN = np.array(
        [-2.8773, -1.7428, -2.8773, -3.0018, -2.8773, 0.0025-np.pi/2, -2.8773, 0.0])
JOINT_LIMIT_MAX = np.array(
        [2.8773, 1.7428, 2.8773, -0.1398, 2.8773, 3.7325-np.pi/2, 2.8773, 0.085])

def coor_transform(tag_frame_pos):
    xt, yt, zt = tag_frame_pos
    x, y, z = -zt, -xt, yt
    franka_frame_pos = np.array([x, y, z]) + offset
    return franka_frame_pos

def reward_function(paths):
    # paths has two keys: observations and actions
    # paths["observations"] : (num_traj, horizon, obs_dim)
    # return paths that contain rewards in path["rewards"]
    # paths["rewards"] should have shape (num_traj, horizon)
    franka = FrankaWrapper()
    num_traj, horizon, obs_dim = paths["observations"].shape
    rewards = np.zeros((num_traj, horizon))
    for i in range(num_traj):
        tags_xyz = paths["observations"][i, :,-3:]
        tags_xyz = tags_xyz[np.linalg.norm(tags_xyz, axis=1) != 0]
        tag_xyz = np.average((tags_xyz), axis=0)
        transformed_tag_xyz = tag_xyz                   ### for current auto-collected reaching data, no need to do coordinate transform

        jointstates = paths["observations"][i, :, :8]
        ee_xyz = np.zeros((horizon, 3))
        for j, js in enumerate(jointstates):
            if (js < JOINT_LIMIT_MAX).all() and (js > JOINT_LIMIT_MIN).all():
                ee_xyz[j] = franka.get_fk(js)
            else:
                ee_xyz[j] = -1

        distance = np.linalg.norm(transformed_tag_xyz - ee_xyz, axis=1)
        assert distance.shape[0] == horizon

        rewards[i] = 1 - distance
    paths["rewards"] = rewards
    return paths

def termination_function(paths):
    for path in paths:
        obs = path["observations"]
        T = obs.shape[0]
        t = 0
        done = False
        while t < T and done is False:
            js = obs[t][:8]
            valid = (js < JOINT_LIMIT_MAX + np.ones(8) * 1).all() and (js > JOINT_LIMIT_MIN - np.ones(8) * 1).all()
            done = not valid
            t = t + 1
            T = t if done else T
        path["observations"] = path["observations"][:T]
        path["actions"] = path["actions"][:T]
        path["rewards"] = path["rewards"][:T]
        path["terminated"] = done
    return paths

