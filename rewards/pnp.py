import numpy as np
from franka_wrapper import FrankaWrapper
from rewards.lifting import z_plane_adjust_batch

# Frame between robot and tag
offset = np.array([1.09826201, -0.05144846999, 0.2276264])

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
    ee_xyzs = np.zeros((num_traj, horizon, 3))
    gripper_dists = np.zeros((num_traj, horizon))
    goal_dists = np.zeros((num_traj, horizon))
    for i in range(num_traj):
        tag_pos = paths["observations"][i, :,-6:-3] #both pos are transformed already
        tag_pos = z_plane_adjust_batch(tag_pos)
        goal_pos = paths["observations"][i, :,-3:]
        jointstates = paths["observations"][i, :, :8]
        ee_xyz = np.zeros((horizon, 3))
        for j, js in enumerate(jointstates):
            if (js < JOINT_LIMIT_MAX).all() and (js > JOINT_LIMIT_MIN).all():
                ee_xyz[j] = franka.get_fk(js)
                ee_xyz[j][2] -= 0.17
            else:
                ee_xyz[j] = -1
        
        gripper_dist = np.linalg.norm(tag_pos - ee_xyz, axis=1)
        goal_dist = np.linalg.norm(tag_pos[:, :2] - goal_pos[:, :2], axis=1)
        height_bonus = np.minimum(0.5, np.maximum(tag_pos[:, 2], 0) * 10) / 2
        height_bonus = (gripper_dist < 0.1) * height_bonus 
        rewards[i] = (1 - (gripper_dist * 2 + goal_dist)) * 0.9 + height_bonus 

        ee_xyzs[i] = ee_xyz
        gripper_dists[i] = gripper_dist
        goal_dists[i] = goal_dist
    paths['ee_xyz']  = ee_xyzs
    paths['gripper_dist']  = gripper_dists
    paths['goal_dist']  = goal_dists
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

