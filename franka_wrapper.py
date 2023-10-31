import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class FrankaWrapper(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, "./franka_panda.xml", frame_skip=5)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        reward = {'r':0}
        done = False
        return ob, reward, done, reward

    def get_fk(self, jointpos):
        qpos = self.init_qpos
        qpos[:7] = jointpos[:7]
        qvel = self.init_qvel
        qvel[:] = 0
        self.set_state(qpos, qvel)
        return self.sim.data.get_site_xpos("end_effector")

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[:7],
                self.sim.data.qvel.flat[:7],
                self.sim.data.get_site_xpos("end_effector"),
            ]
        )

    def get_site_com(self, site_name):
        return

if __name__ == "__main__":
    a = FrankaWrapper()
    print(a.get_fk([0,0,0,0,0,0,0]))
    print(a.get_fk([-0.145, -0.67, -0.052, -2.3, 0.145, 1.13, 0.029]))
    print(a.get_fk([0, -0.67, -0.052, -2.3, 0.145, 1.13, 0.029]))