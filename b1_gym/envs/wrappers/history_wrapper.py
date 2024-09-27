import isaacgym
assert isaacgym
import torch
import gym

class HistoryWrapper(gym.Wrapper):
    def __init__(self, env, reward_scaling=1.0):
        super().__init__(env)
        self.env = env
        self.reward_scaling = reward_scaling

        self.obs_history_length = self.env.cfg.env.num_observation_history
        self.history_frame_skip = self.env.cfg.env.history_frame_skip

        self.num_obs_history = self.obs_history_length * self.num_obs
        self.obs_history_buf = torch.zeros(self.env.num_envs, self.obs_history_length * self.history_frame_skip, self.num_obs, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        self.obs_history = torch.zeros(self.env.num_envs, self.num_obs_history, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        self.num_privileged_obs = self.num_privileged_obs

        self.reward_container.load_env(self)
        
    def step(self, action):
        # privileged information and observation history are stored in info
        obs, rew, done, info = self.env.step(action)
        privileged_obs = info["privileged_obs"]

        self.obs_history_buf = torch.cat((self.obs_history_buf[:, 1:, :], obs.unsqueeze(1)), dim=1)
        self.obs_history = self.obs_history_buf[:, self.history_frame_skip-1::self.history_frame_skip, :].reshape(self.env.num_envs, -1)
        assert self.obs_history[:, -self.num_obs:].allclose(obs[:, :]), "obs_history does not end with obs"
        
        env_ids = self.env.reset_buf.nonzero(as_tuple=False).flatten()
        self.obs_history_buf[env_ids, :, :] = 0
        self.obs_history[env_ids, :] = 0
        
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}, rew * self.reward_scaling, done, info

    def get_observations(self):
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history_buf = torch.cat((self.obs_history_buf[:, 1:, :], obs.unsqueeze(1)), dim=1)
        self.obs_history = self.obs_history_buf[:, self.history_frame_skip-1::self.history_frame_skip, :].reshape(self.env.num_envs, -1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}

    def reset_idx(self, env_ids):  # it might be a problem that this isn't getting called!!
        ret = super().reset_idx(env_ids)
        self.obs_history_buf[env_ids, :, :] = 0
        self.obs_history[env_ids, :] = 0
        return ret

    def reset(self):
        ret = super().reset()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history[:, :] = 0
        return {"obs": ret, "privileged_obs": privileged_obs, "obs_history": self.obs_history}


if __name__ == "__main__":
    from tqdm import trange
    import matplotlib.pyplot as plt

    import ml_logger as logger

    from b1_gym_learn.ppo import Runner
    from b1_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from b1_gym_learn.ppo.actor_critic import AC_Args

    from b1_gym.envs.base.legged_robot_config import Cfg
    from b1_gym.envs.mini_cheetah.mini_cheetah_config import config_mini_cheetah
    config_mini_cheetah(Cfg)

    test_env = gym.make("VelocityTrackingEasyEnv-v0", cfg=Cfg)
    env = HistoryWrapper(test_env)

    env.reset()
    action = torch.zeros(test_env.num_envs, 12)
    for i in trange(3):
        obs, rew, done, info = env.step(action)
        print(obs.keys())
        print(f"obs: {obs['obs']}")
        print(f"privileged obs: {obs['privileged_obs']}")
        print(f"obs_history: {obs['obs_history']}")

        img = env.render('rgb_array')
        plt.imshow(img)
        plt.show()
