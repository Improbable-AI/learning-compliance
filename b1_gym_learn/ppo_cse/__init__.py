import time
from collections import deque
import copy
import os

import torch
# from ml_logger import logger
import wandb

from params_proto import PrefixProto

from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class DataCaches:
    def __init__(self, curriculum_bins):
        from b1_gym_learn.ppo_cse.metrics_caches import SlotCache, DistCache

        self.slot_cache = SlotCache(curriculum_bins)
        self.dist_cache = DistCache()


caches = DataCaches(1)


class RunnerArgs(PrefixProto, cli=False):
    # runner
    algorithm_class_name = 'PPO'
    num_steps_per_env = 24  # per iteration
    max_iterations = 1500  # number of policy updates

    # logging
    save_interval = 400  # check for potential saves every this many iterations
    save_video_interval = 100
    log_freq = 10

    # load and resume
    resume = False
    resume_supercloud = False
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = None  # updated from load_run and chkpt
    resume_curriculum = True
    resume_checkpoint = 'ac_weights_last.pt'



class Runner:

    def __init__(self, env, device='cpu'):
        from .ppo import PPO

        self.device = device
        self.env = env

        actor_critic = ActorCritic(self.env.num_obs,
                                      self.env.num_privileged_obs,
                                      self.env.num_obs_history,
                                      self.env.num_actions,
                                      ).to(self.device)

        from b1_gym import MINI_GYM_ROOT_DIR
        # Load weights from checkpoint 
        if RunnerArgs.resume:
            body = wandb.restore(RunnerArgs.resume_checkpoint, run_path=RunnerArgs.resume_path)            
            actor_critic.load_state_dict(torch.load(body.name))
            print(f"Successfully loaded weights from checkpoint ({RunnerArgs.resume_checkpoint}) and run path ({RunnerArgs.resume_path})")
        elif RunnerArgs.resume_supercloud:
            print(f"Loading weights from checkpoint ({RunnerArgs.resume_checkpoint}) and run path ({RunnerArgs.resume_path} and {MINI_GYM_ROOT_DIR})")
            path = MINI_GYM_ROOT_DIR+ "/resume_runs/" + RunnerArgs.resume_path + "/" + RunnerArgs.resume_checkpoint
            print("path: ", path)
            actor_critic.load_state_dict(torch.load(path))

        self.alg = PPO(actor_critic, device=self.device)
        self.num_steps_per_env = RunnerArgs.num_steps_per_env

        # init storage and model
        self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_obs_history], [self.env.num_actions])

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_recording_it = -RunnerArgs.save_video_interval

        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False, eval_freq=100, curriculum_dump_freq=500, eval_expert=False):
        # from ml_logger import logger
        # initialize writer
        # assert logger.prefix, "you will overwrite the entire instrument server"

        # logger.start('start', 'epoch', 'episode', 'run', 'step')

        wandb.watch(self.alg.actor_critic, log=None, log_freq=RunnerArgs.log_freq)

        if init_at_random_ep_len:
            self.env.episode_length_buf[:] = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # split train and test envs
        num_train_envs = self.env.num_train_envs

        obs_dict = self.env.get_observations()  # TODO: check, is this correct on the first step?
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(
            self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs[:num_train_envs], privileged_obs[:num_train_envs],
                                                 obs_history[:num_train_envs])
                    
                    with torch.no_grad():
                        latent = self.alg.actor_critic.adaptation_module(obs_history[:num_train_envs])
                    ret = self.env.step(actions)
                    obs_dict, rewards, dones, infos = ret
                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict[
                        "obs_history"]

                    obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(
                        self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards[:num_train_envs], dones[:num_train_envs], infos)

                    if 'train/episode' in infos:
                        # with logger.Prefix(metrics="train/episode"):
                        #     logger.store_metrics(**infos['train/episode'])
                        wandb.log(infos['train/episode'], step=it)

                    if 'curriculum' in infos:

                        cur_reward_sum += rewards
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        new_ids_train = new_ids[new_ids < num_train_envs]
                        rewbuffer.extend(cur_reward_sum[new_ids_train].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids_train].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_train] = 0
                        cur_episode_length[new_ids_train] = 0

                    if 'curriculum/distribution' in infos:
                        distribution = infos['curriculum/distribution']

                # stop = time.time()
                # collection_time = stop - start

                # Learning step
                # start = stop
                self.alg.compute_returns(obs_history[:num_train_envs], privileged_obs[:num_train_envs])

                # if it % curriculum_dump_freq == 0:
                #     logger.save_pkl({"iteration": it,
                #                      **caches.slot_cache.get_summary(),
                #                      **caches.dist_cache.get_summary()},
                #                     path=f"curriculum/info.pkl", append=True)

                #     if 'curriculum/distribution' in infos:
                #         logger.save_pkl({"iteration": it,
                #                          "distribution": distribution},
                #                          path=f"curriculum/distribution.pkl", append=True)

            mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student, mean_adaptation_losses_dict = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            # logger.store_metrics(
            #     # total_time=learn_time - collection_time,
            #     time_elapsed=logger.since('start'),
            #     time_iter=logger.split('epoch'),
            #     adaptation_loss=mean_adaptation_module_loss,
            #     mean_value_loss=mean_value_loss,
            #     mean_surrogate_loss=mean_surrogate_loss,
            #     mean_decoder_loss=mean_decoder_loss,
            #     mean_decoder_loss_student=mean_decoder_loss_student,
            #     mean_decoder_test_loss=mean_decoder_test_loss,
            #     mean_decoder_test_loss_student=mean_decoder_test_loss_student,
            #     mean_adaptation_module_test_loss=mean_adaptation_module_test_loss
            # )
            wandb.log({
                "time_iter": learn_time,
                # "time_iter": logger.split('epoch'),
                "adaptation_loss": mean_adaptation_module_loss,
                "mean_value_loss": mean_value_loss,
                "mean_surrogate_loss": mean_surrogate_loss,
                "mean_decoder_loss": mean_decoder_loss,
                "mean_decoder_loss_student": mean_decoder_loss_student,
                "mean_decoder_test_loss": mean_decoder_test_loss,
                "mean_decoder_test_loss_student": mean_decoder_test_loss_student,
                "mean_adaptation_module_test_loss": mean_adaptation_module_test_loss
            }, step=it)


            
            # logger.store_metrics(**mean_adaptation_losses_dict)
            wandb.log(mean_adaptation_losses_dict, step=it)

            if RunnerArgs.save_video_interval:
                self.log_video(it)

            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            # if logger.every(RunnerArgs.log_freq, "iteration", start_on=1):
            # if it % Config.log_freq == 0:
            # logger.log_metrics_summary(key_values={"timesteps": self.tot_timesteps, "iterations": it})
            # logger.job_running()
            wandb.log({"timesteps": self.tot_timesteps, "iterations": it}, step=it)

            if it % RunnerArgs.save_interval == 0:
                    print(f"Saving model at iteration {it}")
                # with logger.Sync():
                #     logger.torch_save(self.alg.actor_critic.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
                #     logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")
                    
                    path = './tmp/legged_data'
                    os.makedirs(path, exist_ok=True)

                    ac_weight_path = f'{path}/ac_weights_{it}.pt'
                    torch.save(self.alg.actor_critic.state_dict(), ac_weight_path)
                    wandb.save(ac_weight_path)

                    ac_weight_path = f'{path}/ac_weights_latest.pt'
                    torch.save(self.alg.actor_critic.state_dict(), ac_weight_path)
                    wandb.save(ac_weight_path)

                    adaptation_module_path = f'{path}/adaptation_module_{it}.jit'
                    adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
                    traced_script_adaptation_module = torch.jit.script(adaptation_module)
                    traced_script_adaptation_module.save(adaptation_module_path)

                    adaptation_module_path = f'{path}/adaptation_module_latest.jit'
                    traced_script_adaptation_module.save(adaptation_module_path)

                    body_path = f'{path}/body_{it}.jit'
                    body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
                    traced_script_body_module = torch.jit.script(body_model)
                    traced_script_body_module.save(body_path)

                    body_path = f'{path}/body_latest.jit'
                    traced_script_body_module.save(body_path)

                    # logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)
                    # logger.upload_file(file_path=body_path, target_path=f"checkpoints/", once=False)

                    ac_weights_path = f"{path}/ac_weights_{it}.pt"
                    torch.save(self.alg.actor_critic.state_dict(), ac_weights_path)
                    ac_weights_path = f"{path}/ac_weights_latest.pt"
                    torch.save(self.alg.actor_critic.state_dict(), ac_weights_path)
                    
                    wandb.save(f"./tmp/legged_data/adaptation_module_{it}.jit")
                    wandb.save(f"./tmp/legged_data/body_{it}.jit")
                    wandb.save(f"./tmp/legged_data/ac_weights_{it}.pt")
                    wandb.save(f"./tmp/legged_data/adaptation_module_latest.jit")
                    wandb.save(f"./tmp/legged_data/body_latest.jit")
                    wandb.save(f"./tmp/legged_data/ac_weights_latest.pt")
                    
            self.current_learning_iteration += num_learning_iterations

        # torch.save(self.alg.actor_critic.state_dict(), f"./tmp/legged_data/ac_weights_last.pt")
        # wandb.save(f"./tmp/legged_data/ac_weights_last.pt")

        path = './tmp/legged_data'

        os.makedirs(path, exist_ok=True)

        adaptation_module_path = f'{path}/adaptation_module_latest.jit'
        adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
        traced_script_adaptation_module = torch.jit.script(adaptation_module)
        traced_script_adaptation_module.save(adaptation_module_path)

        body_path = f'{path}/body_latest.jit'
        body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
        traced_script_body_module = torch.jit.script(body_model)
        traced_script_body_module.save(body_path)

        # logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)
        # logger.upload_file(file_path=body_path, target_path=f"checkpoints/", once=False)
        wandb.save(f"./tmp/legged_data/adaptation_module_latest.jit")
        wandb.save(f"./tmp/legged_data/body_latest.jit")

    def log_video(self, it):
        if it - self.last_recording_it >= RunnerArgs.save_video_interval:
            self.env.start_recording()
            print("START RECORDING")
            self.last_recording_it = it

        frames = self.env.get_complete_frames()
        if len(frames) > 0:
            self.env.pause_recording()
            print("LOGGING VIDEO")
            import numpy as np
            video_array = np.concatenate([np.expand_dims(frame, axis=0) for frame in frames ], axis=0).swapaxes(1, 3).swapaxes(2, 3)
            print(video_array.shape)
            # logger.save_video(frames, f"videos/{it:05d}.mp4", fps=1 / self.env.dt)
            wandb.log({"video": wandb.Video(video_array, fps=1 / self.env.dt)}, step=it)

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_expert
