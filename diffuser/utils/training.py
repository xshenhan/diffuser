import os
import copy
import numpy as np
import torch
import einops
import pdb
import robomimic.utils.tensor_utils as TensorUtils
import wandb
import imageio.v3 as imageio

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs
from .libero_eval import eval_one_task_success, get_diffusion_conditions, plan, navie_action_mse

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        n_saves=5,
        # label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        libero=False,
        eval_cfg=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        # self.label_freq = label_freq
        self.n_saves = n_saves
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.libero = libero
        self.eval_cfg = eval_cfg

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                if isinstance(batch, dict):
                    batch = TensorUtils.to_device(batch, device=self.model.device)
                    loss, infos = self.model.loss(batch)
                else:
                    batch = batch_to_device(batch)
                    loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
            wandb_to_log = {
                "loss": loss.item(),
            }
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.8f}' for key, val in infos.items()])
                if self.libero:
                    eval_info = {
                        "navie_action_mse": navie_action_mse(self.model, batch),
                        "navie_action_mse_ema": navie_action_mse(self.ema_model, batch),
                    }
                    infos_str += f' | {eval_info}'
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}', flush=True)
                wandb_to_log.update(infos)
                if self.libero:
                    wandb_to_log.update(eval_info)

            if self.step == 0 and self.sample_freq and not self.libero:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                if self.libero:
                    success_rate, info = eval_one_task_success(self.ema_model, self.dataset.benchmark.get_task(0), self.eval_cfg)
                    wandb_to_log["ema_success_rate"] = success_rate
                    images = info["images"]
                    os.makedirs("./ema_eval_videos", exist_ok=True)
                    for i, images_i in enumerate(images):
                        for k,v in images_i.items():
                            imageio.imwrite(f"./ema_eval_videos/{self.step}_step_{k}_video_{i}.mp4", v, fps=30)
                    for k in images[0].keys():
                        wandb_to_log[f"ema_eval_videos/{self.step}_step_{k}_video_0"] = wandb.Video(f"./ema_eval_videos/{self.step}_step_{k}_video_0.mp4")

                    success_rate, info = eval_one_task_success(self.model, self.dataset.benchmark.get_task(0), self.eval_cfg)
                    wandb_to_log["success_rate"] = success_rate
                    images = info["images"]
                    os.makedirs("./eval_videos", exist_ok=True)
                    for i, images_i in enumerate(images):
                        for k,v in images_i.items():
                            imageio.imwrite(f"./eval_videos/{self.step}_step_{k}_video_{i}.mp4", v, fps=30)
                    for k in images[0].keys():
                        wandb_to_log[f"eval_videos/{self.step}_step_{k}_video_0"] = wandb.Video(f"./eval_videos/{self.step}_step_{k}_video_0.mp4")
                else:
                    self.render_samples()

            wandb.log(wandb_to_log)
            self.step += 1

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)
        self.cleanup_old_checkpoints()  

    def cleanup_old_checkpoints(self):
        '''
        只保留最新的self.n_saves个检查点文件
        '''
        checkpoints = []
        for f in os.listdir(self.logdir):
            if f.startswith('state_') and f.endswith('.pt'):
                try:
                    epoch = int(f.split('_')[1].split('.')[0])
                    checkpoints.append((epoch, f))
                except:
                    continue
                    
        # 按epoch排序
        checkpoints.sort(reverse=True)
        if len(checkpoints) > self.n_saves:
            # 删除旧的检查点
            for epoch, f in checkpoints[self.n_saves:]:
                path = os.path.join(self.logdir, f)
                try:
                    os.remove(path)
                    print(f'[ utils/training ] Removed old checkpoint: {path}')
                except:
                    print(f'[ utils/training ] Failed to remove checkpoint: {path}')

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda:0')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model(conditions)
            trajectories = to_np(samples.trajectories)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = trajectories[:, :, self.dataset.action_dim:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)
