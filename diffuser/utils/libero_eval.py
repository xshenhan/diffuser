import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import torch
import einops
import os
import time
from myplan.utils.envs import DummyVectorEnv, SubprocVectorEnv, OffScreenRenderEnv
from myplan.utils.configclass import EvalConfigDiffuser
from typing import Tuple, Dict
import tqdm

import numpy as np
def get_diffusion_conditions(diffusion, data):
    diffusion_obs = diffusion.get_diffusion_obs(data)
    conditions = diffusion.get_conditions(diffusion_obs)
    return conditions

def plan(diffusion, conditions, task_embedding, device="cuda", n_samples=1):
    assert n_samples == 1
    conditions = TensorUtils.to_torch(conditions, device)
    if isinstance(conditions, dict):
        conditions = TensorUtils.map_tensor(
            conditions,
            lambda x: einops.repeat(x, 'b d -> (repeat b) d', repeat=n_samples),
        )
    else:
        conditions = TensorUtils.map_tensor(
            conditions,
            lambda x: einops.repeat(x, 'b t d -> (repeat b) t d', repeat=n_samples),
        )
    samples = diffusion(conditions, task_embedding=task_embedding)
    trajectories = samples.trajectories
    # if not isinstance(trajectories, dict):
    #     trajectories = einops.rearrange(trajectories, '(repeat b) t h -> repeat b t h', repeat=n_samples)
    return trajectories

def get_actions(diffusion, data, action_dim=None, n_samples=1):
    if action_dim is None:
        action_dim = data["actions"].shape[-1]
    conditions = get_diffusion_conditions(diffusion, data)
    task_embedding = data["task_emb"]
    trajectories = plan(diffusion, conditions, task_embedding, device="cuda", n_samples=n_samples)
    actions = trajectories[:, :, -action_dim:]
    return actions

def navie_action_mse(diffusion, data, n_samples=1):
    conditions = get_diffusion_conditions(diffusion, data)
    task_embedding = data["task_emb"]   
    trajectories = plan(diffusion, conditions, task_embedding, device="cuda", n_samples=n_samples)
    actions_gt = data["actions"]
    actions_pred = trajectories[:, :, -data["actions"].shape[-1]:]
    actions_navie_mse = torch.nn.functional.mse_loss(actions_pred, actions_gt)
    return actions_navie_mse

def raw_obs_to_tensor_obs(obs, cfg, obs_modality, obs_key_mapping, device="cuda"):
    """
    Prepare the tensor observations as input for the algorithm.
    """
    env_num = len(obs)

    data = {
        "obs": {},
    }

    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        for obs_name in modality_list:
            data["obs"][obs_name] = []
        all_obs_keys += modality_list

    for k in range(env_num):
        for obs_name in all_obs_keys:
            data["obs"][obs_name].append(
            ObsUtils.process_obs(
                    torch.from_numpy(obs[k][obs_key_mapping[obs_name]]),
                    obs_key=obs_name,
                ).float()
            )

    for key in data["obs"]:
        data["obs"][key] = torch.stack(data["obs"][key])

    data = TensorUtils.to_device(data, device=device)
    return data

def eval_one_task_success(diffusion, task, eval_cfg: EvalConfigDiffuser, task_str="", device="cuda", task_emb=None) -> Tuple[float, Dict]:
    num_eval = eval_cfg.n_eval
    query_freq = eval_cfg.query_freq
    if not hasattr(eval_cfg, "num_procs"):
        eval_cfg.num_procs = 1
    env_num = min(eval_cfg.num_procs, eval_cfg.n_eval)
    eval_loop_num = (eval_cfg.n_eval + env_num - 1) // env_num

    # initiate evaluation envs
    env_args = {
        "bddl_file_name": os.path.join(
            eval_cfg.bddl_root_folder, task.problem_folder, task.bddl_file
        ),
        "camera_heights": eval_cfg.img_h,
        "camera_widths": eval_cfg.img_w,
    }
    
    # Try to handle the frame buffer issue
    env_creation = False

    count = 0
    while not env_creation and count < 5:
        try:
            if env_num == 1:
                env = DummyVectorEnv(
                    [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                )
            else:
                env = SubprocVectorEnv(
                    [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                )
            env_creation = True
        except Exception as e:
            print(e)
            time.sleep(5)
            count += 1
    if count >= 5:
        raise Exception("Failed to create environment")
    
    # get fixed init states to control the experiment randomness
    init_states_path = os.path.join(
        eval_cfg.init_files_folder, task.problem_folder, task.init_states_file
    )
    init_states = torch.load(init_states_path, weights_only=False)
    num_success = 0
    images = [{} for _ in range(num_eval)]
    sim_states = [[] for _ in range(num_eval)]
    for i in tqdm.tqdm(range(eval_loop_num), desc="Evaluating success rate"):
        env.reset()
        indices = np.arange(i * env_num, (i + 1) * env_num) % init_states.shape[0]
        init_states_ = init_states[indices]
        dones = [False] * env_num
        steps = 0
        obs = env.set_init_state(init_states_)
        # dummy actions [env_num, 7] all zeros for initial physics simulation
        dummy = np.zeros((env_num, 7))
        for _ in range(5):
            obs, _, _, _ = env.step(dummy)
        query_steps = 0
        while steps < eval_cfg.max_steps:
            steps += 1

            data = raw_obs_to_tensor_obs(obs, eval_cfg, eval_cfg.modality, eval_cfg.obs_key_mapping, device)
            data["task_emb"] = task_emb.repeat(env_num, 1)
            data = TensorUtils.map_tensor(data, lambda x: x.unsqueeze_(1))
            
            # append_images
            for k,v in data["obs"].items():
                if 'rgb' in k:
                    if k not in images[i]:
                        images[i][k] = []
                    image = v[0, 0, ...]
                    images[i][k].append(einops.rearrange(image, 'c h w -> h w c'))

            with torch.inference_mode():
                if query_steps % query_freq == 0:
                    actions_traj = get_actions(diffusion, data, action_dim=dummy.shape[-1], n_samples=1)
                actions = TensorUtils.to_numpy(actions_traj[:, query_steps % query_freq, :])
                query_steps += 1

            obs, reward, done, info = env.step(actions)

            # record the sim states for replay purpose
            sim_state = env.get_sim_state()
            for k in range(env_num):
                if i * env_num + k < eval_cfg.n_eval and sim_states is not None:
                    sim_states[i * env_num + k].append(sim_state[k])

            # check whether succeed
            for k in range(env_num):
                dones[k] = dones[k] or done[k]

            if all(dones):
                break

        # a new form of success record
        for k in range(env_num):
            if i * env_num + k < eval_cfg.n_eval:
                num_success += int(dones[k])
                
    success_rate = num_success / eval_cfg.n_eval
    env.close()
    images = TensorUtils.to_numpy(images)
    images = [{k: np.array(v) for k, v in images_i.items()} for images_i in images]
    info = {
        "sim_states": sim_states,
        "success_rate": success_rate,
        "images": images,
    }
    return success_rate, info