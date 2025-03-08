import diffuser.utils as utils
import hydra
import robomimic.utils.tensor_utils as TensorUtils
import wandb
import argparse
import pickle 
from pathlib import Path
import torch
import robomimic.utils.tensor_utils as TensorUtils
import einops
from diffuser.utils.libero_eval import get_diffusion_conditions, plan, navie_action_mse, eval_one_task_success
import imageio
import os
import pickle
import numpy as np
import random
from loguru import logger
import imageio.v3 as imageio

DEBUG_DIR = Path("/home/tiger/myplan/tmp")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

@hydra.main(version_base=None, config_path="../hydra-config", config_name="base")
def main(cfg):
    args = cfg.params
    output_dir = args.output_dir
    path = Path(output_dir)
    file_name = args.pt_name if hasattr(args, "pt_name") else None
    seed = args.seed if hasattr(args, "seed") else 0
    set_seed(seed)
    logger.info(f"Evaluating on seed: {seed}")

    with open(path / "dataset_config.pkl", "rb") as f:
        dataset_config = pickle.load(f)
    with open(path / "model_config.pkl", "rb") as f:
        model_config = pickle.load(f)
    with open(path / "diffusion_config.pkl", "rb") as f:
        diffusion_config = pickle.load(f)

    if file_name is None:
        state_paths = list(path.glob("state_*.pt"))
        state_paths.sort(key=lambda x: int(x.stem.split("_")[-1]))
        newest_state_path = state_paths[-1]
    else:
        newest_state_path = path / file_name

    dataset = dataset_config()
    model = model_config()
    diffusion = diffusion_config(model)
    data = torch.load(newest_state_path)
    diffusion.load_state_dict(data['model'])

    data = dataset[0]
    data = TensorUtils.to_batch(data)
    data = TensorUtils.to_torch(data, device="cuda")
    actions_navie_mse = navie_action_mse(diffusion, data)
    task = dataset.benchmark.get_task(0)
    success_rate, info = eval_one_task_success(diffusion, task, args.eval_config)
    images = info["images"]
    os.makedirs("./eval_videos", exist_ok=True)
    for i, images_i in enumerate(images):
        for k,v in images_i.items():
            imageio.imwrite(f"./eval_videos/{k}_video_{i}.mp4", np.uint8(255*v), fps=30)
    pickle.dump(info, open(f"./eval_info.pkl", "wb"))
    logger.info(f"actions_navie_mse: {actions_navie_mse}")
    logger.info(f"success_rate: {success_rate}")

if __name__ == "__main__":
    main()
