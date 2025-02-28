import diffuser.utils as utils
import hydra
import robomimic.utils.tensor_utils as TensorUtils
import wandb

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

def get_observation_dim(shape_meta, hidden_dim):
    observation_dim = 0
    for k, v in shape_meta["all_shapes"].items():
        if "rgb" in k:
            observation_dim += hidden_dim
        else:
            observation_dim += v[0]
    return observation_dim

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#
@hydra.main(version_base=None, config_path="../hydra-config", config_name="base")
def main(config):
    args = config.params
    if hasattr(args, 'dataset_type') and args.dataset_type == 'libero':
        dataset_config = utils.Config(
            "datasets.LiberoDataset",
            savepath=(args.savepath, 'dataset_config.pkl'),
            benchmark_name=args.dataset,
            task_order_index=args.task_order_index,
            horizon=args.horizon,
            n_tasks=args.n_tasks,
            dataset_folder=args.dataset_folder,
            obs_modality=args.obs_modality,
            image_encoder=args.image_encoder,
            freeze_image_encoder=args.freeze_image_encoder,
            normalizer=args.normalizer,
            preprocess_fns=args.preprocess_fns,
            use_padding=args.use_padding,
            max_path_length=args.max_path_length,
            task_embedding_format=args.task_embedding_format,
            task_embedding_one_hot_offset=args.task_embedding_one_hot_offset,
            data_max_word_len=args.data_max_word_len,
            max_n_episodes=args.max_n_episodes,
            termination_penalty=args.termination_penalty,
            seed=args.seed,
        )
        dataset = dataset_config()
        shape_meta = dataset.shape_meta
        observation_dim = get_observation_dim(shape_meta, args.image_encoder_config.hidden_dim)
        action_dim = dataset.action_dim
        dataset.observation_dim = observation_dim
    else:
        dataset_config = utils.Config(
            args.loader,
            savepath=(args.savepath, 'dataset_config.pkl'),
            env=args.dataset,
            horizon=args.horizon,
            normalizer=args.normalizer,
            preprocess_fns=args.preprocess_fns,
            use_padding=args.use_padding,
            max_path_length=args.max_path_length,        
        )
        dataset = dataset_config()
    render_config = utils.Config(
        args.renderer,
        savepath=(args.savepath, 'render_config.pkl'),
        env=args.dataset,
    )
    renderer = render_config()
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim




    #-----------------------------------------------------------------------------#
    #------------------------------ model & trainer ------------------------------#
    #-----------------------------------------------------------------------------#

    if hasattr(args, 'dataset_type') and args.dataset_type == 'libero':
        model_config = utils.Config(
            args.model,
            savepath=(args.savepath, 'model_config.pkl'),
            horizon=args.horizon,
            transition_dim=observation_dim + action_dim,
            image_encoder_config=args.image_encoder_config,
            shape_meta=shape_meta,
            cond_dim=observation_dim,
            dim_mults=args.dim_mults,
            attention=args.attention,
            device=args.device,
        )        
    else:
        model_config = utils.Config(
            args.model,
            savepath=(args.savepath, 'model_config.pkl'),
            horizon=args.horizon,
            transition_dim=observation_dim + action_dim,
            cond_dim=observation_dim,
            dim_mults=args.dim_mults,
            attention=args.attention,
            device=args.device,
        )

    model = model_config()

    diffusion_config = utils.Config(
        args.diffusion,
        savepath=(args.savepath, 'diffusion_config.pkl'),
        horizon=args.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=args.n_diffusion_steps,
        loss_type=args.loss_type,
        clip_denoised=args.clip_denoised,
        predict_epsilon=args.predict_epsilon,
        image_encoder_loss_weight=args.image_encoder_loss_weight,
        ## loss weighting
        action_weight=args.action_weight,
        loss_weights=args.loss_weights,
        loss_discount=args.loss_discount,
        device=args.device,
    )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath=(args.savepath, 'trainer_config.pkl'),
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        sample_freq=args.sample_freq,
        save_freq=args.save_freq,
        label_freq=int(args.n_train_steps // args.n_saves),
        save_parallel=args.save_parallel,
        results_folder=args.savepath,
        bucket=args.bucket,
        n_reference=args.n_reference,
        libero=args.dataset_type == 'libero',
    )

    #-----------------------------------------------------------------------------#
    #-------------------------------- instantiate --------------------------------#
    #-----------------------------------------------------------------------------#

    diffusion = diffusion_config(model)

    trainer = trainer_config(diffusion, dataset, renderer)


    #-----------------------------------------------------------------------------#
    #------------------------ test forward & backward pass -----------------------#
    #-----------------------------------------------------------------------------#

    utils.report_parameters(model)

    print('Testing forward...', end=' ', flush=True)
    if hasattr(args, 'dataset_type') and args.dataset_type == 'libero':
        data = dataset[0]
        data = TensorUtils.to_torch(data, device=args.device)
        data = TensorUtils.map_tensor(data, lambda x: x.unsqueeze_(0))
        loss, _ = diffusion.loss(data)
        loss.backward()
    else:
        batch = utils.batchify(dataset[0])
        loss, _ = diffusion.loss(*batch)
        loss.backward()
    print('✓')


    #-----------------------------------------------------------------------------#
    #--------------------------------- main loop ---------------------------------#
    #-----------------------------------------------------------------------------#

    n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

    wandb.init(
        project=args.wandb.project,
        entity=args.wandb.entity,
        name=args.wandb.name,
        config=args,
    )

    for i in range(n_epochs):
        print(f'Epoch {i} / {n_epochs} | {args.savepath}')
        trainer.train(n_train_steps=args.n_steps_per_epoch)

if __name__ == "__main__":
    main()