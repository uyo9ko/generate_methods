from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True,
    channels = 3,
    self_condition=False
)

diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    folder='/home/zhengxb/shenzh_work/data/cancer_fudan_data/neg',
    results_folder = '/home/zhengxb/shenzh_work/AIGC_results/cancer_fudan_neg_exp',
    convert_image_to = 'L',
    train_batch_size = 48,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False              # whether to calculate fid during training
)

trainer.train()