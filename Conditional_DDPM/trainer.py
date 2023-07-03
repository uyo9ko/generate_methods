from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True,
    self_condition = True
)
print( f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    # '/home/zhengxb/shenzh_work/data/area_4w_images',
    folder='/home/zhengxb/shenzh/work0627/seg_infer_out/cropThyroid/',
    mask_folder='/home/zhengxb/shenzh/work0627/seg_infer_out/cropMask/',
    results_folder='cropThyroid',
    train_batch_size = 48,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True,              # whether to calculate fid during training
    # save_and_sample_every=5
)
trainer.load(95)
# trainer.train()
# 
for i in range(50):
    trainer.sample('/home/zhengxb/shenzh/work0627/seg_infer_out/cropMask/',8,i)
