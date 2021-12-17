_base_ = [
    '../_base_/models/dcgan/dcgan_28x28.py',
    '../_base_/datasets/encrypted_traffic_data.py',
    '../_base_/default_runtime.py'
]

# define dataset
# you must set `samples_per_gpu` and `imgs_root`
data = dict(
    samples_per_gpu=128,
    train=dict(imgs_root='/root/autodl-tmp/encrypted_traffic_data/da_data/train'))

# adjust running config
lr_config = None
checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=20)
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=10000)
]

total_iters = 300002

# use ddp wrapper for faster training
use_ddp_wrapper = True
find_unused_parameters = False

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,  # Note that this flag should be False.
    pass_training_status=True)

metrics = dict(
    ms_ssim10k=dict(type='MS_SSIM', num_images=10000),
    swd16k=dict(type='SWD', num_images=16384, image_shape=(3, 64, 64)))
