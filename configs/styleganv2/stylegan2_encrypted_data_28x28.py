_base_ = [
    '../_base_/models/stylegan/stylegan2_base.py',
    '../_base_/datasets/encrypted_traffic_data.py',
    '../_base_/default_runtime.py'
]
model = dict(generator=dict(out_size=28), discriminator=dict(in_size=28))

ema_half_life = 10.  # G_smoothing_kimg

# adjust running config
lr_config = None
checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=20)
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=10000),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema', ),
        interval=1,
        interp_cfg=dict(momentum=0.5**(32. / (ema_half_life * 1000.))),
        priority='VERY_HIGH')
]

total_iters = 300000
metrics = dict(
    fid50k=dict(
        type='FID', num_images=50000, inception_pkl=None, bgr2rgb=True),
    pr50k3=dict(type='PR', num_images=50000, k=3),
    ppl_wend=dict(type='PPL', space='W', sampling='end', num_images=50000))


train_cfg = dict(real_img_key='img')