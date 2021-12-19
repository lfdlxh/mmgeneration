# dataset settings
dataset_type = 'mmcls.EncryptTrafficPic'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=24,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='/root/autodl-tmp/encrypted_traffic_data/da_data/train',
        pipeline=train_pipeline,
         ),
    val=dict(
        type=dataset_type,
        data_prefix='/root/autodl-tmp/encrypted_traffic_data/da_data/test',
        ann_file='data/imagenet21k/meta/val.txt',
        pipeline=test_pipeline,
         ),
    test=dict(
        type=dataset_type,
        data_prefix='/root/autodl-tmp/encrypted_traffic_data/da_data/test',
        ann_file='data/imagenet21k/meta/val.txt',
        pipeline=test_pipeline,
         ))
