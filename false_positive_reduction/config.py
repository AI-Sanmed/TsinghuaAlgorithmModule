from easydict import EasyDict


conf = EasyDict({
    "test_batch_size": 16,
    "in_channels": 1,
    "out_channels_seg": 1,
    "out_channels_cls": 2,
    "dropout_rate": 0.05,
    "model_file": "models/fpred_model_v1.0.pth",
    "resample": True,
    "resample_xy_dist": 0.8,
    "resample_z_dist": 0.8,
    "resample_xy_pixel": 384,
    "winwidth": 1500,
    "wincenter": -250,
    "net2d_num_channels": 64,
    "net3d_num_channels": 32,
    "nodule_prob_thres": 0.5
})