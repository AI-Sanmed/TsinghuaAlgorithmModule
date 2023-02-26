from easydict import EasyDict


conf = EasyDict({
    "batch_size": 1,
    "in_channels": 1,
    "out_channels": 1,
    "model_file": "models/detection_model_v1.0.pth",
    "resample": True,
    "resample_xy_dist": 0.8,
    "resample_z_dist": 0.8,
    "resample_xy_pixel": 384,
    "winwidth": 1500,
    "wincenter": -250,
    "test_batch_size": 16,
    "lung_box_crop": True
})