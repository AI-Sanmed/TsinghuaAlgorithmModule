from easydict import EasyDict

conf = EasyDict({
    "test_batch_size": 4,
    "in_channels": 1,
    "out_channels": 1,
    "dropout_rate": 0.0,
    "model_file": "models/precise_seg_model_v1.0.pth",
    "winwidth": 1500,
    "wincenter": -250,
    "log_filter": True
})