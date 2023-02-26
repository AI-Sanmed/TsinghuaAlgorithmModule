from easydict import EasyDict

conf = EasyDict({
    "test_batch_size": 4,
    "in_channels": 1,
    "reg_dropout_rate": 0.1,
    "cls_dropout_rate": 0.2,
    "model_file": "models/classification_model_v1.0.pth",
    "winwidth": 1500,
    "wincenter": -250,
    "log_filter": True
})