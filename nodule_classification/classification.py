from nodule_classification.config import conf
import sys
import numpy as np
from nodule_classification.net.texture_cls_net import Net32SETextureAux
from scipy.ndimage import zoom
import torch
from skimage import measure
sys.path.append("/home/link/data/algorithm/")
from resample import preprocess_resample
from utils import load_model, separate_to_batches, _get_cube_from_volume, safe_input, Nodule_info


def process_pred(pred_t, pred_c):
    pred = np.zeros(len(pred_t), dtype=np.int16)
    for i in range(len(pred_t)):
        pt = pred_t[i]
        pc = pred_c[i][1]
        if pt >= 1.5:
            pred[i] = 2
        elif pt >= 0.5:
            pred[i] = 1
        else:
            if pc >= 0.5:
                pred[i] = 3
            else:
                pred[i] = 0
    return pred


class NoduleClassification():
    def __init__(self):
        self.config = conf

    def get_nodule_stage3(self, ct_info):
        arr_lst, center_lst, nodule_num_lst = self.preprocess(ct_info, None, self.config)
        pred_lst = self.do_inference(ct_info, arr_lst, self.config)
        self.postprocess(ct_info, pred_lst, nodule_num_lst, self.config)

    def preprocess(self, ct_info, label, config):
        arr_lst = []
        center_lst = []  # z, y, x
        nodule_num_lst = []
        img = ct_info.img_arr
        lo = config.wincenter - config.winwidth / 2
        hi = config.wincenter + config.winwidth / 2
        img = (img - lo) / (hi - lo)
        img[img > 1] = 1
        img[img < 0] = 0
        img *= 0.85
        for k, v in ct_info.nodules.items():
            nodule_num_lst.append(k)
            center = [v.centerZ, v.centerY, v.centerX]
            center_lst.append(center)
            center1 = list(reversed(center))
            arr = _get_cube_from_volume(img, center1, side_len=32)
            arr_lst.append(arr)
        return np.array(arr_lst), center_lst, nodule_num_lst

    def do_inference(self, ct_info, arr_lst, config):
        net = Net32SETextureAux(config.in_channels, 1, reg_dropout=config.reg_dropout_rate, cls_dropout=config.cls_dropout_rate)
        load_model(net, config.model_file)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        arr_lst = separate_to_batches(arr_lst, batch_size=config.test_batch_size)
        pred_lst = []
        with torch.no_grad():
            for img in arr_lst:
                img = torch.from_numpy(img)
                img = torch.unsqueeze(img, 1)
                img = img.to(device=device, dtype=torch.float32)
                pred_t, pred_c = net(img)
                pred_t = pred_t[0]
                pred_c = torch.softmax(pred_c, dim=1)
                pred_c = np.array(pred_c.data.cpu())
                pred_t = np.array(pred_t.data.cpu())
                pred = process_pred(pred_t, pred_c)
                pred_lst.append(pred)
        if len(pred_lst):
            pred_lst = np.concatenate(pred_lst, axis=0)
        return pred_lst

    def postprocess(self, ct_info, pred_lst, nodule_num_lst, config):
        pred_lst = list(pred_lst)
        for p, n in zip(pred_lst, nodule_num_lst):
            ct_info.nodules[n].noduleType = p



        