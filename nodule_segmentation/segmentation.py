from nodule_segmentation.config import conf
import sys
import numpy as np
from nodule_segmentation.net.precise_seg_net import DenseUNet3DGuideEdgeDistLower2ConvCat
from scipy.ndimage import zoom
import torch
from skimage import measure
sys.path.append("/home/link/data/algorithm/")
from resample import preprocess_resample
from utils import load_model, separate_to_batches, _get_cube_from_volume, safe_input, Nodule_info


class NoduleSegmentation():
    def __init__(self):
        self.config = conf

    def get_nodule_stage3(self, ct_info):
        arr_lst, center_lst = self.preprocess(ct_info, None, self.config)
        label_lst = self.do_inference(ct_info, arr_lst, self.config)
        label = self.postprocess(ct_info, label_lst, center_lst, self.config)
        ct_info.nodule_stage3_arr = label

    def preprocess(self, ct_info, label, config):
        arr_lst = []
        center_lst = []  # z, y, x
        img = ct_info.img_arr
        lo = config.wincenter - config.winwidth / 2
        hi = config.wincenter + config.winwidth / 2
        img = (img - lo) / (hi - lo)
        img[img > 1] = 1
        img[img < 0] = 0
        img *= 0.85
        for _, v in ct_info.nodules.items():
            center = [v.centerZ, v.centerY, v.centerX]
            center_lst.append(center)
            center1 = list(reversed(center))
            arr = _get_cube_from_volume(img, center1, side_len=32)
            arr_lst.append(arr)
        return np.array(arr_lst), center_lst

    def do_inference(self, ct_info, arr_lst, config):
        net = DenseUNet3DGuideEdgeDistLower2ConvCat(config.in_channels, config.out_channels, log=config.log_filter)
        load_model(net, config.model_file)
        arr_lst = separate_to_batches(arr_lst, batch_size=config.test_batch_size)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        pred_lst = []
        with torch.no_grad():
            for img in arr_lst:
                img = torch.from_numpy(img)
                img = torch.unsqueeze(img, 1)
                img = img.to(device=device, dtype=torch.float32)
                pred = net(img)
                pred = pred[0]
                pred = torch.sigmoid(pred)
                pred = np.array(pred.data.cpu())[:, 0, :, :, :]
                pred[pred >= 0.5] = 1
                pred[pred < 0.5] = 0
                pred_lst.append(pred)
        if len(pred_lst):
            pred_lst = np.concatenate(pred_lst, axis=0)
        return pred_lst

    def postprocess(self, ct_info, label_lst, center_lst, config):
        label = np.zeros(ct_info.img_arr.shape)
        for l, c in zip(label_lst, center_lst):
            new_c = (max(c[0] - 1, 0), max(c[1] - 1, 0), max(c[2] - 1, 0))
            label = safe_input(label, new_c, l)
        ct_info.nodules = {}
        nodules = measure.regionprops(measure.label(label, background=0))
        for nodule in nodules:
            num = str(nodule.label)
            box_yuan = nodule.bbox
            z_min, y_min, x_min, z_max, y_max, x_max = box_yuan

            nod = Nodule_info()
            nod.box = box_yuan
            nod.xBegin = x_min
            nod.xEnd = x_max
            nod.yBegin = y_min
            nod.yEnd = y_max
            nod.zBegin = z_min
            nod.zEnd = z_max

            nod.centerX = int(round((x_min + x_max) / 2))
            nod.centerY = int(round((y_min + y_max) / 2))
            nod.centerZ = int(round((z_min + z_max) / 2))
            ct_info.nodules[num] = nod
        return label



        