from false_positive_reduction.config import conf
import sys
import numpy as np
from false_positive_reduction.net.fpred_networks import NetCombineClsSegMod
import torch
from skimage import measure
from scipy.ndimage import zoom
# sys.path.append("/home/link/data/algorithm/")
from resample import preprocess_resample
from utils import load_model, separate_to_batches, _get_cube_from_volume, find_center, Nodule_info


def _fp_red(volume, pcube, net, device, thres=0.45):
    with torch.no_grad():
        if type(net) == tuple:
            for n in net:
                n.eval()
        else:
            net.eval()
        if np.sum(pcube) == 0:
            return pcube
        pcube = measure.label(pcube)
        patch_lst = []
        pred_lst = []
        for i in range(1, np.max(pcube) + 1):
            p1 = (pcube == i)
            c1, c2, c3 = find_center(p1)
            patch = _get_cube_from_volume(volume, (c3, c2, c1), side_len=32)
            patch_lst.append(patch)
        patch_lst = np.array(patch_lst)
        batch_lst = separate_to_batches(patch_lst, 32)
        for b in batch_lst:
            b = b.reshape([-1, 1, b.shape[1], b.shape[2], b.shape[3]])
            b = torch.from_numpy(b)
            b = b.to(device=device, dtype=torch.float32)
            if type(net) != tuple:
                pred = net(b)
                if type(pred) == tuple:
                    pred = pred[0]
                pred = torch.softmax(pred, dim=1)
                pred = np.array(pred.data.cpu())[:, 1]
                pred[pred >= thres] = 1
                pred[pred < thres] = 0
                pred_lst.append(pred)
            else:
                pred1_lst = []
                for num, n in enumerate(net):
                    if num == 0:
                        pred = n(b[:, :, 4:36, 4:36, 4:36])
                    elif num == 1:
                        pred = n(b)
                    elif num == 2:
                        pred = n(b[:, :, 8:32, 8:32, 8:32])
                    if type(pred) == tuple:
                        pred = pred[0]
                    pred = torch.softmax(pred, dim=1)
                    pred = np.array(pred.data.cpu())[:, 1]
                    # pred[pred >= thres] = 1
                    # pred[pred < thres] = 0
                    pred1_lst.append(pred)
                pred1 = np.array(pred1_lst)
                pred1 = np.average(pred1, axis=0, weights=[0.4, 0.3, 0.3])
                pred1[pred1 >= thres] = 1
                pred1[pred1 < thres] = 0
                pred_lst.append(pred1)
        pred_lst = np.concatenate(pred_lst, axis=0)
        for i in range(1, np.max(pcube) + 1):
            if pred_lst[i - 1] == 0:
                pcube[pcube == i] = 0
            elif pred_lst[i - 1] == 1:
                pcube[pcube == i] = 1
            else:
                raise ValueError()
        return pcube


class FalsePositiveReduction():
    def __init__(self):
        self.config = conf

    def get_nodule_stage2(self, ct_info):
        img, label1 = self.preprocess(ct_info, self.config)
        label = self.do_inference(ct_info, img, label1, self.config)
        label = self.postprocess(ct_info, label, self.config)
        ct_info.nodule_stage2_arr = label

    def preprocess(self, ct_info, config):
        arr_resampled, l1_resampled = preprocess_resample(np.float32(ct_info.img_arr), ct_info.nodule_stage1_arr, ct_info.spacing[2], ct_info.spacing[0], 
            config.resample_xy_dist, config.resample_xy_pixel, config.resample_z_dist)
        hi = config.wincenter + config.winwidth / 2
        lo = config.wincenter - config.winwidth / 2
        arr_resampled = (arr_resampled - lo) / (hi - lo)
        arr_resampled[arr_resampled > 1] = 1
        arr_resampled[arr_resampled < 0] = 0
        arr_resampled *= 0.85
        l1_resampled[l1_resampled > 0.5] = 1
        l1_resampled[l1_resampled <= 0.5] = 0
        return arr_resampled, l1_resampled

    def do_inference(self, ct_info, img, label1, config):
        net = NetCombineClsSegMod(config.in_channels, config.out_channels_cls, config.out_channels_seg, config.net3d_num_channels, config.net2d_num_channels, dropout_rate=config.dropout_rate, mod=1)
        load_model(net, config.model_file)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        pcube = _fp_red(img, label1, net, device, config.nodule_prob_thres)
        return pcube

    def postprocess(self, ct_info, label, config):
        predf = label
        image = ct_info.img_arr
        spacing = ct_info.spacing
        predf = zoom(predf, (image.shape[0] / predf.shape[0], config.resample_xy_dist / spacing[2], config.resample_xy_dist / spacing[2]), order=0)
        ori_xy_len = image.shape[2]
        if predf.shape[1] < ori_xy_len:
            start = int((ori_xy_len - predf.shape[1]) / 2)
            predf1 = np.zeros((predf.shape[0], ori_xy_len, ori_xy_len))
            predf1[:, start:start + predf.shape[1], start:start + predf.shape[1]] = predf
            predf = predf1
        else:
            start = -int((ori_xy_len - predf.shape[1]) / 2)
            predf = predf[:, start:start + ori_xy_len, start:start + ori_xy_len]

        nodules = measure.regionprops(measure.label(predf, background=0))
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
        
        return predf


