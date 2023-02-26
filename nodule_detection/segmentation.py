from nodule_detection.config import conf
import sys
import numpy as np
from nodule_detection.net.nodule_detection_model import UNet3D
from scipy.ndimage import zoom
import torch
sys.path.append("/home/link/data/algorithm/")
from resample import preprocess_resample
from utils import load_model, separate_to_batches, compute_lung_extendbox


class NoduleDetection():
    def __init__(self):
        self.config = conf

    def get_nodule_stage1(self, ct_info):
        img = self.preprocess(ct_info, None, self.config)
        label = self.do_inference(ct_info, img, self.config)
        label = self.postprocess(ct_info, label, self.config)
        ct_info.nodule_stage1_arr = label

    def preprocess(self, ct_info, label, config):
        arr_resampled, lung_resampled = preprocess_resample(np.float32(ct_info.img_arr), np.float32(ct_info.lung_mask), ct_info.spacing[2], ct_info.spacing[0], 
            config.resample_xy_dist, config.resample_xy_pixel, config.resample_z_dist)
        self.lung_bbox = compute_lung_extendbox(lung_resampled)
        hi = config.wincenter + config.winwidth / 2
        lo = config.wincenter - config.winwidth / 2
        arr_resampled = (arr_resampled - lo) / (hi - lo)
        arr_resampled[arr_resampled > 1] = 1
        arr_resampled[arr_resampled < 0] = 0
        arr_resampled *= 0.85
        return arr_resampled

    def do_inference(self, ct_info, img, config):
        net = UNet3D(config.in_channels, config.out_channels)
        load_model(net, config.model_file)
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, 0)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        img = img.to(device)
        image = img
        if config.lung_box_crop:
            predf0 = np.zeros(img.shape[1:4])
            box = self.lung_bbox
            image = image[:, box[0, 0]: box[0, 1], box[1, 0]: box[1, 1], box[2, 0]: box[2, 1]]  
        with torch.no_grad():
            test_slices_interval = 32
            test_cube_slices = 64
            xy_size = 64
            xy_interval = 32
            batch_size = config.test_batch_size
            image_test_list = []
            test_result_list = []
            times = np.zeros(np.array(image.shape[1:4]))  # 表示每个格子计算的次数
            predf = np.zeros(np.array(image.shape[1:4]))
            assert image.shape[0] == 1
            inds = range(0, image.shape[1] - test_cube_slices, test_slices_interval)
            z_begins_lst = [i for i in inds]
            z_begins_lst.append(image.shape[1] - test_cube_slices)
            for ind in inds:
                image_test_list.append(image[:, ind:ind + test_cube_slices, :, :])
            image_test_list.append(image[:, image.shape[1] - test_cube_slices:image.shape[1], :, :])

            for num, (i, z) in enumerate(zip(image_test_list, z_begins_lst)):
                i = i[0]
                inds1 = range(0, image.shape[2] - xy_size, xy_interval)
                begins_lst = [i for i in inds1]
                begins_lst.append(image.shape[2] - xy_size)
                inds2 = range(0, image.shape[3] - xy_size, xy_interval)
                begins_lst2 = [i for i in inds2]
                begins_lst2.append(image.shape[3] - xy_size)
                pp = np.zeros((64, 256, 256))
                patch_lst = []
                for ix in begins_lst:
                    for iy in begins_lst2:
                        patch_lst.append(i[:, ix:ix + xy_size, iy:iy + xy_size])
                i = torch.stack(patch_lst)
                sh = i.shape
                i = i.reshape([sh[0], 1, sh[1], sh[2], sh[3]])
                batch_lst = separate_to_batches(i, batch_size)
                p_lst = []
                for i1 in batch_lst:
                    i1 = i1.to(device=device, dtype=torch.float32)
                    pred = net(i1)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    pred = pred.data.cpu()
                    p_lst.append(pred)
                p = torch.cat(p_lst, dim=0)
                p = np.array(p.data.cpu())
                p = np.squeeze(p)
                p = 1 / (1 + np.exp(-p))
                idx = 0
                for patch in p:
                    x_coord = begins_lst[idx // len(begins_lst2)]
                    y_coord = begins_lst2[idx % len(begins_lst2)]
                    predf[z:z + test_cube_slices, x_coord:x_coord + xy_size, y_coord:y_coord + xy_size] += patch
                    times[z:z + test_cube_slices, x_coord:x_coord + xy_size, y_coord:y_coord + xy_size] += np.ones(
                        (test_cube_slices, xy_size, xy_size))
                    idx += 1
                test_result_list.append(pp)

            predf[predf >= 1] = 1
            predf[predf < 1] = 0
            if config.lung_box_crop:
                predf0[box[0, 0]: box[0, 1], box[1, 0]: box[1, 1], box[2, 0]: box[2, 1]] = predf
                predf = predf0
            predf = np.bool_(predf)
            return predf

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
        return predf
        