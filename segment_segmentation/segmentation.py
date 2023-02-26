import sys
sys.path.append("segment_segmentation/")
from config import conf
import torch
import numpy as np
from net.lobe_net import UNet3DGuideEdgeLower2Cat
from scipy.ndimage import zoom
import sys
from fine_segment import segment_lung_mask
from utils import load_model, separate_to_batches, _get_cube_from_volume, find_center
from segment_segmentation.segment_segmentation_fast import do_seg


def compute_lung_extendbox(mask, margin=(5, 5, 5)):
    zz, yy, xx = np.where(mask)
    min_box = np.max([[0, zz.min() - margin[0]],
                      [0, yy.min() - margin[1]],
                      [0, xx.min() - margin[2]]], axis=1, keepdims=True)

    max_box = np.min([[mask.shape[0], zz.max() + margin[0]],
                      [mask.shape[1], yy.max() + margin[1]],
                      [mask.shape[2], xx.max() + margin[2]]], axis=1, keepdims=True)

    box = np.concatenate([min_box, max_box], axis=1)
    return box


class SegmentSegmentation():
    def __init__(self) -> None:
        self.config = conf
        self.sh = None
        self.box = None

    def segment_segmentation(self, ct_info):
        img = self.preprocess(ct_info, self.config)
        label = self.do_inference(ct_info, img, self.config)
        label = self.postprocess(ct_info, label, self.config)
        ct_info.lobe_arr = label
        segment_label = self.do_segment_segmentation(ct_info, self.config)
        ct_info.segment_arr = segment_label

    def preprocess(self, ct_info, config):
        mask = segment_lung_mask(ct_info.img_arr)
        lo = config.wc - config.ww / 2
        hi = config.wc + config.ww / 2
        image = ct_info.img_arr
        image = (image - lo) / (hi - lo)
        image[image > 1] = 1
        image[image < 0] = 0
        image = image * 0.85
        ct_info.lung_mask = mask

        box = compute_lung_extendbox(mask)
        self.box = box
        image = image[box[0, 0]: box[0, 1], box[1, 0]: box[1, 1], box[2, 0]: box[2, 1]]
        mask = mask[box[0, 0]: box[0, 1], box[1, 0]: box[1, 1], box[2, 0]: box[2, 1]]

        sh = image.shape
        self.sh = sh
        resize_factor = (160 / sh[0], 128 / sh[1], 128 / sh[2])
        image = zoom(image, resize_factor, order=0)
        mask = zoom(mask, resize_factor, order=0)

        return image
    
    def do_inference(self, ct_info, img, config):
        net = UNet3DGuideEdgeLower2Cat(1, 6)
        with torch.no_grad():
            load_model(net, config.model_file)
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
            img = img.to(device=device, dtype=torch.float32)
            label, _ = net(img)
            label = torch.argmax(label, dim=1)
            label = np.array(label.data.cpu())[0]
        return label

    def postprocess(self, ct_info, label, config):
        resize_back_factor = (self.sh[0] / 160, self.sh[1] / 128, self.sh[2] / 128)
        resampled = zoom(label, resize_back_factor, order=0)
        arr = np.zeros(list(reversed(ct_info.img_itk.GetSize())))
        arr[self.box[0, 0]: self.box[0, 1], self.box[1, 0]: self.box[1, 1], self.box[2, 0]: self.box[2, 1]] = resampled
        return arr

    def do_segment_segmentation(self, ct_info, config):
        segment_array = do_seg(ct_info.lobe_arr, ct_info.airway_arr, list(reversed(ct_info.img_itk.GetSpacing())), config.ball_mask, 
            config.do_opening)
        return segment_array

    def lobe_segmentation(self, ct_info):
        img = self.preprocess(ct_info, self.config)
        label = self.do_inference(ct_info, img, self.config)
        label = self.postprocess(ct_info, label, self.config)
        ct_info.lobe_arr = label




