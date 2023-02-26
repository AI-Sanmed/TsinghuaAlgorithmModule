import SimpleITK as sitk
from segment_segmentation.segmentation import SegmentSegmentation
from nodule_detection.segmentation import NoduleDetection
from utils import CT_info,mkdir,get_lung_mask
from nodule_register.registration import Registration
from false_positive_reduction.segmentation import FalsePositiveReduction
from nodule_segmentation.segmentation import NoduleSegmentation
from nodule_classification.classification import NoduleClassification
from lobe_location.locate import Locate
import os

if __name__ == "__main__":
    img_path = r'/data/boyu/dataset/daiding_B4_471_0222/img'
    save_path =r'/data/boyu/dataset/daiding_B4_471_0222/pred'
    # img_path = r'input'
    # save_path =r'output'
    mkdir(save_path)
    for i in os.listdir(img_path):
        print(i)
        img_file = os.path.join(img_path,i)
        save_file = os.path.join(save_path,i)


        ct_info = CT_info(img_file)

        get_lung_mask(ct_info)

        d = NoduleDetection()
        f = FalsePositiveReduction()
        s = NoduleSegmentation()
        c = NoduleClassification()
        d.get_nodule_stage1(ct_info)
        f.get_nodule_stage2(ct_info)
        s.get_nodule_stage3(ct_info)
        ct_info.write_nii(ct_info.nodule_stage3_arr, save_file)
