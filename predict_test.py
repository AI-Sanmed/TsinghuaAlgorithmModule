import SimpleITK as sitk
from segment_segmentation.segmentation import SegmentSegmentation
from nodule_detection.segmentation import NoduleDetection
from utils import CT_info
from nodule_register.registration import Registration
from false_positive_reduction.segmentation import FalsePositiveReduction
from nodule_segmentation.segmentation import NoduleSegmentation
from nodule_classification.classification import NoduleClassification
import os
import numpy as np
import time


if __name__ == "__main__":
    base_path = "/home/user/dataset/nodule_shareDir/nodule_test_nii_v2.1/img/"
    save_res_folder = "labeltest_10case/"
    if not os.path.exists(save_res_folder):
        os.mkdir(save_res_folder)
    lst = os.listdir(base_path)
    d = NoduleDetection()
    f = FalsePositiveReduction()
    s = NoduleSegmentation()
    c = NoduleClassification()
    time_lst = []
    for i in lst:
        ct_info = CT_info(base_path + i)
        t1 = time.time()
        d.get_nodule_stage1(ct_info)
        f.get_nodule_stage2(ct_info)
        s.get_nodule_stage3(ct_info)
        c.get_nodule_stage3(ct_info)
        t2 = time.time()
        time_lst.append(t2 - t1)
        # ct_info.write_nii(ct_info.nodule_stage3_arr, save_res_folder + i)
        print("time_cost: {:6f}".format(np.mean(time_lst)))