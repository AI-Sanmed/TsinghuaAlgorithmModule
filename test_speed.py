import SimpleITK as sitk
from segment_segmentation.segmentation import SegmentSegmentation
from nodule_detection.segmentation import NoduleDetection
from utils import CT_info
from nodule_register.registration import Registration
from false_positive_reduction.segmentation import FalsePositiveReduction
from nodule_segmentation.segmentation import NoduleSegmentation
from nodule_classification.classification import NoduleClassification
from lobe_location.locate import Locate
import time
import os
import numpy as np


if __name__ == "__main__":
    num_test = 1
    detection_lst = []
    seg_lst = []
    tex_lst = []
    base_path = "/home/user/dataset/nodule_shareDir/SegTestMain_v3.0_756_Jan16_origin/img/"
    lst = os.listdir(base_path)
    for num, i in enumerate(lst):
        ct_info = CT_info(base_path + i)
        d = NoduleDetection()
        f = FalsePositiveReduction()
        s = NoduleSegmentation()
        c = NoduleClassification()
        t0 = time.time()
        d.get_nodule_stage1(ct_info)
        f.get_nodule_stage2(ct_info)
        t1 = time.time()
        s.get_nodule_stage3(ct_info)
        t2 = time.time()
        c.get_nodule_stage3(ct_info)
        t3 = time.time()
        detection_lst.append(t1 - t0)
        seg_lst.append(t2 - t1)
        tex_lst.append(t3 - t2)
        print(num)

    with open("speed_{}.txt".format(num_test), 'a') as F:
        F.write("Detection time: {:.3f}, seg time: {:.3f}, tex time: {:.3f}\n".format(np.mean(detection_lst), np.mean(seg_lst), np.mean(tex_lst)))

        