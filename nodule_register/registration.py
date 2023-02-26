from nodule_register.config import conf
import sys
import numpy as np
from nodule_register.registration_pipeline_new import pipeline
import SimpleITK as sitk
sys.path.append("/home/link/data/algorithm/")
from resample import preprocess_resample
from utils import load_model, separate_to_batches, CT_info


class Registration():
    def __init__(self):
        self.config = conf

    def registration(self, ct_info_1, ct_info_2, config):
        orii = ct_info_1.img_itk
        tari = ct_info_2.img_itk
        oril = sitk.GetImageFromArray(ct_info_1.nodule_stage3_arr)
        tarl = sitk.GetImageFromArray(ct_info_2.nodule_stage3_arr)
        reg_idx0, reg_idx, tar_paired, ori_cen, tar_cen = pipeline(orii, tari, oril, tarl, dist_thres=config.dist_thres)
        lst = []
        for num, i in enumerate(reg_idx0):
            to_append = []
            to_append.append(list(ori_cen[num]))
            if reg_idx[i] >= 0:
                to_append.append(list(tar_cen[reg_idx[reg_idx0[num]]]))
            else:
                to_append.append(None)
            lst.append(to_append)
        for num, i in enumerate(tar_paired):
            if i < 0:
                lst.append([None, list(tar_cen[num])])
        return lst


if __name__ == "__main__":
    ct_info_1 = CT_info("/home/user/dataset/nodule_shareDir/nodule_nii_case/img/CT00482.nii.gz")
    ct_info_2 = CT_info("/home/user/dataset/nodule_shareDir/nodule_nii_case/img/CT00482_2.nii.gz")
    ct_info_1.nodule_stage3_arr = sitk.GetArrayFromImage(sitk.ReadImage("/home/user/dataset/nodule_shareDir/nodule_nii_case/label/CT00482.nii.gz"))
    ct_info_2.nodule_stage3_arr = sitk.GetArrayFromImage(sitk.ReadImage("/home/user/dataset/nodule_shareDir/nodule_nii_case/label/CT00482_2.nii.gz"))
    r = Registration()
    r.registration(ct_info_1, ct_info_2, r.config)

