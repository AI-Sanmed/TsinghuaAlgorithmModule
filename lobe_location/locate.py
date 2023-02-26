import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom, distance_transform_edt

def get_min_dist(volume, point, spacing, downsample=3): # volume is labeled
    volume = zoom(volume, [1 / downsample, 1 / downsample, 1 / downsample], order=0)
    volume = np.int32(volume)
    point = [int(round(point[0] / downsample)), int(round(point[1] / downsample)), int(round(point[2] / downsample))]
    dist_lst = []
    dist_lst1 = []
    ind_lst = []
    for i in range(1, np.max(volume) + 1):
        vi = (volume == i)
        vi = 1 - vi
        vid = distance_transform_edt(vi)
        vid1, ind = distance_transform_edt(1 - vi, sampling=spacing, return_indices=True)
        dist_lst.append(vid[point[0]][point[1]][point[2]])
        dist_lst1.append(vid1[point[0]][point[1]][point[2]])
        ind_lst.append(ind[:, point[0]:point[0] + 1, point[1]:point[1] + 1, point[2]:point[2] + 1].squeeze())
    dist_lst = np.array(dist_lst)
    idx = np.argmin(dist_lst, axis=0) + 1
    return idx, dist_lst1[idx - 1] * downsample, ind_lst[idx - 1] * downsample


class Locate():
    def __init__(self):
        pass

    def execute(self, ct_info):
        if ct_info.lobe_arr is None:
            raise ValueError("Missing lobe result.")
        if len(ct_info.nodules)  == 0:
            raise ValueError("Missing nodule precise segmentation result.")
        for k, v in ct_info.nodules.items():
            point = [v.centerZ, v.centerY, v.centerX]
            idx, dist, ind = get_min_dist(ct_info.lobe_arr, point, ct_info.spacing)
            ct_info.nodules[k].lunglobe = idx
            ct_info.nodules[k].minDistToWall = dist
            ct_info.nodules[k].minDistToWallInd = list(ind)
