import numpy as np
from scipy.ndimage import center_of_mass
import SimpleITK as sitk
from scipy import ndimage


def ir(x):
    return int(round(x))


def crop_mask_area(cube, mask):
    idx = np.where(mask == 1)
    ax0_lo = np.min(idx[0])
    ax0_hi = np.max(idx[0])
    ax1_lo = np.min(idx[1])
    ax1_hi = np.max(idx[1])
    ax2_lo = np.min(idx[2])
    ax2_hi = np.max(idx[2])
    slc = [slice(ax0_lo, ax0_hi), slice(ax1_lo, ax1_hi), slice(ax2_lo, ax2_hi)]
    return cube[tuple(slc)], tuple(slc)


def make_distance_map(cube, lobe_mask):
    """
    :param cube: input cube with value 0 as bg and value 1 as branch qiguan
    :param lobe_mask: mask of lobe
    :return: distance transmap in lobe region
    """
    cube = np.float32(cube)
    area, slc = crop_mask_area(cube, lobe_mask)
    area = 1 - area
    dt = ndimage.distance_transform_edt(area)
    cube[slc] = dt
    cube = cube * lobe_mask
    return cube


def get_bbox(vol, perc=95):
    idxs = np.where(vol)
    a = (100 - perc) / 2
    xmin = np.percentile(idxs[0], a)
    xmax = np.percentile(idxs[0], 100 - a)
    ymin = np.percentile(idxs[1], a)
    ymax = np.percentile(idxs[1], 100 - a)
    zmin = np.percentile(idxs[2], a)
    zmax = np.percentile(idxs[2], 100 - a)
    return [xmin, xmax, ymin, ymax, zmin, zmax]


def calc_plane_params(point, vec):  # vec是法向量。point和vec都是z，y，x
    a = vec[0]
    b = vec[1]
    c = vec[2]
    d = (-a * point[0] - b * point[1] - c * point[2])
    return a, b, c, d


def plane_in_vol(vol, a, b, c, d):
    vx = np.arange(vol.shape[0])
    v = np.arange(vol.shape[2])
    dim = np.argmax(np.abs((a, b, c)))
    if dim == 0:
        y, z = np.meshgrid(v, v)
        x = np.rint((b*y + c*z + d) / -a).astype(int)
        x[x >= vol.shape[0]] = vol.shape[0] - 1
        x[x < 0] = 0
    elif dim == 1:
        x, z = np.meshgrid(vx, v)
        y = np.rint((a*x + c*z + d) / -b).astype(int)
        y[y >= vol.shape[1]] = vol.shape[1] - 1
        y[y < 0] = 0
    elif dim == 2:
        x, y = np.meshgrid(vx, v)
        z = np.rint((a*x + b*y + d) / -c).astype(int)
        z[z >= vol.shape[2]] = vol.shape[2] - 1
        z[z < 0] = 0
    plane_voxels = np.dstack([x,y,z]).reshape(-1,3)
    v1 = np.zeros(vol.shape)
    v1[plane_voxels[:, 0], plane_voxels[:, 1], plane_voxels[:, 2]] = 1
    return v1


def seg_rul(lobe):
    lobe = (lobe == 1)
    center = center_of_mass(lobe)
    center = np.int32(np.round(center))
    marker1 = np.zeros(lobe.shape)
    marker1[center[0]:, center[1], :] = 1
    vec2 = [1.732 / 2, 0.5, 0]  # z, y, x
    params2 = calc_plane_params(center, vec2)
    marker2 = plane_in_vol(lobe, *params2)
    marker2[:, :center[1], :] = 0
    vec3 = [-1.732 / 2, 0.5, 0]
    params3 = calc_plane_params(center, vec3)
    marker3 = plane_in_vol(lobe, *params3)
    marker3[:, center[1]:, :] = 0
    map1 = make_distance_map(marker1, lobe)
    map2 = make_distance_map(marker2, lobe)
    map3 = make_distance_map(marker3, lobe)
    all_maps = np.stack([map1, map2, map3])
    all_maps = np.argmin(all_maps, axis=0)
    all_maps += 1
    all_maps = all_maps * lobe
    return all_maps


def seg_rml(lobe):
    lobe = (lobe == 2)
    bbox = get_bbox(lobe)
    point = [bbox[1], bbox[3], bbox[5]]  # z任意，x，y均最大
    vec1= [0, -1.732 / 2, 0.5]  # z, y, x
    params1 = calc_plane_params(point, vec1)
    marker1 = plane_in_vol(lobe, *params1)
    vec2 = [0, -0.5, 1.732 / 2]  # z, y, x
    params2 = calc_plane_params(point, vec2)
    marker2 = plane_in_vol(lobe, *params2)
    map1 = make_distance_map(marker1, lobe)
    map2 = make_distance_map(marker2, lobe)
    all_maps = np.stack([map1, map2])
    all_maps = np.argmin(all_maps, axis=0)
    all_maps += 4
    all_maps = all_maps * lobe
    return all_maps


def seg_rll(lobe):
    lobe = (lobe == 3)
    bbox = get_bbox(lobe)
    zlevel_6 = int(round(bbox[0] + 0.8 * (bbox[1] - bbox[0])))
    zmax_7 = int(round(bbox[0] + 0.7 * (bbox[1] - bbox[0])))
    marker6 = np.zeros(lobe.shape)
    marker6[zlevel_6, :, :] = 1
    point = [int(round(bbox[0])), int(round((bbox[2] + bbox[3]) / 2)), int(round(bbox[4] * 0.4 + bbox[5] * 0.6))]
    vec7 = [0, 0.5, 1.732 / 2]
    params7 = calc_plane_params(point, vec7)
    marker7 = plane_in_vol(lobe, *params7)
    marker7[zmax_7:, :, :] = 0
    marker7[:, point[1]:, :] = 0
    vec8 = [0, -0.965, 0.259]
    params8 = calc_plane_params(point, vec8)
    marker8 = plane_in_vol(lobe, *params8)
    marker8[zmax_7:, :, :] = 0
    marker8[:, point[1]:, :] = 0
    vec9 = [0, 0.707, 0.707]
    params9 = calc_plane_params(point, vec9)
    marker9 = plane_in_vol(lobe, *params9)
    marker9[zmax_7 - 5:, :, :] = 0
    marker9[:, :point[1], :] = 0
    vec10 = [0, 0.707, -0.707]
    params10 = calc_plane_params(point, vec10)
    marker10 = plane_in_vol(lobe, *params10)
    marker10[zmax_7 - 5:, :, :] = 0
    marker10[:, :point[1], :] = 0
    map6 = make_distance_map(marker6, lobe)
    map7 = make_distance_map(marker7, lobe)
    map8 = make_distance_map(marker8, lobe)
    map9 = make_distance_map(marker9, lobe)
    map10 = make_distance_map(marker10, lobe)
    all_maps = np.stack([map6, map7, map8, map9, map10])
    all_maps = np.argmin(all_maps, axis=0)
    all_maps += 6
    all_maps = all_maps * lobe
    return all_maps


def seg_lul(lobe):
    lobe = (lobe == 4)
    bbox = get_bbox(lobe)
    point = [ir(bbox[0] * 0.6 + bbox[1] * 0.4), ir(bbox[2] * 0.33 + bbox[3] * 0.67), bbox[5]]
    vec1 = [0.087, 0.996, 0]
    params1 = calc_plane_params(point, vec1)
    marker1 = plane_in_vol(lobe, *params1)
    marker1[:point[0], :, :] = 0
    vec2 = [0.574, 0.819, 0]
    params2 = calc_plane_params(point, vec2)
    marker2 = plane_in_vol(lobe, *params2)
    marker2[:point[0], :, :] = 0
    vec3 = [0.906, 0.423, 0]
    params3 = calc_plane_params(point, vec3)
    marker3 = plane_in_vol(lobe, *params3)
    marker3[:point[0], :, :] = 0
    vec4 = [0.819, -0.574, 0]
    params4 = calc_plane_params(point, vec4)
    marker4 = plane_in_vol(lobe, *params4)
    marker4[point[0]:, :, :] = 0
    map1 = make_distance_map(marker1, lobe)
    map2 = make_distance_map(marker2, lobe)
    map3 = make_distance_map(marker3, lobe)
    map4 = make_distance_map(marker4, lobe)
    all_maps = np.stack([map1, map2, map3, map4])
    all_maps = np.argmin(all_maps, axis=0)
    all_maps += 11
    all_maps = all_maps * lobe
    return all_maps


def seg_lll(lobe):
    lobe = (lobe == 5)
    bbox = get_bbox(lobe)
    zlevel_5 = int(round(bbox[0] + 0.85 * (bbox[1] - bbox[0])))
    zmax_7 = int(round(bbox[0] + 0.7 * (bbox[1] - bbox[0])))
    marker5 = np.zeros(lobe.shape)
    marker5[zlevel_5, :, :] = 1
    point6 = [ir(bbox[0] * 0.65 + bbox[1] * 0.35), ir(bbox[2] * 0.83 + bbox[3] * 0.17), ir(bbox[4])]
    vec6 = [-0.5, 1.732 / 2, 0]
    param6 = calc_plane_params(point6, vec6)
    marker6 = plane_in_vol(lobe, *param6)
    marker6[zmax_7:, :, :] = 0
    point7 = [ir(bbox[0] * 0.65 + bbox[1] * 0.35), ir(bbox[2] * 0.5 + bbox[3] * 0.5), ir(bbox[4])]
    vec7 = [-0.259, 0.966, 0]
    param7 = calc_plane_params(point7, vec7)
    marker7 = plane_in_vol(lobe, *param7)
    marker7[zlevel_5:, :, :] = 0
    marker8 = np.zeros(lobe.shape)
    marker8[:, ir(bbox[2] * 0.17 + bbox[3] * 0.83), :] = 1
    marker8[zmax_7:, :, :] = 0
    map5 = make_distance_map(marker5, lobe)
    map6 = make_distance_map(marker6, lobe)
    map7 = make_distance_map(marker7, lobe)
    map8 = make_distance_map(marker8, lobe)
    all_maps = np.stack([map5, map6, map7, map8])
    all_maps = np.argmin(all_maps, axis=0)
    all_maps += 15
    all_maps = all_maps * lobe
    return all_maps



if __name__ == "__main__":
    lobe = sitk.ReadImage("/home/link/data/algorithm/segment_segmentation/ATM_003_0000_lobe_mask.nii.gz")
    lobe = sitk.GetArrayFromImage(lobe)
    maps = seg_rul(lobe)
    maps += seg_rml(lobe)
    maps += seg_rll(lobe)
    maps += seg_lul(lobe)
    maps += seg_lll(lobe)
    a = sitk.GetImageFromArray(maps)
    sitk.WriteImage(a, '1.nii.gz')