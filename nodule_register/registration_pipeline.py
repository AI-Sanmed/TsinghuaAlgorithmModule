import SimpleITK as sitk
from scipy import ndimage
from skimage import measure
import numpy as np


def calc_distance(arr1, arr2, pix_dist):
    diff = arr1 - arr2
    diff = diff * pix_dist
    diff = np.sqrt(np.sum(diff ** 2))
    return diff


def match_nodules_revise(reg_arr, tar_arr, pix_dist, dist_thres=30):
    reg_idx = np.ones((reg_arr.shape[0])) * -1
    tar_paired = np.ones((tar_arr.shape[0])) * -1
    if reg_arr.shape[1] == 0 or tar_arr.shape[1] == 0:
        return reg_idx, tar_paired
    reg_possible_pairs = []
    for i in range(reg_arr.shape[0]):
        arr1 = reg_arr[i, :]
        dist_arr = []
        for j in range(tar_arr.shape[0]):
            arr2 = tar_arr[j, :]
            dist = calc_distance(arr1, arr2, pix_dist)
            if dist < dist_thres:
                dist_arr.append([j, dist])
        dist_arr = np.array(dist_arr)
        if dist_arr.shape[0]:
            dist_arr = dist_arr[dist_arr[:, 1].argsort()]
        reg_possible_pairs.append(dist_arr)
        if not len(dist_arr):
            continue
        else:
            reg_idx[i] = dist_arr[0][0]
            j = int(reg_idx[i])
            if tar_paired[j] == -1:  # 目前还没有reg中结节与tar的此结节配对
                tar_paired[j] = i
            else:
                i1 = i  # i1表示目前需要调整的reg中的结节，初始化为当前结节i
                while True:
                    this_dist = reg_possible_pairs[i1][0][1]
                    other_dist = reg_possible_pairs[int(tar_paired[j])][0][1]
                    if other_dist >= this_dist:  # 此处的结节比上一个配对的结节要好，上一个需要再找一个
                        i2 = int(tar_paired[j])  # 上一个配对的结节需要调整
                        tar_paired[j] = i1
                        i1 = i2
                        reg_possible_pairs[i1] = np.delete(reg_possible_pairs[i1], np.s_[0], axis=0)
                        if len(reg_possible_pairs[i1]) == 0:
                            reg_idx[i1] = -1  # 上一个不配对任何结节，退出循环
                            break
                        else:  # i1有多个可能的配对结节，此时去掉第一个，找到第二个（即距离第二近的），寻找对应的结节
                            j = int(reg_possible_pairs[i1][0][0])  # j为第二近的tar中的结节的序号
                            reg_idx[i1] = j
                            if tar_paired[j] == -1:  # 此时tar中的结节不再冲突，退出循环
                                tar_paired[j] = i1
                                break
                    else:  # 上一个配对的结节比此处的结节要好，此处需要再找一个
                        reg_possible_pairs[i1] = np.delete(reg_possible_pairs[i1], np.s_[0], axis=0)
                        if len(reg_possible_pairs[i1]) == 0:
                            reg_idx[i1] = -1  # 此处不配对任何结节，退出循环
                            break
                        else:  # i1有多个可能的配对结节，此时去掉第一个，找到第二个（即距离第二近的），寻找对应的结节
                            j = int(reg_possible_pairs[i1][0][0])  # j为第二近的tar中的结节的序号
                            reg_idx[i1] = j
                            if tar_paired[j] == -1:  # 此时tar中的结节不再冲突，退出循环
                                tar_paired[j] = i1
                                break
    return np.int16(reg_idx), np.int16(tar_paired)


def get_dist_matrix(reg_cen, tar_cen, pixel_dist):
    dist_arr = np.zeros((reg_cen.shape[0], tar_cen.shape[0]))
    for num1, i in enumerate(reg_cen):
        for num2, j in enumerate(tar_cen):
            dist_arr[num1][num2] = calc_distance(i, j, pixel_dist)
    return dist_arr


def get_resample_size(ori_size, resample_ratio):
    lst = []
    for num, (o, d) in enumerate(zip(ori_size, resample_ratio)):
        if num < 2:
            lst.append(int(512 * d))
        else:
            lst.append(int(o * d))
    return tuple(lst)


def get_resample_spacing(ori_spacing, resample_ratio):
    lst = []
    for o, d in zip(ori_spacing, resample_ratio):
        lst.append(o / d)
    return tuple(lst)


def command_iteration(method):
    # print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f} : {method.GetOptimizerPosition()}")
    pass


def crop_mask_area(cube, mask):
    idx = np.where(mask == 1)
    ax0_lo = np.min(idx[0])
    ax0_hi = np.max(idx[0])
    ax1_lo = np.min(idx[1])
    ax1_hi = np.max(idx[1])
    ax2_lo = np.min(idx[2])
    ax2_hi = np.max(idx[2])
    slc = [slice(ax0_lo, ax0_hi + 1), slice(ax1_lo, ax1_hi + 1), slice(ax2_lo, ax2_hi + 1)]
    slc = tuple(slc)
    return cube[slc], (ax0_lo, ax1_lo, ax2_lo)


def find_centers(array):
    """
    Find centers of each connective area in the array.
    :param array: ndarray with values of 0 or 1
    :return: array of center coords
    """
    array1 = measure.label(array)
    center_lst = []
    for i in range(1, np.max(array1) + 1):
        array0 = (array1 == i)
        arrayc, hashiqi = crop_mask_area(array0, array0)
        center = ndimage.center_of_mass(arrayc)
        center_lst_0 = []
        for (i, j) in zip(center, hashiqi):
            center_lst_0.append(i + j + 1)  # In clinical use, the first slice is slice 1, the first row is row 1. Here is all 0.
        center_lst.append(center_lst_0)
    if np.max(array) == 0:
        center_lst.append([])
    center_lst = np.array(center_lst)
    return center_lst


def transform_points(ori_cen, tfs):
    lst1 = []
    for i in ori_cen:
        i = tuple(i)
        lst1.append(tfs.TransformPoint(i))
    return np.array(lst1)


def generate_result_string(ori_cen, reg_cen, tar_cen, reg_idx_0, reg_idx, tar_paired):
    string = ""
    reg_idx = np.int16(reg_idx)
    reg_idx_0 = np.int16(reg_idx_0)
    tar_paired_0 = np.int16(tar_paired)
    for i in range(len(ori_cen)):
        string += "Origin: ({}, {}, {}), ".format(ori_cen[i][0], ori_cen[i][1], ori_cen[i][2])
        idx_reg = reg_idx_0[i]
        if idx_reg > -1:
            string += "registrated_coords: ({}, {}, {}), ".format(reg_cen[idx_reg][0], reg_cen[idx_reg][1], reg_cen[idx_reg][2])
            idx_tar = reg_idx[idx_reg]
            if idx_tar > -1:
                string += "paired with nodule in target: ({}, {}, {}).".format(tar_cen[idx_tar][0], tar_cen[idx_tar][1], tar_cen[idx_tar][2])
            else:
                string += "not paired with any nodules in target."
        else:
            string += "registration failed."
        string += "\n"
    idx2 = np.where(tar_paired_0 == -1)
    for j in idx2[0]:
        string += "Target: ({}, {}, {}) not paired with any nodules in origin.\n".format(tar_cen[j][0], tar_cen[j][1], tar_cen[j][2])
    return string


def get_transform(ori, tar):
    """
    Gets transform matrix from pair of CT scans
    :param ori: Moving CT scan read by SimpleITK with data type float32
    :param tar: Fixed CT scan read by SimpleITK with data type float32
    :return: sitk.Transform transformation matrix
    """
    ori.SetOrigin((0, 0, 0))
    tar.SetOrigin((0, 0, 0))
    ori_spacing = 512 / ori.GetSize()[0]  # 对于正常的512*512的CT，应该为1
    tar_spacing = 512 / tar.GetSize()[0]
    ori.SetSpacing((ori_spacing, ori_spacing, 1))
    tar.SetSpacing((tar_spacing, tar_spacing, 1))
    resample_rate = 0.25
    moving_image = sitk.Resample(image1=ori,
                                 size=get_resample_size(ori.GetSize(), (resample_rate, resample_rate, resample_rate)),
                                 transform=sitk.Transform(),
                                 interpolator=sitk.sitkLinear, outputOrigin=ori.GetOrigin(),
                                 outputDirection=ori.GetDirection(),
                                 outputSpacing=[1 / resample_rate, 1 / resample_rate, 1 / resample_rate],
                                 defaultPixelValue=0,
                                 outputPixelType=ori.GetPixelID())
    fixed_image = sitk.Resample(image1=tar,
                                size=get_resample_size(tar.GetSize(), (resample_rate, resample_rate, resample_rate)),
                                transform=sitk.Transform(),
                                interpolator=sitk.sitkLinear, outputOrigin=tar.GetOrigin(),
                                outputDirection=tar.GetDirection(),
                                outputSpacing=[1 / resample_rate, 1 / resample_rate, 1 / resample_rate],
                                defaultPixelValue=0,
                                outputPixelType=tar.GetPixelID())

    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.AffineTransform(3),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()

    registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)

    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.625, minStep=1e-5,
                                                                 numberOfIterations=500, relaxationFactor=0.5)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0.5])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))

    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    return final_transform


def match_nodules(ori, tar, final_transform, spacing, dist_thres=30):
    ori.SetOrigin((0, 0, 0))
    tar.SetOrigin((0, 0, 0))
    ori_spacing = 512 / ori.GetSize()[0]  # 对于正常的512*512的CT，应该为1
    tar_spacing = 512 / tar.GetSize()[0]
    ori.SetSpacing((ori_spacing, ori_spacing, 1))
    tar.SetSpacing((tar_spacing, tar_spacing, 1))
    reg = sitk.Resample(ori, tar, final_transform, sitk.sitkLinear, 0.0, ori.GetPixelID())
    reg_arr = sitk.GetArrayFromImage(reg)
    reg_arr = np.round(reg_arr)
    tar_arr = sitk.GetArrayFromImage(tar)
    ori_arr = sitk.GetArrayFromImage(ori)
    reg_cen = find_centers(reg_arr)
    tar_cen = find_centers(tar_arr)
    ori_cen = find_centers(ori_arr)
    reg_idx, tar_paired = match_nodules_revise(reg_cen, tar_cen, spacing, dist_thres=dist_thres)
    ori_cen_2 = transform_points(reg_cen[:, ::-1] / np.array([tar.GetSize()[0] / 512, tar.GetSize()[0] / 512, 1.]), final_transform)[:, ::-1]
    ori_cen_2 = ori_cen_2 / 512 * np.array([512, ori.GetSize()[0], ori.GetSize()[0]])
    reg_idx_0, tar_paired_0 = match_nodules_revise(ori_cen, ori_cen_2, spacing)
    ori_cen = np.round(ori_cen)
    tar_cen = np.round(tar_cen)
    reg_cen = np.round(reg_cen)
    string = generate_result_string(ori_cen, reg_cen, tar_cen, reg_idx_0, reg_idx, tar_paired)
    dist_arr = get_dist_matrix(reg_cen, tar_cen, spacing)
    print(dist_arr)
    return reg_idx_0, reg_idx, tar_paired, ori_cen, tar_cen
    # matrix = np.array(final_transform.GetParameters()[:9]).reshape([3, 3])
    # trans = np.array(final_transform.GetParameters()[9:]).reshape([3, 1])
    # center = np.array(final_transform.GetFixedParameters()).reshape([3, 1])
    # ori_cen_reg = ori_cen.T
    # ori_cen_reg = np.dot(matrix, (ori_cen_reg - center)) + trans + center
    # ori_cen_reg = ori_cen_reg.T


def pipeline(orii, tari, oril, tarl, dist_thres=30):
    final_transform = get_transform(orii, tari)
    spacing = tuple(reversed(orii.GetSpacing()))
    reg_idx0, reg_idx, tar_paired, ori_cen, tar_cen = match_nodules(oril, tarl, final_transform, spacing, dist_thres=dist_thres)
    return reg_idx0, reg_idx, tar_paired, ori_cen, tar_cen




