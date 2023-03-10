import os
import warnings

import SimpleITK as sitk
import numpy as np
import torch
from scipy import ndimage
from scipy import ndimage as ndi
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure, binary_opening
from skimage import measure
from skimage.morphology import remove_small_objects


def load_itk_image(filename):
    """
    :param filename: CT name to be loaded
    :return: CT image, CT origin, CT spacing in z, y, x
    """
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing


def load_model(model, model_file):
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model


def separate_to_batches(array, batch_size):
    total_num = array.shape[0]
    num_batch = total_num // batch_size
    batch_lst = []
    for num in range(0, num_batch):
        batch_lst.append(array[num * batch_size:num * batch_size + batch_size])
    if total_num % batch_size != 0:
        batch_lst.append(array[num_batch * batch_size:])
    return batch_lst


class CT_info():
    def __init__(self, nii_file):
        self.CT_id = None
        self.img_itk = None
        self.img_arr = None
        self.origin = None
        self.spacing = None

        self.load_nii(nii_file)

        self.lung_mask = None
        self.lobe_arr = None
        self.airway_arr = None
        self.segment_arr = None  # segment segmentation
        self.nodule_stage1_arr = None  # detection result
        self.nodule_stage2_arr = None  # fpred result
        self.nodule_stage3_arr = None  # precise seg result

        self.nodules = {}

    def load_nii(self, nii_file):
        arr, origin, spacing = load_itk_image(nii_file)
        img = sitk.ReadImage(nii_file)
        self.img_arr = arr
        self.origin = origin
        self.spacing = spacing
        self.img_itk = img
        self.CT_id = os.path.basename(nii_file).replace('.nii.gz', '')

    def write_nii(self, array, nii_file):
        origin = self.img_itk.GetOrigin()
        spacing = self.img_itk.GetSpacing()
        itk = sitk.GetImageFromArray(array)
        itk.SetOrigin(origin)
        itk.SetSpacing(spacing)
        sitk.WriteImage(itk, nii_file)


class Nodule_info:
    def __init__(self):
        self.name = None
        self.label = None
        self.centerX = None
        self.centerY = None
        self.centerZ = None
        self.DimX = None
        self.DimY = None
        self.DimZ = None
        self.lungPart = None
        self.lunglobe = None
        self.aveDiameter2D = None
        self.largestSliceZCoordinate = None
        self.majorDiameter2D = None
        self.minorDiameter2D = None
        self.SegmentationCenterX = None
        self.SegmentationCenterY = None
        self.SegmentationCenterZ = None
        self.Radius = None
        self.EllipsoidRadius0 = None
        self.EllipsoidRadius1 = None
        self.EllipsoidRadius2 = None
        self.volume = None
        self.huAve = None
        self.huStd = None
        self.HUAvgAirInLung = None
        self.HUStdDevAirInLung = None
        self.huMin = None
        self.huMax = None
        self.noduleType = None
        self.calcPct = None
        self.IsCalcNodule = None
        self.OrigDetMaligScore = None

        self.box = None  # [z_min, y_min, x_min, z_max, y_max, x_max]

        self.xBegin = None
        self.yBegin = None
        self.zBegin = None
        self.xEnd = None
        self.yEnd = None
        self.zEnd = None

        self.minDistToWall = None
        self.minDistToWallInd = None  # z, y, x


def _get_cube_from_volume(volume, center, side_len=40):  # ?????????????????????pad??????????????????????????????
    assert side_len % 2 == 0
    sh = volume.shape  # volume: z, y, x, center: x, y, z
    bi = side_len / 2
    bi = int(bi)
    x_left_pad = -min(0, center[0] - bi)
    x_right_pad = max(sh[2], center[0] + bi) - sh[2]
    y_left_pad = -min(0, center[1] - bi)
    y_right_pad = max(sh[1], center[1] + bi) - sh[1]
    z_left_pad = -min(0, center[2] - bi)
    z_right_pad = max(sh[0], center[2] + bi) - sh[0]
    cube = volume[max(center[2] - bi, 0):center[2] + bi, max(center[1] - bi, 0):center[1] + bi,
           max(center[0] - bi, 0):center[0] + bi]
    cube = np.pad(cube, ((z_left_pad, z_right_pad), (y_left_pad, y_right_pad), (x_left_pad, x_right_pad)))
    # if cube.shape[0] != 40 or cube.shape[1] != 40 or cube.shape[2] != 40:
    #     print("Volume shape: ")
    #     print(volume.shape)
    #     print("Center coord: ")
    #     print(center)
    #     print("Final shape: ")
    #     print(cube.shape)
    assert cube.shape == (side_len, side_len, side_len)
    return cube


def find_diam(array):
    # the number of connected areas in array must be 1 or 0.
    a1 = np.max(array, axis=(1, 2))
    a2 = np.max(array, axis=(0, 2))
    a3 = np.max(array, axis=(0, 1))
    a1 = np.sum(a1)
    a2 = np.sum(a2)
    a3 = np.sum(a3)
    return (a1 + a2 + a3) / 3


def find_center(array):
    # the number of connected areas in array must be 1
    # sequence of coords in return is the same of that in array
    a1 = np.max(array, axis=(1, 2))
    a2 = np.max(array, axis=(0, 2))
    a3 = np.max(array, axis=(0, 1))
    c1 = np.mean(np.where(a1 == 1)[0])
    c2 = np.mean(np.where(a2 == 1)[0])
    c3 = np.mean(np.where(a3 == 1)[0])
    c1 = round(c1)
    c2 = round(c2)
    c3 = round(c3)
    return c1, c2, c3


def safe_input(cube, point, disk):
    # put the disk into cube at point
    # avoids long time cost of binary dilation
    # avoids out of bounds
    # disk shape must be odd, disk must be cube
    d = int((disk.shape[0] - 1) / 2)
    d1 = disk.shape[0]
    d2 = int((disk.shape[1] - 1) / 2)
    d12 = disk.shape[1]
    d3 = int((disk.shape[2] - 1) / 2)
    d13 = disk.shape[2]
    biass = [0, 0, 0, 0, 0, 0]  # bias for db0, ub0, ..., ub2
    db0 = point[0] - d
    ub0 = point[0] - d + d1
    db1 = point[1] - d2
    ub1 = point[1] - d2 + d12
    db2 = point[2] - d3
    ub2 = point[2] - d3 + d13
    bounds = [db0, ub0, db1, ub1, db2, ub2]
    for idx in range(len(bounds)):
        if bounds[idx] < 0:
            biass[idx] = -bounds[idx]
            bounds[idx] = 0
        if bounds[idx] > cube.shape[idx // 2]:
            biass[idx] = cube.shape[idx // 2] - bounds[idx]
            bounds[idx] = cube.shape[idx // 2]
    cube[bounds[0]:bounds[1], bounds[2]:bounds[3], bounds[4]:bounds[5]] = disk[biass[0]:d1 + biass[1],
                                                                          biass[2]:d12 + biass[3],
                                                                          biass[4]:d13 + biass[5]]
    return cube


def compute_lung_extendbox(mask, margin=(10, 10, 10)):
    zz, yy, xx = np.where(mask)
    min_box = np.max([[0, zz.min() - margin[0]],
                      [0, yy.min() - margin[1]],
                      [0, xx.min() - margin[2]]], axis=1, keepdims=True)

    max_box = np.min([[mask.shape[0], zz.max() + margin[0]],
                      [mask.shape[1], yy.max() + margin[1]],
                      [mask.shape[2], xx.max() + margin[2]]], axis=1, keepdims=True)

    box = np.concatenate([min_box, max_box], axis=1)

    return box


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

######################
def get_lung_mask(ct_nodule):
    img_arr = ct_nodule.img_arr
    lung_mask = segment_lung_mask(img_arr)
    ct_nodule.lung_mask = lung_mask


def is_backgroud(label_region, origin_mask, k_num=5, margin=2):
    zz, yy, xx = np.where(label_region)
    z_length = zz.max() - zz.min() + 1
    y_length = yy.max() - yy.min() + 1
    x_length = xx.max() - xx.min() + 1
    # z_length, y_length, x_length = label_region.shape
    # ??????????????????5???????????????????????????????????????????????????????????????
    start_index = z_length / 2 - (k_num / 2) * margin
    end_index = z_length / 2 + (k_num / 2) * margin + 1  # range?????????????????????
    # ???????????????????????????????????????????????????????????????slice??????????????????????????????????????????????????????
    if start_index < 0 or end_index > z_length:
        start_index = z_length / 2
        end_index = z_length / 2 + 1  # range?????????????????????
        margin = 1
    selected_slice_index = list(range(int(start_index), int(end_index), margin))
    selected_slice_index_is_background = [False] * k_num
    for idx, index in enumerate(selected_slice_index):
        # ??????????????????????????????????????????????????????????????????????????????????????????
        y_region, x_region = np.where(label_region[index])
        # ???????????????????????????????????????????????????????????????????????????????????????
        if len(y_region) and len(x_region):
            y_region_length = y_region.max() - y_region.min() + 1
            x_region_length = x_region.max() - x_region.min() + 1
        else:
            continue

        filled_region = ndi.binary_fill_holes(label_region[index])
        vessel_like = np.logical_xor(filled_region, origin_mask[index])
        # vessel_like = np.logical_xor(filled_region, label_region[index])
        y_vessel, x_vessel = np.where(vessel_like)
        # ???????????????????????????????????????????????????????????????????????????????????????
        if len(y_vessel) and len(x_vessel):
            y_vessel_length = y_vessel.max() - y_vessel.min() + 1
            x_vessel_length = x_vessel.max() - x_vessel.min() + 1
        else:
            selected_slice_index_is_background[idx] = True
            continue

        label = measure.label(vessel_like)
        # ???????????????????????????????????????????????????3???????????????????????????????????????3??????????????????????????????hard code
        if label.max() < 3:
            selected_slice_index_is_background[idx] = True
            continue

        # ??????????????????????????????????????????
        if y_vessel.min() > 0.5 * (y_region.min() + y_region.max()) and y_vessel_length < 0.33 * y_region_length:
            selected_slice_index_is_background[idx] = True
            continue

    # ???????????????????????????????????????slice????????????????????????True
    if sum(selected_slice_index_is_background) > k_num / 2:
        return True
    return False


def is_local_lung(chest_mask, k_num=5, margin=2, min_area=2000):
    """
    :brief ??????????????????????????????????????????????????????
    :param[in] chest_mask: np.ndarray, the mask of chest
    :param[in] k_num: int, uneven number, like the k in KNN, the number of sampled slice for judge
    :param[in] margin: int, the sample step for the middle slice
    :param[in] min_area: int, about the area of trachea
    :return: bool, true means the dcm_img is local lung.
    """
    z_length, y_length, x_length = chest_mask.shape
    # ??????????????????5???????????????????????????????????????????????????????????????
    start_index = z_length / 2 - (k_num / 2) * margin
    end_index = z_length / 2 + (k_num / 2) * margin + 1  # range?????????????????????
    # ???????????????????????????????????????????????????????????????slice??????????????????????????????????????????????????????
    if start_index < 0 or end_index > z_length:
        start_index = z_length / 2
        end_index = z_length / 2 + 1  # range?????????????????????
        margin = 1

    selected_slice_index = list(range(start_index, end_index, margin))
    # ????????????????????????
    selected_slice_index_is_local = [False] * k_num
    for idx, index in enumerate(selected_slice_index):
        # ??????????????????????????????????????????????????????????????????????????????????????????
        chest_mask[index] = binary_opening(chest_mask[index], iterations=2)
        label_image = measure.label(chest_mask[index])
        regions_image = measure.regionprops(label_image)
        max_area, seq = 0, 0
        for region in regions_image:
            if region.area > max_area:
                max_area = region.area
                seq = region.label
        # ???????????????????????????????????????????????????????????????????????????
        # mask[iz] = label_image == seq
        chest_mask[index] = np.in1d(label_image, [seq]).reshape(label_image.shape)

        filled_chest = ndi.binary_fill_holes(chest_mask[index])
        # ??????????????????????????????????????????????????????5 * min_area????????????????????????????????????5?????????????????????
        max_bronchi_area = 5 * min_area
        if np.sum(filled_chest) - np.sum(chest_mask[index]) < max_bronchi_area:
            selected_slice_index_is_local[idx] = True
            continue
        else:
            label = measure.label(filled_chest ^ chest_mask[index])
            vals, counts = np.unique(label, return_counts=True)
            counts = sorted(counts[vals != 0].tolist(), reverse=True)
            # ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
            if label.max() >= 2 and counts[0] > max_bronchi_area and counts[1] > max_bronchi_area:
                continue

        # ????????????????????????????????????????????????????????????case???????????????????????????????????????????????????????????????????????????
        # ??????case??????????????????????????????????????????????????????????????????????????????????????????
        yy, xx = np.where(filled_chest)
        border = 10  # 10???????????????????????????
        if (yy.min() - 0 < border or y_length - 1 - yy.max() < border) and (
                xx.min() - 0 < border or x_length - 1 - xx.max() < border):
            selected_slice_index_is_local[idx] = True
            continue

    # ???????????????????????????????????????slice????????????????????????True
    if sum(selected_slice_index_is_local) > k_num / 2:
        return True
    return False


def largest_one_or_two_label_volume(mask, origin_mask, bg=-1, is_half=False, is_local=False, is_circle=False):
    """
    :brief ????????????????????????????????????????????????????????????
    :param[in] mask: ndarray, ??????????????????labels, ?????????measure.label??????
    :param[in] bg: int, ?????????????????????-1
    :return: ?????????????????????mask??????????????????
    """
    ret_mask = np.zeros_like(mask)
    labels = measure.label(mask, background=0)
    vals, counts = np.unique(labels, return_counts=True)
    counts = counts[vals != bg].astype('float')
    vals = vals[vals != bg]

    # ??????????????????????????????????????????????????????????????????????????????
    # if not PREPROCESS_USE_SLICE_LOCATOR:
    is_half = True

    order = np.argsort(counts)[::-1].tolist()  # ??????????????????????????????

    # ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    minimal_lung_region = 10000  # ???????????????5??????????????????????????????????????????????????????????????????????????????????????????
    while len(order) >= 2 and (is_local or is_circle):
        max_region = labels == vals[order[0]]
        # ??????????????????????????????????????????
        if np.sum(max_region) > minimal_lung_region and is_backgroud(max_region, origin_mask):
            order.pop(0)
        else:
            break

    # ?????????????????????????????????????????????????????????????????????????????????
    if len(order) == 1 or not is_half:
        ret_mask = labels == vals[order[0]]
        return ret_mask.astype('uint8'), 1
    elif len(order) >= 2:
        for i in range(1, len(order)):
            if counts[order[i]] < minimal_lung_region:
                ret_mask = labels == vals[order[0]]
                return ret_mask.astype('uint8'), 1
            # ?????????????????????????????????5???
            if not is_local and counts[order[0]] / counts[order[i]] > 5.0:
                continue
            # ???????????????10??????????????????????????????????????????????????????????????????
            if is_local and counts[order[0]] / counts[order[i]] > 5.0:
                continue
            else:
                # ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????Z????????????????????????
                # ??????????????????????????????????????????????????????5???????????????5?????????????????????????????????????????????????????????????????????????????????????????????
                # ????????????????????????????????????Z????????????????????????????????????????????????????????????
                # ????????????????????????????????????i == 1, ???????????????????????????????????????????????????????????????????????????????????????case?????????????????????
                if i == 1:
                    z_first, y_first, x_first = np.where(labels == vals[order[0]])
                    z_first_start, z_first_end = z_first.min(), z_first.max()
                    z_first_length = z_first_end - z_first_start + 1

                    y_first_start, y_first_end = y_first.min(), y_first.max()
                    y_first_length = y_first_end - y_first_start + 1

                    x_first_start, x_first_end = x_first.min(), x_first.max()
                    x_first_length = x_first_end - x_first_start + 1

                z_second, y_second, x_second = np.where(labels == vals[order[i]])
                z_second_start, z_second_end = z_second.min(), z_second.max()
                z_second_length = z_second_end - z_second_start + 1

                y_second_start, y_second_end = y_second.min(), y_second.max()
                y_second_length = y_second_end - y_second_start + 1

                x_second_start, x_second_end = x_second.min(), x_second.max()
                x_second_length = x_second_end - x_second_start + 1

                # ??????Z????????????????????????????????????????????????3?????????pass???????????????
                if z_second_start > z_first_end or z_second_end < z_first_start or \
                        float(z_first_length) / z_second_length > 3.0 or float(z_second_length) / z_first_length > 3.0:
                    if i == len(order) - 1:
                        ret_mask = labels == vals[order[0]]
                        return ret_mask.astype('uint8'), 1
                    continue

                # ??????Y????????????????????????????????????????????????3??????????????????????????????????????????1???1?????????pass???????????????
                if y_second_start > y_first_end or y_second_end < y_first_start or \
                        float(y_first_length) / y_second_length > 3.0 or float(y_second_length) / y_first_length > 3.0:
                    if i == len(order) - 1:
                        ret_mask = labels == vals[order[0]]
                        return ret_mask.astype('uint8'), 1
                    continue

                # ????????????????????????????????????????????????X??????????????????????????????????????????????????????3???/??????????????????????????????????????????pass????????????
                if float(x_first_length) / x_second_length > 3.0 or float(x_second_length) / x_first_length > 3.0 or \
                        x_first_length + x_second_length > labels.shape[2] or (
                        x_second_start > x_first_start and x_second_end < x_first_end):
                    if i == len(order) - 1:
                        ret_mask = labels == vals[order[0]]
                        return ret_mask.astype('uint8'), 1
                    continue

                ret_mask = np.logical_or((labels == vals[order[0]]), (labels == vals[order[i]])).astype('uint8')
                return ret_mask, 2
    else:
        return mask, 0


def postprocess_mask(mask, iterations=5):
    """
    :brief ????????????????????????mask
    :param[in] mask: 3d array, ???mask???shape: [z, y, x]
    :param[in] iterations: int, ????????????
    :return: ???????????????mask
    """
    struct = generate_binary_structure(3, 1)
    dilatedMask = binary_dilation(mask, structure=struct, iterations=iterations)
    return dilatedMask


def fill_2d_hole(mask):
    """
    :brief ???3D?????????????????????2D????????????mask????????????
    :param[in] mask: 3d array, ?????????mask, shape: [z, y, x]
    :return: ??????????????????????????????mask, uint8
    """
    for i in range(mask.shape[0]):
        label = measure.label(mask[i])
        properties = measure.regionprops(label)
        for prop in properties:
            bb = prop.bbox
            mask[i][bb[0]:bb[2], bb[1]:bb[3]] = mask[i][bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image

    return mask.astype('uint8')


def find_chest(mask, is_local=False):
    """
    :brief ??????????????????
    :param[in&out] mask: 3d array, ?????????????????????????????????????????????mask
    :return: ?????????????????????mask, bool
    """
    # ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????case??????????????????????????????????????????
    if is_local:
        struct = generate_binary_structure(3, 1)
        struct[0, :] = 0  # Z?????????????????????
        struct[2, :] = 0  # Z?????????????????????
        mask = binary_opening(mask, structure=struct, iterations=2)
        init_mask = mask.copy()
        # ????????????????????????????????????????????????????????????????????????????????????????????????????????????
        # ???????????????????????????????????????????????????????????????
        mask[:, 0:1, :] = True
        mask[:, -1:, :] = True
        mask[:, :, 0:1] = True
        mask[:, :, -1:] = True

        # mask[:, 0:1, 2:-2] = True
        # mask[:, -1:, 2:-2] = True
        # mask[:, 2:-2, 0:1] = True
        # mask[:, 2:-2, -1:] = True

    for iz in range(mask.shape[0]):
        mask[iz] = ndi.binary_fill_holes(mask[iz])  # fill body
        label_image = measure.label(mask[iz])
        regions_image = measure.regionprops(label_image)
        max_area = 0
        seq = 0
        for region in regions_image:
            if region.area > max_area:
                max_area = region.area
                seq = region.label
        # ???????????????????????????????????????????????????????????????????????????
        # mask[iz] = label_image == seq
        mask[iz] = np.in1d(label_image, [seq]).reshape(label_image.shape)

    return mask


def segment_lung_mask(ct_series, chest_th=-500, lung_th=-320, min_object_size=10, max_areas=250000, min_areas=2000):
    """
    :brief ?????????dcm?????????????????????????????????????????????????????????
    :param[in] dcm_img: 3d array, ????????????CT??????, shape: [z, y, x]
    :param[in] chest_th: int, ????????????, ?????????-300
    :param[in] lung_th: int, ????????????????????????-320
    :param[in] min_object_size: int, ????????????????????????????????????300
    :param[in] max_areas: int, ??????????????????????????????250000
    :param[in] min_areas: ??????????????????????????????4000
    :return: ????????????????????????????????????mask???3d array, uint8, shape: [z, y, x]
    """
    dcm_img = ct_series
    # ?????????????????????512??????????????????size??????????????????512???????????????????????????????????????????????????????????????????????????????????????????????????hardcode
    if dcm_img.shape[1] != 512 or dcm_img.shape[2] != 512:
        rate = (float(dcm_img.shape[1]) / 512) * (float(dcm_img.shape[2]) / 512)
        min_areas = min_areas * rate
        max_areas = max_areas * rate

    # CT????????????-32767 ~ 32767?????????case???????????????????????????????????????lung_th?????????????????????????????????????????????
    if dcm_img.max() > 20000:
        lung_th = -500  # ???????????????????????????????????????????????????????????????

    # ????????????????????????????????????????????????slice?????????????????????
    # ?????????????????????????????????lung_slice_range????????????None
    ret_mask = np.zeros_like(dcm_img, dtype='uint8')
    # if PREPROCESS_USE_SLICE_LOCATOR:
    #     z_start, z_end = ct_series.lung_slice_range_label[0][0], ct_series.lung_slice_range_label[0][1] + 1
    #     dcm_img = dcm_img[z_start: z_end]

    # chest segmentation????????????????????????????????????????????????????????????-1000???????????????-600+???????????????OK
    mask = dcm_img > chest_th

    # ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    is_local = False
    # ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

    # ????????????????????????????????????????????????????????????3??????min_areas????????????????????????????????????
    if is_local:
        min_areas = 3 * min_areas

    mask = find_chest(mask, is_local=is_local)

    # ??????????????????????????????????????????????????????????????????????????????dcm??????PixelPaddingValue??????????????????????????????????????????CT???????????????????????????
    # ??????CT?????????2000 ??? 3000????????????????????????5000?????????resize????????????????????????????????????resize???????????????CT??????????????????????????????
    # ????????????????????????????????????????????????CT?????????????????????resize???????????????????????????????????????500???1500?????????????????????5000????????????????????????
    is_circle = False
    ct_interval = dcm_img.max() - dcm_img.min()
    if 10000 > dcm_img.max() and ct_interval > 5000:
        is_circle = True
        th = dcm_img.min()
        circle_mask = dcm_img > th
        mask = np.logical_and(mask, circle_mask)

        # mid_slice = circle_mask[circle_mask.shape[0] / 2]
        # erosion_circle = binary_erosion(mid_slice, iterations=2)
        # circle_contour = np.logical_xor(mid_slice, erosion_circle)  # ?????????????????????0???????????????mask

    temp_x = mask * dcm_img

    # lung segmentation
    fine_mask = temp_x < lung_th
    # ???????????????????????????????????????????????????????????????3D??????????????????case????????????????????????
    # ????????????????????????????????????????????????????????????????????????????????????200slices/3s
    fine_mask = ndimage.median_filter(fine_mask.astype('uint8'), size=(1, 3, 3))
    fine_mask, num_regions = largest_one_or_two_label_volume(fine_mask, temp_x < lung_th, bg=0, is_local=is_local,
                                                             is_circle=is_circle)

    for i in range(fine_mask.shape[0]):
        # ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
            # int????????????????????????????????????????????????????????????????????????mask?????????1????????????????????????
            fine_mask[i] = remove_small_objects(fine_mask[i].astype('bool'), min_size=10)
        slice_label = measure.label(fine_mask[i], background=0)
        properties = measure.regionprops(slice_label)
        areas = np.array([[p.area, p.bbox_area, p.centroid[1], p.centroid[0]] for p in properties])
        max_label = np.max(slice_label)
        if max_label != 0 and np.max(areas[:, 0]) > max_areas:
            fine_mask[i] = 0

        # ????????????????????????????????????????????????????????????????????????????????????????????????????????????
        # ????????????/bbox?????????????????????????????????????????????0.1 * bbox??????????????????????????????????????????????????????????????????0.3?????????????????????
        if max_label == 1 and areas[0, 0] < min_areas and areas[0, 0] > 0.3 * areas[0, 1]:
            fine_mask[i] = 0

        # ???????????????????????????????????????????????????????????????????????????????????????(?????????500??????)????????????????????????????????????
        # ???????????????????????????????????????????????????????????????, ??????/????????????????????????(??????????????????-????????????)??????40???80?????????????????????20?????????????????????
        if max_label > 1 and (areas[0, 0] < min_areas and (np.std(areas[:, 2]) < 20 and np.std(areas[:, 3]) < 20)):
            fine_mask[i] = 0

    # fill the hole
    # TODO: ???????????????????????????????????????????????????10x10x10?????????????????????????????????
    fine_mask = fill_2d_hole(fine_mask)
    # if PREPROCESS_USE_SLICE_LOCATOR:
    #     ret_mask[z_start: z_end] = fine_mask
    # else:
    #     ret_mask = fine_mask
    ret_mask = fine_mask

    return ret_mask


if __name__ == "__main__":
    a = np.arange(1000000).reshape((100, 100, 100))
    b = _get_cube_from_volume(a, (47, 46, 45), 32)
    c = np.zeros((100, 100, 100))
    c = safe_input(c, (44, 45, 46), b)
    c
