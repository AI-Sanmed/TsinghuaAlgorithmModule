import numpy as np
import cv2
import pylidc as pl
import SimpleITK as sitk


def ShuffleImage(image, dimension):
    """
    此函数用于将一个三维的图片数组进行翻转。
    :param image: 三维数组
    :param dimension: 表示翻转方向
    :return: 翻转后的数组
    """
    if dimension == 0:
        return image
    elif dimension == 1:  # [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] -> [[[1, 2], [5, 6]], [[3, 4], [7, 8]]]
        p = []
        for j in range(0, len(image[0])):
            q = []
            for k in range(0, len(image)):
                q.append(image[k][j])
            p.append(q)
        p = np.array(p)
        return p
    elif dimension == 2:  # [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] -> [[[1, 3], [5, 7]], [[2, 4], [6, 8]]]
        p = []
        for i in range(0, len(image[0][0])):
            q = []
            for j in range(0, len(image)):
                r = []
                for k in range(0, len(image[0])):
                    r.append(image[j][k][i])
                q.append(r)
            p.append(q)
        p = np.array(p)
        return p
    else:
        raise ValueError("Variable 'dimension' can only be 0, 1 or 2")


def resample_z(pic, target_z, is_label=False):
    target_z = int(target_z)
    pic_shuffle = ShuffleImage(pic, 1)
    resized_lst = []
    assert pic.shape[1] == pic.shape[2]
    orig_xy = pic.shape[2]
    eps = 1e-5
    for i in pic_shuffle:
        i1 = cv2.resize(i + eps, (orig_xy, target_z))
        resized_lst.append(i1)
    resampled_pic = np.array(resized_lst)
    resampled_pic = ShuffleImage(resampled_pic, 1)
    if is_label:
        resampled_pic[resampled_pic >= 0.5] = 1
        resampled_pic[resampled_pic < 0.5] = 0
    return resampled_pic


def resample_hires(image, label, dist):
    origin = (0, 0, 0)
    factor = dist / 0.6
    crop_len = round(512 / factor)
    final_dist = crop_len * dist / 256
    if crop_len <= 512:
        crop_start = int((512 - crop_len) // 2)
        origin_new = list(origin)
        origin_new[0] = origin[0] + crop_start * dist
        origin_new[1] = origin[1] + crop_start * dist
        image_cropped = image[:, crop_start:round(crop_start + crop_len), crop_start:round(crop_start + crop_len)]
    else:
        pad_start = int((crop_len - 512) // 2)
        origin_new = list(origin)
        origin_new[0] = origin[0] - pad_start * dist
        origin_new[1] = origin[1] - pad_start * dist
        image_cropped = np.pad(image, (
        (0, 0), (pad_start, crop_len - 512 - pad_start), (pad_start, crop_len - 512 - pad_start)), 'constant',
                               constant_values=0)
    image_resized = []
    for i in image_cropped:
        i1 = cv2.resize(i, (512, 512))
        image_resized.append(i1)
    image_cropped = np.array(image_resized)
    if type(label) is np.ndarray:
        if crop_len <= 512:
            crop_start = int((512 - crop_len) // 2)
            label_cropped = label[:, crop_start:round(crop_start + crop_len), crop_start:round(crop_start + crop_len)]
        else:
            pad_start = int((crop_len - 512) // 2)
            label_cropped = np.pad(label, ((0, 0),
                                           (pad_start, crop_len - 512 - pad_start),
                                           (pad_start, crop_len - 512 - pad_start)), 'constant',
                                   constant_values=0)
        label_resized = []
        for i in label_cropped:
            i1 = cv2.resize(i, (512, 512))
            label_resized.append(i1)
        label_cropped = np.array(label_resized)
        label_cropped = np.round(label_cropped)
        label_cropped[label_cropped > 1] = 1
        label_cropped[label_cropped < -1] = -1
        return image_cropped, label_cropped, origin_new, final_dist
    else:
        return image_cropped, origin_new, final_dist


def resample(image, label, dist, final_dist=1.2, final_pixel=256):
    # assert final_pixel == 256 or final_pixel == 512
    origin = (0, 0, 0)
    # factor = dist / (final_dist / (512 / final_pixel))
    image_xy_len = image.shape[2]
    factor = (dist * image_xy_len) / (final_dist * final_pixel)
    # if final_pixel == 256:
    #     factor = dist / (final_dist / 2)
    # elif final_pixel == 512:
    #     factor = dist / final_dist
    crop_len = round(image_xy_len / factor)
    final_dist = crop_len * dist / final_pixel
    if crop_len <= image_xy_len:
        crop_start = int((image_xy_len - crop_len) // 2)
        origin_new = list(origin)
        origin_new[0] = origin[0] + crop_start * dist
        origin_new[1] = origin[1] + crop_start * dist
        image_cropped = image[:, crop_start:round(crop_start + crop_len), crop_start:round(crop_start + crop_len)]
    else:
        pad_start = int((crop_len - image_xy_len) // 2)
        origin_new = list(origin)
        origin_new[0] = origin[0] - pad_start * dist
        origin_new[1] = origin[1] - pad_start * dist
        image_cropped = np.pad(image, (
        (0, 0), (pad_start, crop_len - image_xy_len - pad_start), (pad_start, crop_len - image_xy_len - pad_start)), 'constant',
                               constant_values=-1024)
    image_resized = []
    for i in image_cropped:
        i1 = cv2.resize(i, (final_pixel, final_pixel))
        image_resized.append(i1)
    image_cropped = np.array(image_resized)
    if type(label) is np.ndarray:
        if crop_len <= image_xy_len:
            crop_start = int((image_xy_len - crop_len) // 2)
            label_cropped = label[:, crop_start:round(crop_start + crop_len), crop_start:round(crop_start + crop_len)]
        else:
            pad_start = int((crop_len - image_xy_len) // 2)
            label_cropped = np.pad(label, ((0, 0),
                                           (pad_start, crop_len - image_xy_len - pad_start),
                                           (pad_start, crop_len - image_xy_len - pad_start)), 'constant',
                                   constant_values=-1024)
        label_resized = []
        for i in label_cropped:
            i1 = cv2.resize(np.float32(i), (final_pixel, final_pixel))
            label_resized.append(i1)
        label_cropped = np.array(label_resized)
        label_cropped = np.round(label_cropped)
        label_cropped[label_cropped > 1] = 1
        label_cropped[label_cropped < -1] = -1
        return image_cropped, label_cropped, origin_new, final_dist
    else:
        return image_cropped, origin_new, final_dist


def preprocess_resample(image, label, dist, slice_thickness, final_xy_dist=1.2, final_xy_pixel=256, final_z_dist=1.2):
    # resample to 1.2*1.2*1.2
    image, label, _, _ = resample(image, label, dist, final_xy_dist, final_xy_pixel)
    z_base = len(image)
    improve_factor = slice_thickness / final_z_dist
    z_resample = z_base * improve_factor
    z_resample = int(z_resample)
    image = resample_z(image, z_resample)
    label_cube = resample_z(label, z_resample)
    label = np.round(label_cube)
    return image, label


def resample_back_xy(image, dist, final_dist=1.2, orig_pixel=512):
    factor = final_dist / dist
    final_pixel = image.shape[2]
    back_xy = factor * final_pixel
    resampled_lst = []
    for i in image:
        i = cv2.resize(image, (back_xy, back_xy))
        if back_xy > orig_pixel:
            begin = (back_xy - orig_pixel) // 2
            i = i[begin:begin + orig_pixel, begin:begin + orig_pixel]
        else:
            begin - (orig_pixel - back_xy) // 2
            j = np.zeros(orig_pixel, orig_pixel)
            j[begin:begin + back_xy, begin:begin + back_xy] = i
            i = j
        resampled_lst.append(j)



if __name__ == "__main__":
    a = sitk.ReadImage("CT00124_image.nii.gz")
    xy_dist = a.GetSpacing()[0]
    z_dist = a.GetSpacing()[2]
    a1 = sitk.GetArrayFromImage(a)
    a1 = np.swapaxes(a1, 1, 2)
    label = np.zeros(a1.shape)
    b, bl = preprocess_resample(a1, label, xy_dist, z_dist, final_xy_dist=0.8, final_xy_pixel=384)
    for num, i in enumerate(b):
        i = (i + 1000) / 1500
        cv2.imwrite("./test/{}.png".format(num), i * 255)