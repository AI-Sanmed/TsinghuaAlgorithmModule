import numpy as np
from skimage import morphology, measure
from scipy.ndimage import zoom, label
import nibabel
from scipy import ndimage
import SimpleITK as sitk
import time
import segment_segmentation.rough_segmentation as r


def check_results(vol, min_num, max_num):
    for i in range(min_num, max_num + 1):
        if np.sum(vol == i) < 5000:
            print("Result check failed when i = {}".format(i))
            raise RuntimeError()


class Point:
    def __init__(self, coords, parent, children, level):
        self.coords = coords
        self.parent = parent  # point class of parent
        self.children = children  # list of coord of children
        self.unvisited_children = children.copy()
        self.level = level
        self.max_dist_to_leaf = 0  # only >1 children points have valid dist
        self.min_dist_to_leaf = 999999999  # only >1 children points have valid dist
        self.parent_node = None
        self.children_node = []  # list of Point classes of children node
        self.unvisited_children_node = []
        self.new_level = -1
        self.spike_label = False  # True indicates that this point may be a spike (maoci in Chinese)


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
    cube[bounds[0]:bounds[1], bounds[2]:bounds[3], bounds[4]:bounds[5]] = disk[biass[0]:d1 + biass[1], biass[2]:d12 + biass[3], biass[4]:d13 + biass[5]]
    return cube


def fast_centers_mass(skel, skel_l, conn, ori_lobe_mask):
    skel, slc = crop_mask_area(skel, ori_lobe_mask)
    skel_l, _ = crop_mask_area(skel_l, ori_lobe_mask)
    lo_coord = np.array([[slc[0].start, slc[1].start, slc[2].start]])
    centers_mass = ndimage.center_of_mass(skel, skel_l, conn)
    centers_mass = np.array(centers_mass)
    centers_mass += lo_coord
    return centers_mass



def get_largest_conn(cube, num=1):
    """
    此函数取输入二值三维数组的最大连通域，将不是最大连通域的值均变为0。
    """
    if np.sum(cube) == 0:
        return cube
    result = measure.label(cube)
    result1 = result.reshape([-1])
    lst = np.bincount(result1)
    lst[0] = 0
    if num == 1:
        a = np.argmax(lst)
        result[result != a] = 0
        result[result == a] = 1
    elif num > 1:
        if len(lst) <= num:
            print("Warning: num of connective areas less than num wanted.")
        a = np.argsort(lst)
        a = a[::-1]
        a = a[:num]
        for i in a:
            result[result == i] = -1
        result[result > 0] = 0
        result *= -1
    return result


def list_indexing(cube, index):
    return cube[index[0]][index[1]][index[2]]


def find_children(skel, skel_c, coord, parent):
    """
    find children around a certain coord.
    :param skel: skeleton graph
    :param skel_c: skeleton graph where children that have been added in list have 0 value
    :param coord: certain coord
    :param parent: parent of certain coord
    :return: list of children
    """
    skel1 = skel.copy()
    skel1[coord[0]][coord[1]][coord[2]] = 2
    cube = skel1[max(0, coord[0] - 1):min(skel.shape[0], coord[0] + 2), max(0 ,coord[1] - 1):min(skel.shape[1], coord[1] + 2), max(0, coord[2] - 1):min(skel.shape[2], coord[2] + 2)]
    idxs = np.where(cube == 1)
    idx0 = np.where(cube == 2)
    ch_lst = []
    for i in range(len(idxs[0])):
        c = [idxs[0][i], idxs[1][i], idxs[2][i]]
        c[0] = c[0] + coord[0] - idx0[0][0]
        c[1] = c[1] + coord[1] - idx0[1][0]
        c[2] = c[2] + coord[2] - idx0[2][0]
        if c != parent and list_indexing(skel_c, c):
            ch_lst.append(c)
    return ch_lst


def make_tree(skel):
    skel_visited = skel.copy()  # 0: visited point
    skel_children = skel.copy()  # 0: point that has been added in a children list
    current_level = 0
    i = 1
    while True:
        root_p = np.where(skel[-i] == 1)
        if len(root_p[0]):
            break
        i += 1
    root_coord = [skel.shape[0] - i, root_p[0][0], root_p[1][0]]
    ch_lst = find_children(skel, skel_children, root_coord, None)
    all_points = []
    all_node_points = []
    all_leaf_points = []
    root_point = Point(root_coord, None, ch_lst, current_level)
    skel_visited[root_coord[0]][root_coord[1]][root_coord[2]] = 0
    skel_children[root_coord[0]][root_coord[1]][root_coord[2]] = 0
    for idx in ch_lst:
        skel_children[idx[0]][idx[1]][idx[2]] = 0
    current_point = root_point
    all_points.append(current_point)
    while True:
        break_flag = 0
        if len(current_point.unvisited_children) == 0:  # need to find back
            this_node = current_point
            current_max_dist_to_leaf = current_point.max_dist_to_leaf
            current_min_dist_to_leaf = current_point.min_dist_to_leaf
            while True:
                current_point = current_point.parent
                if current_point is None:
                    break_flag = 1
                    break
                if len(current_point.children) > 1 and this_node.parent_node is None:
                    this_node.parent_node = current_point
                    current_point.children_node.append(this_node)
                    current_point.unvisited_children_node.append(this_node)
                    this_node = current_point
                current_max_dist_to_leaf += 1
                current_min_dist_to_leaf += 1
                if not len(current_point.unvisited_children):
                    if current_max_dist_to_leaf > current_point.max_dist_to_leaf:
                        current_point.max_dist_to_leaf = current_max_dist_to_leaf
                    if current_min_dist_to_leaf < current_point.min_dist_to_leaf:
                        current_point.min_dist_to_leaf = current_min_dist_to_leaf
                else:
                    if current_max_dist_to_leaf > current_point.max_dist_to_leaf:
                        current_point.max_dist_to_leaf = current_max_dist_to_leaf
                    if current_min_dist_to_leaf < current_point.min_dist_to_leaf:
                        current_point.min_dist_to_leaf = current_min_dist_to_leaf
                    current_level = current_point.level
                    break
        if break_flag == 1:
            break
        next_coord = current_point.unvisited_children.pop()
        if list_indexing(skel_visited, next_coord) == 0:
            continue
        ch_lst = find_children(skel, skel_children, next_coord, current_point.coords)
        for idx in ch_lst:
            skel_children[idx[0]][idx[1]][idx[2]] = 0
        next_point = Point(next_coord, current_point, ch_lst, current_level)
        if len(ch_lst) == 0:
            next_point.min_dist_to_leaf = 0
            all_leaf_points.append(next_point)
        all_points.append(next_point)
        if len(ch_lst) > 1:
            next_point.level += 1
            all_node_points.append(next_point)
            current_level += 1
        if len(all_points) % 20 == 0:
            print("Finished visiting nodes: {} of {}".format(len(all_points), np.sum(skel)))
        skel_visited[next_coord[0]][next_coord[1]][next_coord[2]] = 0
        current_point = next_point
    return all_points, all_node_points, all_leaf_points


def get_new_level(all_node_points):
    all_node_points_new = []
    root = all_node_points[0]
    all_node_points_new.append(root)
    assert root.parent_node is None
    current_node = root
    current_level_diff = 0
    if root.min_dist_to_leaf < 5:
        current_level_diff += 1
    root.new_level = root.level - current_level_diff
    while True:
        if len(current_node.unvisited_children_node):
            i = current_node.unvisited_children_node.pop()
            if 0 < i.min_dist_to_leaf < 3:  # may be a node with a spike branch
                current_level_diff += 1
                i.spike_label = True
            i.new_level = i.level - current_level_diff
            if i.max_dist_to_leaf != 0:
                all_node_points_new.append(i)
            current_node = i
        else:
            if current_node.parent_node is None:
                break
            else:
                current_node = current_node.parent_node
                assert current_node.new_level >= 0
                current_level_diff = current_node.level - current_node.new_level
    return all_node_points_new


def load_volume(path):
    # vol = nibabel.load(path)
    # vol = np.array(vol.dataobj).T
    itkimage = sitk.ReadImage(path)
    vol = sitk.GetArrayFromImage(itkimage)
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    # if vol.shape[2] % 2 == 1:
    #     vol = vol[:, :, :-1]
    return vol, spacing


def erase(cube, coord, radius=1):
    cube[max(0, coord[0] - radius):min(cube.shape[0], coord[0] + radius + 1), max(0, coord[1] - radius):min(cube.shape[1], coord[1] + radius + 1), max(0, coord[2] - radius):min(cube.shape[2], coord[2] + radius + 1)] = 0
    return cube


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


def check_dist(point_lst, point, thres=2):
    flag = 0
    for i in point_lst:
        if np.sum(np.square(np.array(i) - np.array(point))) < thres ** 2:
            flag = 1
    return flag


def check_z(point_lst, point):
    flag = 0
    for i in point_lst:
        if i[0] < point[0]:
            flag = 1
    return flag


def z_cos(coord1, coord2):  # z-y-x coord
    coord1 = np.array(coord1)
    coord2 = np.array(coord2)
    diff = coord2 - coord1
    diff = diff / np.sqrt(np.sum(diff ** 2))
    return -diff[0]


def min_z_cos(coord1, coord2):
    lst = []
    for j in coord2:
        lst.append(z_cos(coord1, j))
    return np.min(lst)


def keep_conn_with_leaf(branch_array, all_leaf, min_level=0):
    """
    Only keep connective regions with leaf. Should be used after erasing node points and before getting largest conns.
    :param branch_array: array after erasing node points
    :param all_leaf: list containing point classes of leaf points
    :return: array with conns with leaf points kept
    """
    all_leaf_points = []
    all_leaf_points_levels = []
    for l in all_leaf:
        all_leaf_points.append(l.coords)
        all_leaf_points_levels.append(l.new_level)
    arr1 = measure.label(branch_array)
    num_conn_with_leaf = []
    final_array = np.zeros(branch_array.shape)
    for num, c in enumerate(all_leaf_points):
        if list_indexing(arr1, c) != 0 and list_indexing(arr1, c) not in num_conn_with_leaf and all_leaf_points_levels[num] >= min_level:
            num_conn_with_leaf.append(list_indexing(arr1, c))
    for i in num_conn_with_leaf:
        arr0 = (arr1 == i)
        final_array += arr0
    return final_array


def generate_ball_mask(volume, center, radius=7):
    """
    generates a ball mask to limit the length of qiguan
    :param volume: a cube with shape of main CT volume (z, y, x)
    :param center: ndarray of center coord n*(z, y, x)
    :param radius: ball radius
    :return: mask in the volume
    """
    b = morphology.ball(radius)
    a = np.zeros(volume.shape)
    center = np.int32(np.round(center))
    for i in center:
        x = np.zeros(volume.shape)
        x = safe_input(x, i, b)
        a += x
        a[a > 1] = 1
    return a


def calc_dist(lst1, lst2, spacing):
    lst1 = np.array(lst1)
    lst2 = np.array(lst2)
    return np.sqrt(np.sum(((lst1 - lst2) * spacing) ** 2))


def seg_rul(all_node_points, vol_lobe, skel, all_leaf_points, ori_lobe_area, ball_mask=0, show_center_mass=False):
    print("Segmenting RUL...")
    ori_lobe_area = (ori_lobe_area == 1)  # lobe area before downsampling
    ori_shape = ori_lobe_area.shape
    lobe_area = (vol_lobe == 1)
    lobe_area_ori = lobe_area.copy()
    point_lst = []
    point_coord_lst = []
    point_children_lst = []
    for _ in range(2):
        lobe_area = morphology.binary_dilation(lobe_area, np.ones((3, 3, 3)))
    for i in all_node_points:
        if list_indexing(lobe_area, i.coords) and i.spike_label is False:
            lst1 = [i.level, i.min_dist_to_leaf, i.max_dist_to_leaf, i.new_level]
            point_lst.append(lst1)
            point_coord_lst.append(i.coords)
            point_children_lst.append(i.children)
    point_lst = np.array(point_lst)
    # idxs = np.argsort(point_lst[:, 3])  # uses new dist here
    node_coords = []
    # node_levels = []
    # found_node_count = -1  # first
    # currernt_level = point_lst[idxs[0]][0] - 1
    # current_max_dist = point_lst[idxs[0]][2]
    # current_min_dist_to_point = 99999999
    # select_mode = 1  # Firstly select the first branch point (mode 1), then find the closest point to it (mode 2).
    # currernt_coord = [999999, 0, 0]
    # for idx in idxs:
    #     if check_dist(node_coords, point_coord_lst[idx]) or point_lst[idx][3] == 2:
    #         continue
    #     if select_mode == 1:
    #         this_level = point_lst[idx][0]
    #         if this_level == currernt_level:
    #             this_max_dist = point_lst[idx][2]
    #             if this_max_dist > current_max_dist:
    #                 current_max_dist = this_max_dist
    #                 currernt_coord = point_coord_lst[idx]
    #         else:
    #             node_coords.append(
    #                 currernt_coord)  # first coord in coords lst will be appended, so it needs to be deleted afterwards
    #             node_levels.append(currernt_level)
    #             if len(node_coords) == 2:
    #                 current_min_dist_to_point = calc_dist(node_coords[1], point_coord_lst[idx], np.ones(3))  # Can only be used here. Don't copy this method in other situations.
    #             found_node_count += len(point_children_lst[idx]) - 1
    #             currernt_level = this_level
    #             current_max_dist = point_lst[idx][2]
    #             currernt_coord = point_coord_lst[idx]
    #     elif select_mode == 2:
    #         this_level = point_lst[idx][0]
    #         if this_level == currernt_level:
    #             this_min_dist_to_point = calc_dist(node_coords[1], point_coord_lst[idx], np.ones(3))
    #             if this_min_dist_to_point < current_min_dist_to_point:
    #                 current_min_dist_to_point = this_min_dist_to_point
    #                 currernt_coord = point_coord_lst[idx]
    #         else:
    #             node_coords.append(
    #                 currernt_coord)  # first coord in coords lst will be appended, so it needs to be deleted afterwards
    #             found_node_count += 1
    #     if found_node_count >= 1:
    #         select_mode = 2
    #     if found_node_count >= 2:
    #         break
    # del node_coords[0]
    first_level = max(3, np.min(point_lst[:, 3]))
    idx3 = np.where(point_lst[:, 3] == first_level)
    idx31 = -1
    max_max_dist_to_leaf = 0
    for i in idx3[0]:
        if point_lst[i][2] > max_max_dist_to_leaf:
            max_max_dist_to_leaf = point_lst[i][2]
            idx31 = i
    if idx31 == -1:
        print("LUL segmentation failed. No node with level >= 3 found.")
        raise RuntimeError()
        return np.ones(ori_lobe_area.shape) * ori_lobe_area
    node_coords.append(point_coord_lst[idx31])
    idx4 = np.where(point_lst[:, 3] == first_level + 1)[0]
    min_dist = 9999999999
    current_coord = None  # find the point with the minimum dist to idx3
    for idx in idx4:
        dist = calc_dist(node_coords[0], point_coord_lst[idx], [1, 1, 1])
        if dist < min_dist:
            min_dist = dist
            current_coord = point_coord_lst[idx]
    # assert current_coord is not None
    if current_coord is None:
        if len(idx3[0]) > 1:
            node_coords.append(point_coord_lst[idx3[0][1]])
        else:
            print("Warning: only 1 node found in LUL")
            raise ValueError()
    else:
        node_coords.append(current_coord)
    print("Node coords: ")
    print(node_coords)
    skel = np.bool_(skel)
    skel0 = skel.copy()
    skel = skel & lobe_area
    skel = erase(skel, node_coords[0], radius=1)
    if len(node_coords) >= 2:
        skel = erase(skel, node_coords[1], radius=1)
    skel0 = erase(skel0, node_coords[0], radius=1)
    if len(node_coords) >= 2:
        skel0 = erase(skel0, node_coords[1], radius=1)
    # skel_link_out = get_largest_conn(skel0, num=1)
    # skel = skel & (~skel_link_out)  # delete the one that connects with main branch
    skel_0 = skel.copy()
    skel = skel * lobe_area_ori
    skel = keep_conn_with_leaf(skel, all_leaf_points)
    if np.max(measure.label(skel)) < 3:
        skel = skel_0
    skel = get_largest_conn(skel, num=3)
    sh = skel.shape
    skel = zoom(skel, (ori_shape[0] / sh[0], ori_shape[1] / sh[1], ori_shape[2] / sh[2]), order=0)
    # find which conn belongs to which segment
    skel_l = measure.label(skel)
    if True:
        centers_mass = fast_centers_mass(skel, skel_l, [1, 2, 3], ori_lobe_area)
        print("Centers of mass: ")
        print(centers_mass)
    if ball_mask:
        mask = generate_ball_mask(skel, centers_mass, ball_mask)
        skel_l = skel_l * mask
    apical_segment = np.argmax(centers_mass[:, 2] + centers_mass[:, 0]) + 1  # one with maximal x+z value
    centers_mass[apical_segment - 1] = 0
    posterior_segment = np.argmax(centers_mass[:, 1]) + 1  # one with maximal y value. small y is chest side, large y is back side.
    centers_mass[posterior_segment - 1] = 0
    anterior_segment = np.argmax(centers_mass[:, 0]) + 1  # the remaining conn
    # Now we have a mask of rul and three branches of qiguan indicating respective lung segment.
    # We want to make distance maps of each branch.
    print("Making distance maps...")
    anterior_map = make_distance_map((skel_l == anterior_segment), ori_lobe_area)
    posterior_map = make_distance_map((skel_l == posterior_segment), ori_lobe_area)
    apical_map = make_distance_map((skel_l == apical_segment), ori_lobe_area)
    print("Processing distance maps...")
    all_maps = np.stack([apical_map, posterior_map, anterior_map])
    all_maps = np.argmin(all_maps, axis=0)
    all_maps += 1
    all_maps = all_maps * ori_lobe_area
    check_results(all_maps, 1, 3)
    return all_maps


def seg_rml(all_node_points, vol_lobe, skel, all_leaf_points, ori_lobe_area, ball_mask=0, show_center_mass=False):
    print("Segmenting RML...")
    ori_lobe_area = (ori_lobe_area == 2)  # lobe area before downsampling
    ori_shape = ori_lobe_area.shape
    lobe_area = (vol_lobe == 2)
    point_lst = []
    point_coord_lst = []
    point_children_lst = []
    for i in all_node_points:
        if list_indexing(lobe_area, i.coords) and i.spike_label is False:
            lst1 = [i.level, i.min_dist_to_leaf, i.max_dist_to_leaf, i.new_level]
            point_lst.append(lst1)
            point_coord_lst.append(i.coords)
            point_children_lst.append(i.children)
    point_lst = np.array(point_lst)
    if len(point_lst) == 0:
        point_lst = list(point_lst)
        for i in range(2):
            lobe_area = morphology.binary_dilation(lobe_area, np.ones((3, 3, 3)))
        for i in all_node_points:
            if list_indexing(lobe_area, i.coords) and i.spike_label is False:
                lst1 = [i.level, i.min_dist_to_leaf, i.max_dist_to_leaf, i.new_level]
                point_lst.append(lst1)
                point_coord_lst.append(i.coords)
                point_children_lst.append(i.children)
        point_lst = np.array(point_lst)
        if len(point_lst) == 0:
            point_lst = list(point_lst)
            for i in all_node_points:
                if list_indexing(lobe_area, i.coords):
                    lst1 = [i.level, i.min_dist_to_leaf, i.max_dist_to_leaf, i.new_level]
                    point_lst.append(lst1)
                    point_coord_lst.append(i.coords)
                    point_children_lst.append(i.children)
            point_lst = np.array(point_lst)
            if len(point_lst) == 0:
                lobe_area = zoom(lobe_area, (ori_shape[0] / sh[0], ori_shape[1] / sh[1], ori_shape[2] / sh[2]), order=0)
                raise RuntimeError()
                return lobe_area + 4
    # idxs = np.argsort(point_lst[:, 0])
    node_coords = []
    # node_levels = []
    # found_node_count = -1  # first
    # currernt_level = point_lst[idxs[0]][0]
    # current_max_dist = point_lst[idxs[0]][2]
    # currernt_coord = point_coord_lst[idxs[0]]
    # if len(point_coord_lst) > 1:
    #     for idx in idxs:
    #         if point_lst[idx][2] < 30 or point_lst[idx][1] >= 38 or check_dist(node_coords, point_coord_lst[idx]):
    #             continue
    #         this_level = point_lst[idx][0]
    #         if this_level == currernt_level:
    #             this_max_dist = point_lst[idx][2]
    #             if this_max_dist > current_max_dist:
    #                 current_max_dist = this_max_dist
    #                 currernt_coord = point_coord_lst[idx]
    #         else:
    #             node_coords.append(
    #                 currernt_coord)  # first coord in coords lst will be appended, so it needs to be deleted afterwards
    #             node_levels.append(currernt_level)
    #             found_node_count += 1
    #             currernt_level = this_level
    #             current_max_dist = point_lst[idx][2]
    #             currernt_coord = point_coord_lst[idx]
    #         if found_node_count >= 1:
    #             break
    #     if len(node_coords) == 0:
    #         node_coords = [point_coord_lst[0]]
    #     elif len(node_coords) == 1:
    #         node_coords = [point_coord_lst[0]]
    #     else:
    #         del node_coords[0]
    # elif len(point_coord_lst) == 1:
    #     node_coords = point_coord_lst
    # else:
    #     print("Warning: No node points found in RML. Lobe segmentation or qiguan segmentation may be wrong.")
    #     return lobe_area + 4
    idx4 = np.where(point_lst[:, 3] == 4)[0]
    if len(idx4) == 0:
        idx4 = np.where(point_lst[:, 3] == 5)[0]
        if len(idx4) == 0:
            idx4 = np.where(point_lst[:, 3] == 6)[0]
            if len(idx4) == 0:
                idx4 = np.where(point_lst[:, 3] == 3)[0]
                if len(idx4) == 0:
                    print("Warning: No valid nodes found in RML. Lobe seg may be erroneous.")
                    lobe_area = zoom(lobe_area, (ori_shape[0] / sh[0], ori_shape[1] / sh[1], ori_shape[2] / sh[2]), order=0)
                    raise RuntimeError()
                    return lobe_area + 4
    min_y = 9999999999
    current_coord = None  # find the point with the minimum y value
    for idx in idx4:
        y = point_coord_lst[idx][2]
        if y < min_y:
            min_y = y
            current_coord = point_coord_lst[idx]
    if current_coord is None:
        print("RML segmentation failed. Cannot find node.")
        return (np.ones(ori_lobe_area.shape) + 4) * ori_lobe_area
    assert current_coord is not None
    node_coords.append(current_coord)
    print("Node coords: ")
    print(node_coords)
    # print("Node levels: ")
    # print(node_levels)
    skel = np.bool_(skel)
    # skel0 = skel.copy()
    skel = skel & lobe_area
    skel = erase(skel, node_coords[0])
    # skel0 = erase(skel0, node_coords[0])
    # skel_link_out = get_largest_conn(skel0, num=1)
    # skel = skel & (~skel_link_out)  # delete the one that connects with main branch
    skel = keep_conn_with_leaf(skel, all_leaf_points)
    skel = get_largest_conn(skel, num=2)
    sh = skel.shape
    skel = zoom(skel, (ori_shape[0] / sh[0], ori_shape[1] / sh[1], ori_shape[2] / sh[2]), order=0)
    # find which conn belongs to which segment
    skel_l = measure.label(skel)
    centers_mass = fast_centers_mass(skel, skel_l, [1, 2], ori_lobe_area)
    if show_center_mass:
        print("Centers of mass: ")
        print(centers_mass)
    if ball_mask:
        mask = generate_ball_mask(skel, centers_mass, ball_mask)
        skel_l = skel_l * mask
    medial_segment = np.argmin(centers_mass[:, 1]) + 1  # the forward one
    centers_mass[medial_segment - 1] = 0
    lateral_segment = np.argmax(centers_mass[:, 1]) + 1  # the remaining one
    print("Making distance maps...")
    medial_map = make_distance_map((skel_l == medial_segment), ori_lobe_area)
    lateral_map = make_distance_map((skel_l == lateral_segment), ori_lobe_area)
    print("Processing distance maps...")
    all_maps = np.stack([lateral_map, medial_map])
    all_maps = np.argmin(all_maps, axis=0)
    all_maps += 4
    all_maps = all_maps * ori_lobe_area
    check_results(all_maps, 4, 5)
    return all_maps


def seg_rll(all_node_points, vol_lobe, skel, all_leaf_points, ori_lobe_area, spacing, ball_mask=0, show_center_mass=False):
    print("Segmenting RLL...")
    ori_lobe_area = (ori_lobe_area == 3)  # lobe area before downsampling
    ori_shape = ori_lobe_area.shape
    lobe_area = (vol_lobe == 3)
    point_lst = []
    point_coord_lst = []
    point_children_lst = []
    for i in all_node_points:
        if list_indexing(lobe_area, i.coords) and i.spike_label is False:
            lst1 = [i.level, i.min_dist_to_leaf, i.max_dist_to_leaf, i.new_level]
            point_lst.append(lst1)
            point_coord_lst.append(i.coords)
            point_children_lst.append(i.children)
    point_lst = np.array(point_lst)
    x = np.min(point_lst[:, 3])
    y = np.max(point_lst[:, 3])
    if y - x < 3:  # Total num of levels less than 3. Needs dilation.
        for i in range(3):
            lobe_area = morphology.binary_dilation(lobe_area, np.ones((3, 3, 3)))
        point_lst = []
        point_coord_lst = []
        point_children_lst = []
        for i in all_node_points:
            if list_indexing(lobe_area, i.coords) and y - i.new_level <= 3:
                lst1 = [i.level, i.min_dist_to_leaf, i.max_dist_to_leaf, i.new_level]
                point_lst.append(lst1)
                point_coord_lst.append(i.coords)
                point_children_lst.append(i.children)
        point_lst = np.array(point_lst)
    idxs = np.argsort(point_lst[:, 3])
    node_coords = []
    node_levels = []
    found_node_count = 0  # first
    currernt_level = point_lst[idxs[0]][3]
    current_max_dist = point_lst[idxs[0]][2]
    currernt_coord = point_coord_lst[idxs[0]]
    stop_flag = 0
    for num, idx in enumerate(idxs):
        # if point_lst[idx][2] - point_lst[idx][1] > 20 or point_lst[idx][1] >= 38:
        #     continue
        if check_dist(node_coords, point_coord_lst[idx]) or check_z(node_coords, point_coord_lst[idx]):
            continue
        this_level = point_lst[idx][3]
        if this_level == currernt_level:
            this_max_dist = point_lst[idx][2]
            if this_max_dist > current_max_dist:
                current_max_dist = this_max_dist
                currernt_coord = point_coord_lst[idx]
        else:
            if check_z(node_coords, currernt_coord) == 0:
                node_coords.append(currernt_coord)
                node_levels.append(currernt_level)
                found_node_count += 1
            currernt_level = this_level
            current_max_dist = point_lst[idx][2]
            currernt_coord = point_coord_lst[idx]
        # if found_node_count >= 4:
        #     stop_flag = 1
        #     break
        if num + 1 == len(idxs):
            node_coords.append(currernt_coord)
    # if stop_flag == 0:
    #     node_coords.append(currernt_coord)
    #     node_levels.append(currernt_level)
    print("Node coords: ")
    print(node_coords)
    print("Node levels: ")
    print(node_levels)
    skel = np.bool_(skel)
    skel = skel & lobe_area
    if len(node_coords) < 4:
        for num, idx in enumerate(idxs):
        # if point_lst[idx][2] - point_lst[idx][1] > 20 or point_lst[idx][1] >= 38:
        #     continue
            if check_dist(node_coords, point_coord_lst[idx]):
                continue
            this_level = point_lst[idx][3]
            if this_level == currernt_level:
                this_max_dist = point_lst[idx][2]
                if this_max_dist > current_max_dist:
                    current_max_dist = this_max_dist
                    currernt_coord = point_coord_lst[idx]
            else:
                node_coords.append(currernt_coord)
                node_levels.append(currernt_level)
                found_node_count += 1
                currernt_level = this_level
                current_max_dist = point_lst[idx][2]
                currernt_coord = point_coord_lst[idx]
            if found_node_count >= 4:
                stop_flag = 1
                break
            if num + 1 == len(idxs):
                node_coords.append(currernt_coord)
        # if stop_flag == 0:
        #     node_coords.append(currernt_coord)
        #     node_levels.append(currernt_level)
        print("Node coords: ")
        print(node_coords)
        print("Node levels: ")
        print(node_levels)
        skel = np.bool_(skel)
        skel = skel & lobe_area
        if len(node_coords) < 4:
            print("RUL segmentation failed. Found less than 4 nodes.")
            return (np.ones(ori_lobe_area.shape) + 6) * ori_lobe_area
    skel = erase(skel, node_coords[0])
    skel = erase(skel, node_coords[1])
    skel = erase(skel, node_coords[2])
    skel = erase(skel, node_coords[3])
    # find which conn belongs to which segment
    skel_0 = skel.copy()
    skel = keep_conn_with_leaf(skel, all_leaf_points)
    if np.max(measure.label(skel)) < 5:
        skel = skel_0
    skel = get_largest_conn(skel, num=5)
    sh = skel.shape
    skel = zoom(skel, (ori_shape[0] / sh[0], ori_shape[1] / sh[1], ori_shape[2] / sh[2]), order=0)
    skel_l = measure.label(skel)
    if True:
        centers_mass = fast_centers_mass(skel, skel_l, [1, 2, 3, 4, 5], ori_lobe_area)
        print("Centers of mass: ")
        print(centers_mass)
    if ball_mask:
        mask = generate_ball_mask(skel, centers_mass, ball_mask)
        skel_l = skel_l * mask
    centers_mass_copy = centers_mass.copy()
    # TODO(): find corresponding conns
    superior_segment = np.argmax(centers_mass[:, 0]) + 1  # one with maximal z value (highest)
    centers_mass[superior_segment - 1] = 0
    medial_basal_segment = np.argmax(12 * centers_mass[:, 0] + centers_mass[:, 2]) + 1  # highest except superior segment.
    x_of_medial_basal_segment = centers_mass[np.argmax(centers_mass[:, 0])][2]
    # if x_of_medial_basal_segment < (np.max(centers_mass_copy[:, 2]) * 3 + np.average(centers_mass_copy[:, 2])) / 4 and len(node_coords) >= 5:  # if x is too small (smaller than mean of max and average of 5 segments)
    if False:
        skel[skel_l == medial_basal_segment] = 0  # then this is a fake duanzhiqiguan, delete it
        skel = erase(skel, [int(round(node_coords[4][0] / spacing[0])) * 2, int(round(node_coords[4][1] / spacing[1])) * 2, int(round(node_coords[4][2] / spacing[2])) * 2], radius=2)
        skel = get_largest_conn(skel, num=5)
        skel_l = measure.label(skel)
        centers_mass = fast_centers_mass(skel, skel_l, [1, 2, 3, 4, 5], ori_lobe_area)
        print("Centers of mass after modification at 6: ")
        print(centers_mass)
        if ball_mask:
            mask = generate_ball_mask(skel, centers_mass, ball_mask)
            skel_l = skel_l * mask
        superior_segment = np.argmax(centers_mass[:, 0]) + 1  # one with maximal z value (highest)
        centers_mass[superior_segment - 1] = 0
        medial_basal_segment = np.argmax(centers_mass[:, 0]) + 1  # highest except superior segment.
        centers_mass[medial_basal_segment - 1] = 0
        posterior_basal_segment = np.argmax(centers_mass[:, 2] + centers_mass[:, 1]) + 1  # x+y value larger than the remaining two.
        if np.argmax(centers_mass[:, 0]) + 1 == posterior_basal_segment and len(node_coords) >= 6:  # segment 10 is the maximum z among 8, 9 and 10, obviously this is wrong. a fake above 8 below 7 occurs.
            skel[skel_l == posterior_basal_segment] = 0  # delete this
            skel = erase(skel, node_coords[5])
            skel = get_largest_conn(skel, num=5)
            skel_l = measure.label(skel_l)
            centers_mass = fast_centers_mass(skel, skel_l, [1, 2, 3, 4, 5], ori_lobe_area)
            print("Centers of mass after modification at 7: ")
            print(centers_mass)
            if ball_mask:
                mask = generate_ball_mask(skel, centers_mass, ball_mask)
                skel_l = skel_l * mask
            superior_segment = np.argmax(centers_mass[:, 0]) + 1  # one with maximal z value (highest)
            centers_mass[superior_segment - 1] = 0
            medial_basal_segment = np.argmax(centers_mass[:, 0]) + 1  # highest except superior segment.
            centers_mass[medial_basal_segment - 1] = 0
            posterior_basal_segment = np.argmax(centers_mass[:, 2] + centers_mass[:, 1]) + 1  # x+y value larger than the remaining two.
            centers_mass[posterior_basal_segment - 1] = 0
            anterior_basal_segment = np.argmax(centers_mass[:, 0]) + 1  # higher than lateral basal
            centers_mass[anterior_basal_segment - 1] = 0
            lateral_basal_segment = np.argmax(centers_mass[:, 0]) + 1  # the remaining one
        else:  # normal at 7
            centers_mass[posterior_basal_segment - 1] = 0
            anterior_basal_segment = np.argmax(centers_mass[:, 0]) + 1  # higher than lateral basal
            centers_mass[anterior_basal_segment - 1] = 0
            lateral_basal_segment = np.argmax(centers_mass[:, 0]) + 1  # the remaining one
    else:  # normal at 6
        centers_mass[medial_basal_segment - 1] = 0
        posterior_basal_segment = np.argmax(centers_mass[:, 2] + centers_mass[:, 1]) + 1  # x+y value larger than the remaining two.
        if np.argmax(centers_mass[:, 0]) + 1 == posterior_basal_segment and len(node_coords) >= 5:  # segment 10 is the maximum z among 8, 9 and 10, obviously this is wrong. a fake above 8 below 7 occurs.
            skel[skel_l == posterior_basal_segment] = 0  # delete this
            skel = erase(skel, node_coords[4])
            skel = get_largest_conn(skel, num=5)
            skel_l = measure.label(skel_l)
            centers_mass = fast_centers_mass(skel, skel_l, [1, 2, 3, 4, 5], ori_lobe_area)
            print("Centers of mass after modification at 7: ")
            print(centers_mass)
            if ball_mask:
                mask = generate_ball_mask(skel, centers_mass, ball_mask)
                skel_l = skel_l * mask
            superior_segment = np.argmax(centers_mass[:, 0]) + 1  # one with maximal z value (highest)
            centers_mass[superior_segment - 1] = 0
            medial_basal_segment = np.argmax(centers_mass[:, 0]) + 1  # highest except superior segment.
            centers_mass[medial_basal_segment - 1] = 0
            posterior_basal_segment = np.argmax(centers_mass[:, 2] + centers_mass[:, 1]) + 1  # x+y value larger than the remaining two.
            centers_mass[posterior_basal_segment - 1] = 0
            anterior_basal_segment = np.argmax(centers_mass[:, 0]) + 1  # higher than lateral basal
            centers_mass[anterior_basal_segment - 1] = 0
            lateral_basal_segment = np.argmax(centers_mass[:, 0]) + 1  # the remaining one
        else:  # normal at 7
            centers_mass[posterior_basal_segment - 1] = 0
            anterior_basal_segment = np.argmax(centers_mass[:, 0]) + 1  # higher than lateral basal
            centers_mass[anterior_basal_segment - 1] = 0
            lateral_basal_segment = np.argmax(centers_mass[:, 0]) + 1  # the remaining one
    print("Making distance maps...")
    superior_map = make_distance_map((skel_l == superior_segment), ori_lobe_area)
    medial_basal_map = make_distance_map((skel_l == medial_basal_segment), ori_lobe_area)
    anterior_basal_map = make_distance_map((skel_l == anterior_basal_segment), ori_lobe_area)
    lateral_basal_map = make_distance_map((skel_l == lateral_basal_segment), ori_lobe_area)
    posterior_basal_map = make_distance_map((skel_l == posterior_basal_segment), ori_lobe_area)
    print("Processing distance maps...")
    all_maps = np.stack([superior_map, medial_basal_map, anterior_basal_map, lateral_basal_map, posterior_basal_map])
    all_maps = np.argmin(all_maps, axis=0)
    all_maps += 6
    all_maps = all_maps * ori_lobe_area
    check_results(all_maps, 6, 10)
    return all_maps


def seg_lul(all_node_points, vol_lobe, skel, all_leaf_points, spacing, ori_lobe_area, ball_mask=0, show_center_mass=False):
    print("Segmenting LUL...")
    ori_lobe_area = (ori_lobe_area == 4)  # lobe area before downsampling
    ori_shape = ori_lobe_area.shape
    lobe_area = (vol_lobe == 4)
    for i in range(3):
        lobe_area = morphology.binary_dilation(lobe_area, np.ones((3, 3, 3)))
    point_lst = []
    point_coord_lst = []
    point_children_lst = []
    for i in all_node_points:
        if list_indexing(lobe_area, i.coords) and i.spike_label is False:
            lst1 = [i.level, i.min_dist_to_leaf, i.max_dist_to_leaf, i.new_level]
            point_lst.append(lst1)
            point_coord_lst.append(i.coords)
            point_children_lst.append(i.children)
    point_lst = np.array(point_lst)
    point_lst_1 = np.where(point_lst[:, 3] > 2)[0]
    point_lst = point_lst[point_lst_1, :]
    point_coord_lst = np.array(point_coord_lst)
    point_coord_lst = point_coord_lst[point_lst_1]
    idxs = np.argsort(point_lst[:, 3])
    node_coords = []
    node_levels = []
    found_node_count = 0  # first
    idx4 = np.where(point_lst[:, 3] == point_lst[idxs[0]][3])[0]
    min_level = point_lst[idxs[0]][3] + 1
    max_z = 0
    current_coord = None  # find the point with the maximum z value (to distinguish from zhiqiguan num 6)
    for idx in idx4:
        z = point_coord_lst[idx][0]
        if z > max_z:
            max_z = z
            current_coord = point_coord_lst[idx]
    assert current_coord is not None
    node_coords.append(current_coord)
    node_levels.append(point_lst[0][3])
    idx5 = np.where(point_lst[:, 3] == point_lst[idxs[0]][3] + 1)[0]
    point_lst0 = point_lst.copy()
    point_coord_lst0 = point_coord_lst.copy()
    point_lst = point_lst[idx5, :]
    point_coord_lst = point_coord_lst[idx5]
    idx51 = np.argsort(point_coord_lst[:, 0])  # find the top 2 z values
    if len(point_lst) == 1:
        node_coords.append(point_coord_lst[0])
        for idx in idxs:
            if point_lst0[idx][1] >= 8:
                if (len(node_coords) and calc_dist(node_coords[0], point_coord_lst0[idx], spacing) < 8.5): # or point_lst0[idx][3] <= point_lst0[0][3] + 1:
                    continue
                node_coords.append(point_coord_lst0[idx])  # Don't delete node coord 0 here
                node_levels.append(point_lst0[idx, 3])
                found_node_count += 1
            if found_node_count >= 1:
                break
        if found_node_count == 0:
            for idx in idxs:
                if point_lst0[idx][1] >= 4:
                    if (len(node_coords) and calc_dist(node_coords[0], point_coord_lst0[idx], spacing) < 8.5): # or point_lst0[idx][3] <= point_lst0[0][3] + 1:
                        continue
                    node_coords.append(point_coord_lst0[idx])  # Don't delete node coord 0 here
                    node_levels.append(point_lst0[idx, 3])
                    found_node_count += 1
                if found_node_count >= 1:
                    break
    else:
        idx51 = idx51[::-1]
        idx51 = idx51[:2]
        for idx in idx51:
            node_coords.append(point_coord_lst[idx])
    print("Node coords: ")
    print(node_coords)
    print("Node levels: ")
    print(node_levels)
    skel = np.bool_(skel)
    skel = skel & lobe_area
    if len(node_coords) < 3:
        print("LUL segmentation failed. Found less than 3 nodes.")
        return (np.ones(ori_lobe_area.shape) + 10) * ori_lobe_area
    skel = erase(skel, node_coords[0])
    skel = erase(skel, node_coords[1])
    skel = erase(skel, node_coords[2])
    # find which conn belongs to which segment
    skel = keep_conn_with_leaf(skel, all_leaf_points, min_level=min_level)
    skel = get_largest_conn(skel, num=4)
    sh = skel.shape
    skel = zoom(skel, (ori_shape[0] / sh[0], ori_shape[1] / sh[1], ori_shape[2] / sh[2]), order=0)
    skel_l = measure.label(skel)
    if True:
        centers_mass = fast_centers_mass(skel, skel_l, [1, 2, 3, 4], ori_lobe_area)
        print("Centers of mass: ")
        print(centers_mass)
    if ball_mask:
        mask = generate_ball_mask(skel, centers_mass, ball_mask)
        skel_l = skel_l * mask
    # TODO(): find corresponding conns
    apicoposterior_segment = np.argmax(centers_mass[:, 1]) + 1  # one with maximal y value
    centers_mass[apicoposterior_segment - 1] = 0
    anterior_segment = np.argmax(centers_mass[:, 0]) + 1  # highest except apicoposterior segment.
    centers_mass[anterior_segment - 1] = 0
    superior_segment = np.argmax(centers_mass[:, 0]) + 1  # highest except superior and medial basal
    centers_mass[superior_segment - 1] = 0
    inferior_segment = np.argmax(centers_mass[:, 0]) + 1  # the remaining one
    print("Making distance maps...")
    apicoposterior_map = make_distance_map((skel_l == apicoposterior_segment), ori_lobe_area)
    anterior_map = make_distance_map((skel_l == anterior_segment), ori_lobe_area)
    superior_map = make_distance_map((skel_l == superior_segment), ori_lobe_area)
    inferior_map = make_distance_map((skel_l == inferior_segment), ori_lobe_area)
    print("Processing distance maps...")
    all_maps = np.stack([apicoposterior_map, anterior_map, superior_map, inferior_map])
    all_maps = np.argmin(all_maps, axis=0)
    all_maps += 11
    # all_maps[all_maps > 1] += 1
    all_maps = all_maps * ori_lobe_area
    # check_results(all_maps, 11, 14)
    return all_maps, lobe_area, node_coords[0]


def seg_lll(all_node_points, vol_lobe, skel, all_leaf_points, lul_dilated_lobe_area, lul_first_node_coord, spacing, ori_lobe_area, ball_mask=0, show_center_mass=False):
    print("Segmenting LLL...")
    ori_lobe_area = (ori_lobe_area == 5)  # lobe area before downsampling
    ori_shape = ori_lobe_area.shape
    lobe_area = (vol_lobe == 5)
    lobe_area_exclude = lul_dilated_lobe_area
    for i in range(5):
        lobe_area = morphology.binary_dilation(lobe_area, np.ones((3, 3, 3)))
    lobe_area = lobe_area & (~lobe_area_exclude)
    point_lst = []
    point_coord_lst = []
    point_children_lst = []
    point_children_node_lst = []
    for i in all_node_points:
        if list_indexing(lobe_area, i.coords) and i.min_dist_to_leaf > 2:
            lst1 = [i.level, i.min_dist_to_leaf, i.max_dist_to_leaf, i.new_level]
            point_lst.append(lst1)
            point_coord_lst.append(i.coords)
            point_children_lst.append(i.children)
            point_children_node_lst.append(i.children_node)
    for i in point_children_node_lst:
        for j in range(len(i)):
            i[j] = i[j].coords
    point_lst = np.array(point_lst)
    idxs = np.argsort(point_lst[:, 3])  # sort by changed level
    node_coords = []
    node_levels = []
    found_node_count = 0  # first
    currernt_level = point_lst[idxs[0]][3] - 1
    current_max_dist = point_lst[idxs[0]][2]
    currernt_coord = [9999999, 0, 0]
    stop_flag = 0
    max_min_index = 2  # 2 is max, 1 is min, 表示比最大还是最小距离
    for idx in idxs:
        if check_dist(node_coords, point_coord_lst[idx]) or check_z(node_coords, point_coord_lst[idx]) or calc_dist(point_coord_lst[idx], lul_first_node_coord, spacing) < 10:
            continue
        if point_lst[idx][3] <= 2:
            continue
        this_level = point_lst[idx][3]
        if this_level >= max(5, np.min(point_lst[:, 3]) + 2):
            max_min_index = 1
            c = min_z_cos(point_coord_lst[idx], point_children_node_lst[idx])
            if c < 0.5:
                for cn_coord in point_children_node_lst[idx]:
                    if cn_coord in point_coord_lst:
                        idx0 = point_coord_lst.index(cn_coord)
                        this_max_dist = point_lst[idx0][max_min_index]
                        if this_max_dist > current_max_dist:
                            current_max_dist = this_max_dist
                            currernt_coord = cn_coord
                continue
        if this_level == currernt_level:
            this_max_dist = point_lst[idx][max_min_index]
            if this_max_dist > current_max_dist:
                current_max_dist = this_max_dist
                currernt_coord = point_coord_lst[idx]
        else:
            if check_z(node_coords, currernt_coord) == 0:
                node_coords.append(currernt_coord)
                node_levels.append(currernt_level)
                found_node_count += 1
            currernt_level = this_level
            current_max_dist = point_lst[idx][max_min_index]
            currernt_coord = point_coord_lst[idx]
        if found_node_count >= 4:
            stop_flag = 1
            break
    if stop_flag == 0:
        node_coords.append(currernt_coord)
        node_levels.append(currernt_level)
    del node_coords[0]
    del node_levels[0]
    if len(node_levels) < 3:
        print("First attempt failed. Trying again with possible error...")
        for idx in idxs:
            if check_dist(node_coords, point_coord_lst[idx]) or check_z(node_coords, point_coord_lst[idx]) or calc_dist(point_coord_lst[idx], lul_first_node_coord, spacing) < 10:
                continue
            if idx == idxs[0]:
                continue
            this_level = point_lst[idx][0]
            if this_level == currernt_level:
                this_max_dist = point_lst[idx][2]
                if this_max_dist > current_max_dist:
                    current_max_dist = this_max_dist
                    currernt_coord = point_coord_lst[idx]
            else:
                if check_z(node_coords, currernt_coord) == 0:
                    node_coords.append(currernt_coord)
                    node_levels.append(currernt_level)
                    found_node_count += 1
                currernt_level = this_level
                current_max_dist = point_lst[idx][2]
                currernt_coord = point_coord_lst[idx]
            if found_node_count >= 4:
                stop_flag = 1
                break
        if stop_flag == 0:
            node_coords.append(currernt_coord)
            node_levels.append(currernt_level)
        del node_coords[0]
        del node_levels[0]
    print("Node coords: ")
    print(node_coords)
    print("Node levels: ")
    print(node_levels)
    skel = np.bool_(skel)
    skel = skel & lobe_area
    if len(node_coords) < 3:
        print("LLL segmentation failed. Found less than 3 nodes")
        return (np.ones(ori_lobe_area.shape) + 14) * ori_lobe_area
    skel = erase(skel, node_coords[0])
    skel = erase(skel, node_coords[1])
    skel = erase(skel, node_coords[2])
    # find which conn belongs to which segment
    skel_0 = skel.copy()
    skel = keep_conn_with_leaf(skel, all_leaf_points)
    if np.max(measure.label(skel)) < 4:
        skel = skel_0
    skel = get_largest_conn(skel, num=4)
    sh = skel.shape
    skel = zoom(skel, (ori_shape[0] / sh[0], ori_shape[1] / sh[1], ori_shape[2] / sh[2]), order=0)
    skel_l = measure.label(skel)
    if True:
        centers_mass = fast_centers_mass(skel, skel_l, [1, 2, 3, 4], ori_lobe_area)
        print("Centers of mass: ")
        print(centers_mass)
    if ball_mask:
        mask = generate_ball_mask(skel, centers_mass, ball_mask)
        skel_l = skel_l * mask
    superior_segment = np.argmax(centers_mass[:, 0]) + 1  # one with maximal z value
    centers_mass[superior_segment - 1] = 0
    centers_mass[superior_segment - 1][1] = -999999
    pb_segment = np.argmax(centers_mass[:, 1] - centers_mass[:, 0]) + 1
    centers_mass[pb_segment - 1] = 0
    ab_segment = np.argmax(centers_mass[:, 0]) + 1
    centers_mass[ab_segment - 1] = 0
    lb_segment = np.argmax(centers_mass[:, 0]) + 1
    print("Making distance maps...")
    superior_map = make_distance_map((skel_l == superior_segment), ori_lobe_area)
    ab_map = make_distance_map((skel_l == ab_segment), ori_lobe_area)
    lb_map = make_distance_map((skel_l == lb_segment), ori_lobe_area)
    pb_map = make_distance_map((skel_l == pb_segment), ori_lobe_area)
    print("Processing distance maps...")
    all_maps = np.stack([superior_map, ab_map, lb_map, pb_map])
    all_maps = np.argmin(all_maps, axis=0)
    all_maps += 15
    # all_maps[all_maps > 6] += 1
    all_maps = all_maps * ori_lobe_area
    check_results(all_maps, 15, 18)
    return all_maps


def do_seg(vol_lobe, vol_qiguan, spacing, ball_mask=0, do_opening=False, show_center_mass=True):
    assert show_center_mass
    if do_opening:
        vol_qiguan = morphology.binary_opening(vol_qiguan, np.ones((3, 3, 3)))
    vol_lobe_ori = vol_lobe.copy()
    tar_s = 2
    spacing_copy = spacing.copy()
    zoom_rate = (spacing[0] / tar_s, spacing[1] / tar_s, spacing[2] / tar_s)
    spacing = (tar_s, tar_s, tar_s)
    vol_qiguan = zoom(vol_qiguan, zoom_rate, order=0)
    vol_lobe = zoom(vol_lobe, zoom_rate, order=0)
    vol_qiguan = np.bool_(vol_qiguan)
    vol_qiguan = get_largest_conn(vol_qiguan)


    vol_qiguan_skel = morphology.skeletonize_3d(vol_qiguan)
    vol_qiguan_skel[vol_qiguan_skel > 1] = 1
    all_points, all_node_points, all_leaf_points = make_tree(vol_qiguan_skel)
    try:
        all_node_points = get_new_level(all_node_points)
    except:
        for i in range(len(all_node_points)):
            all_node_points[i].new_level = all_node_points[i].level
    try:
        rul_map = seg_rul(all_node_points, vol_lobe, vol_qiguan_skel, all_leaf_points, vol_lobe_ori, ball_mask=ball_mask, show_center_mass=show_center_mass)
    except:
        print("RUL failed.")
        rul_map = r.seg_rul(vol_lobe_ori)
    try:
        rml_map = seg_rml(all_node_points, vol_lobe, vol_qiguan_skel, all_leaf_points, vol_lobe_ori, ball_mask=ball_mask, show_center_mass=show_center_mass)
    except:
        print("RML failed.")
        rml_map = r.seg_rml(vol_lobe_ori)
    try:
        rll_map = seg_rll(all_node_points, vol_lobe, vol_qiguan_skel, all_leaf_points, vol_lobe_ori, spacing_copy, ball_mask=ball_mask, show_center_mass=show_center_mass)
    except:
        print("RLL failed.")
        rll_map = r.seg_rll(vol_lobe_ori)
    try:
        lul_map, area, point = seg_lul(all_node_points, vol_lobe, vol_qiguan_skel, all_leaf_points, spacing, vol_lobe_ori, ball_mask=ball_mask, show_center_mass=show_center_mass)
    except:
        print("LUL failed.")
        lul_map = r.seg_lul(vol_lobe_ori)
    try:
        lll_map = seg_lll(all_node_points, vol_lobe, vol_qiguan_skel, all_leaf_points, area, point, spacing, vol_lobe_ori, ball_mask=ball_mask, show_center_mass=show_center_mass)
    except:
        print("LLL failed.")
        lll_map = r.seg_lll(vol_lobe_ori)
    all_maps = rul_map + rml_map + rll_map + lul_map + lll_map
    return all_maps