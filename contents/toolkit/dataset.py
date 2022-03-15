import numpy as np
import cv2
import torch

import config
import im_capture_tool as im_capture
from toolkit import ml
import re


def from_types(types_dir_name: list, tag):
    files = []
    for type_dir_name in types_dir_name:
        dir = f'data/{type_dir_name}/'
        files.extend(ml.getAllFiles(dir))
    return from_file_list(
        files, tag)


def from_just_class_nums(nums: list):
    d = get_all_classes_dict()
    cats_x = []
    cats_y = []
    for k in d:
        if int(k) in nums:
            x, y = from_types([f'{k}_{d[k]}'], int(k))
            cats_x.append(x)
            cats_y.append(y)
    # data = torch.cat([data, data2])
    # tags = torch.cat([tags, tags2])
    return torch.cat(cats_x), torch.cat(cats_y)


def from_file_list(file_list: list, tag):
    data = np.zeros(0)
    tags = []
    for i in file_list:
        cls = tag
        im = cv2.imread(i)
        im = im / 255
        # print(im.shape)
        data = np.append(data, im)
        tags.append(cls)

    # target = keras.utils.to_categorical(np.array(tags), num_classes=2).reshape((-1,2))
    data = data.reshape((-1, 255, 480, 3))
    data = torch.transpose(torch.Tensor(data).cuda() if config.use_gpu else torch.Tensor(data), 1, 3)
    tags = torch.tensor(tags).cuda() if config.use_gpu else torch.tensor(tags)
    return data, tags


def from_single_file(file, tag):
    return from_file_list([file], tag)


def from_screen():
    f = im_capture.region_screenshot(0, 0, 1920, 1020, config.resolution_for_capture_image)
    return from_single_file(f, 0)[0]


def get_all_classes_dict():
    """
   @:returns {'0':'xiao_bo'}
    """

    datas = ml.getAllFiles('data')
    d = {}
    for data in datas:
        # print(data)
        data = data.replace('data\\', '')
        name = re.findall('^\d+', data)[0]
        real_name = data.replace(f'{name}_', '')
        d[name] = real_name
    return d


if __name__ == '__main__':
    print(get_all_classes_dict())
