import os
import pathlib
import time

import config
from toolkit import ml
from ctypes import windll, Structure, c_long, byref
from toolkit.WindowWriter import WindowWriter
# from PIL import Image
import cv2


def user_config():
    types = create(['0_xxx'])

    each = 200

    # use fixed count strategy
    start_of_each = 0

    # uncomment this line to enable use auto start index when just single type(s)
    start_of_each = start_of_each if not len(types) == 1 else len(ml.getAllFiles(f'data/{types[0]}'))

    return types, each, start_of_each


def create(types):
    for t in types:
        ml.ensure_dir(f'data/{t}')
    return types


class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]


def region_screenshot(l, t, r, b, resolution=0.5):
    output = 'tmp/quick/0.png'
    ml.ensure_dir('tmp/quick')
    pic = ml.screen_shot()

    img = cv2.imread(pic)
    crop_img = img[int(t):int(b), int(l):int(r)]
    crop_img = cv2.resize(crop_img, (0, 0), fx=resolution, fy=resolution)

    cv2.imwrite(output, crop_img)

    return output


def queryMousePosition(coordinates_scale=5 / 4):
    pt = POINT()
    windll.user32.GetCursorPos(byref(pt))
    return {"x": pt.x * coordinates_scale, "y": pt.y * coordinates_scale}


if __name__ == '__main__':

    types, each, start_of_each = user_config()

    ml.ensure_dir('dir')
    current_dir = pathlib.Path().resolve()

    w = WindowWriter(1499, 161, 0.8)

    print(current_dir)
    w.set_text('beign in 3 sec...')

    for t in types:
        w.set_text(f'next is:{t}')
        t = f'data/{t}'
        time.sleep(2)
        ml.ensure_dir(t)
        os.system('cls')
        for i in range(each):
            if i % 10 == 0:
                w.set_text(f'{i}/{each}')

            i += start_of_each
            pos = queryMousePosition()
            x = pos['x']
            y = pos['y']
            # width, height = im_size, im_size
            # f=region_screenshot(max(x - width, 0)
            #                          , max(0, y - height), x, y)
            f = region_screenshot(*config.im_capture__left_top_right_bottom, config.resolution_for_capture_image)
            ml.copy_force(f, f'{t}/{i}.png')
