from toolkit import ml
import time
import pyautogui
import re
import tkinter
import ctypes
from tkinter import messagebox
import random
from PIL import Image
from toolkit.WindowWriter import WindowWriter

import time

from toolkit.WindowWriter import WindowWriter
import shutil
from toolkit import quick_predict


def main():
    w = WindowWriter(1499, 161, 0.8)
    # print(model(dataset.from_screen()))
    w.set_text('go')
    while True:
        scene = quick_predict.predict_with_file('model/xiaobo_status_check.pt')
        # fix prediction
        if scene <= 1:
            quick_predict.clean_heavy_models()
            scene = quick_predict.predict_with_file('model/fishing_dialog_status.pt', use_only_once=True)

        w.set_text(f'Scene:{scene}')

        time.sleep(1)
        shutil.rmtree('tmp')
