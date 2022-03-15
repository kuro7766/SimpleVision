import time

from toolkit.WindowWriter import WindowWriter
import shutil
from toolkit import quick_predict

w = WindowWriter(1499, 161, 0.8)
# print(model(dataset.from_screen()))
w.set_text('go')
while True:
    scene = quick_predict.predict_with_file('model/test')

    w.set_text(f'Scene:{scene}')

    time.sleep(1)
    shutil.rmtree('tmp')
