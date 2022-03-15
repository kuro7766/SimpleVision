import torch
from toolkit import dataset
import os

_cache = {}


def clean_heavy_models():
    kl = []
    for k in _cache:
        kl.append(k)
    for k in kl:
        del _cache[k]


def predict_with_file(model_file, use_only_once=False) -> int:
    if not model_file.startswith('model'):
        model_file = os.path.join('model/', model_file)
    if _cache.__contains__(model_file):
        model = _cache[model_file]
    else:
        model = torch.jit.load(model_file)
        model.cuda().eval()

    scene = (torch.argmax(model(dataset.from_screen())))

    if use_only_once:
        if _cache.__contains__(model_file):
            del _cache[model_file]
    else:
        _cache[model_file] = model
    return int(scene)
