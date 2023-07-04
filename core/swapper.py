import os
import subprocess
import shutil
from tqdm import tqdm
import cv2
import insightface
import core.globals
from core.analyser import get_face

FACE_SWAPPER = None


def get_face_swapper():
    global FACE_SWAPPER
    if FACE_SWAPPER is None:
        model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../inswapper_128.onnx')
        FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=core.globals.providers)
    return FACE_SWAPPER