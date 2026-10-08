import numpy as np
import onnxruntime as ort
from PIL import Image

session = ort.InferenceSession("../models/depth-anything-v2-small/onnx/model.onnx")

def compute(image):
    original_width, original_height = image.size

    image_redim = image.resize((518, 518))
    x = np.array(image_redim).astype(np.float32) / 255.0
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = np.transpose(x, (2, 0, 1))[np.newaxis].astype(np.float32)

    depth = session.run(None, {"pixel_values": x})[0][0]

    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
    return Image.fromarray(depth.astype(np.uint8)).resize((original_width, original_height))