from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

config_file = "../configs/e2e_vertex_rcnn_R_50_FPN_doubleBranch.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

def load(path):
    pil_image = Image.open(path).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()

image = load(r'E:\ResearchDOC\term2\MSRCNN_polyrnn\maskrcnn-benchmark\dataForLittleTest\IMG\austin2_02_16.tif')
# imshow(image)

predictions = coco_demo.run_on_opencv_image(image)
imshow(predictions)