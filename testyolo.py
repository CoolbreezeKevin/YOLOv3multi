import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from model_utils.config import config as default_config
from src.yolo import YOLOV3DarkNet53, YoloWithLossCell
from src.initializer import load_yolov3_params
from src.util import load_backbone
import numpy as np
import mindspore
import json


if __name__=="__main__":
    net = YOLOV3DarkNet53(is_training=True, config=default_config,)
    # load_yolov3_params(default_config, net)
    load_backbone(net, default_config.pretrained_backbone, default_config)
    # load_backbone(net, default_config.pretrained_backbone, default_config)
    print("Success")
    x = mindspore.Tensor(np.ones([1, 3, 1024, 1024]), mindspore.float32)
    output_big, output_me, output_small, seg_road = net(x)
    print("Success")