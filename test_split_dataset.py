# from sahi.slicing import slice_coco
# # coco_dict, save_path = slice_coco(
# #     coco_annotation_file_path: str,
# #     image_dir: str,
# #     output_coco_annotation_file_name: str,
# #     output_dir: Optional[str] = None,
# #     ignore_negative_samples: bool = False,
# #     slice_height: int = 512,
# #     slice_width: int = 512,
# #     overlap_height_ratio: float = 0.2,
# #     overlap_width_ratio: float = 0.2,
# #     min_area_ratio: float = 0.1,
# #     out_ext: Optional[str] = None,
# #     verbose: bool = False,
# # )
# import gdal

import os
import json
import copy
import cv2
from tqdm import tqdm
import numpy as np
from clip.coco_clip_ann import CoCoDatasetImageCropper
from clip.clip import DatasetImageCropper
from src.aug_transforms import tranform_test
js_path = '/dataset/train/instances_train.json'

img = cv2.imread('testim/images_clip/1_0_0.tif',-1)
mask = cv2.imread('testim/masks_clip/1_0_0.png',-1)
mask = np.expand_dims(mask, 2)
coco2 = json.load(open(js_path,'r'))
transformed = tranform_test(image=img, mask=mask, bboxes=np.array([[56, 56, 200, 200, 1]]) )
print(len(coco2["annotations"]))
# newjs = coco2.copy()
# print(len(newjs['images']))
# newjs['images'] = newjs['images'][0:3]

# print(len(newjs["annotations"]))

# newjs['annotations'] = []
# for im in newjs['images']:
#     for i in tqdm(coco2["annotations"]):
#         if im["id"]==i["image_id"]:
#             newjs['annotations'].append(i)

# b = json.dumps(newjs)
# f2 = open('newjs.json', 'w')
# f2.write(b)
# f2.close()

# coco2 = json.load(open('newjs.json','r'))
# mask_path = '/dataset/train/masks'
# image_path = '/dataset/train/images'

# for im in coco2["images"]:
#     file_name = im["file_name"]
#     im_path = os.path.join(image_path, file_name)
#     ma_path = os.path.join(mask_path, file_name.replace(".tif", ".png"))
#     os.system(f"cp {os.path.abspath(im_path)} /storage/official/cv/yolov3_darknet53/testim/im")
#     os.system(f"cp {os.path.abspath(ma_path)} /storage/official/cv/yolov3_darknet53/testim/ma")
# os.system(f"cp ./newjs.json /storage/official/cv/yolov3_darknet53/testim")

# cropper = COCODatasetImageCropper()
# cropper(data_path='/storage/official/cv/yolov3_darknet53/testim/im',
#         output_path='/storage/official/cv/yolov3_darknet53/testim/images_clip',
#         w_size=1024,
#         h_size=1024,
#         image_type='tif',
#         padding_fill=True,
#         h_overlap=512,
#         w_overlap=512,
#         name_rule='complete',
#         remove_ratio=0,
#         coco_json_path='/storage/official/cv/yolov3_darknet53/testim/newjs.json',
#         coco_out_path='/storage/official/cv/yolov3_darknet53/testim/outjs.json',
#         coco_thre=5,
#         key=lambda name:int(name.split('/')[-1].split('.')[0]))

# coco2 = json.load(open('/storage/official/cv/yolov3_darknet53/testim/outjs.json','r'))

cropper = DatasetImageCropper()
cropper(data_path='/storage/official/cv/yolov3_darknet53/testim/ma',
        output_path='/storage/official/cv/yolov3_darknet53/testim/masks_clip',
        w_size=1024,
        h_size=1024,
        image_type='png',
        padding_fill=True,
        h_overlap=512,
        w_overlap=512,
        name_rule='complete',
        remove_ratio=0,
        key=lambda name:int(name.split('/')[-1].split('.')[0]))
