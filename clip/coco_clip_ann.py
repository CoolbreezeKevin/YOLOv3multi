# -*- coding: utf-8 -*-
"""
@author Majx
@date 2022年09月09日 19:00:39

@describe TODO
"""
import numpy as np
import glob
import os
import cv2
from tqdm import tqdm, trange
from osgeo import gdal
import re
import json
from pycocotools.coco import COCO
# image_settings = ['jpg','tif','png','jpeg','tiff']
import numpy as np
import glob
import os
import cv2
from tqdm import tqdm, trange
from osgeo import gdal
import re
# image_settings = ['jpg','tif','png','jpeg','tiff']
class COCODatasetImageCropper(object):
    def __init__(self):
        self.image_settings = ['jpg', 'tif', 'png', 'jpeg', 'tiff']
        self.clip_h_iter = 0
        self.clip_w_iter = 0
        self.clip_h = 0
        self.clip_w = 0
        self.split_num = []
        self.name_rules = ['pure','normal','complete']
    def __call__(self,
                 data_path,
                 output_path,
                 w_size,
                 h_size,
                 image_type,
                 padding_fill=True,
                 is_geotiff_image=False,
                 h_overlap=0,
                 w_overlap=0,
                 name_rule='pure',
                 remove_ratio=0.1,
                 coco_json_path=None,
                 coco_out_path=None,
                 coco_thre=5,
                 key=lambda name:int(name.split('_')[-2])):
        assert image_type in self.image_settings, f".name should be in {self.image_settings}"
        assert name_rule in self.name_rules, f".name rules need to be in {self.name_rules}."
        isExists = os.path.exists(output_path)
        if not isExists:
            os.makedirs(output_path)
        image_num = 0
        total_num = 0
        new_ann_list = []
        new_image_list = []
        name_id_dict = {}
        if coco_json_path is not None:
            assert coco_out_path is not None,f"coco json output path needed."
            coco = COCO(coco_json_path)
            for i,j in coco.imgs.items():
                name_id_dict[j['file_name']] = j['id']

        images = sorted(glob.glob(os.path.join(data_path , "*."+image_type)),key=key)
        if output_path[-1] != '/':
            out_images = output_path + "/"
        else:
            out_images = output_path
        print('Start Crop...' + '\n')
        for index in trange(0, len(images)):
            ori_img_name = images[index].split('/')[-1]
            cur_id = name_id_dict[ori_img_name]
            img = cv2.imread(images[index],flags=cv2.IMREAD_UNCHANGED)
            # a = np.max(img)
            path = images[index].split('/')[-1].split('.')[0]
            self.clip_h = (img.shape[0]-h_overlap) % (h_size-h_overlap) + h_overlap
            self.clip_w = (img.shape[1]-w_overlap) % (w_size-w_overlap) + w_overlap
            self.clip_h_iter = (img.shape[0]-h_overlap) // (h_size-h_overlap) + 1 if self.clip_h != 0 else (img.shape[0]-h_overlap) // (h_size-h_overlap)
            self.clip_w_iter = (img.shape[1]-w_overlap) // (w_size-w_overlap)  + 1 if self.clip_w != 0 else (img.shape[1]-w_overlap) // (w_size-w_overlap)
            if padding_fill == True:
                if self.clip_h != 0:
                    padding_mtx = np.zeros((h_size - self.clip_h,img.shape[1],
                                            img.shape[2])) if len(img.shape) == 3 else np.zeros((h_size - self.clip_h,img.shape[1]))
                    img = np.concatenate((img,padding_mtx),axis=0).astype('uint8')
                if self.clip_w != 0:
                    padding_mtx = np.zeros((img.shape[0],w_size - self.clip_w,
                                            img.shape[2])) if len(img.shape) == 3 else np.zeros((img.shape[0],w_size - self.clip_w))
                    img = np.concatenate((img,padding_mtx),axis=1).astype('uint8')
                x,y = img.shape[0],img.shape[1]
                for k in range(0,self.clip_h_iter):
                # for k in range(0, x, h_size - overlap):
                    if k==self.clip_h_iter - 1 and self.clip_h / h_size <= remove_ratio:
                        continue
                    for j in range(0,self.clip_w_iter):
                        if j == self.clip_w_iter - 1 and self.clip_w / w_size <= remove_ratio:
                            continue
                        img_crop = img[k * (h_size - h_overlap):(k + 1) * h_size - k * h_overlap,
                                       j * (w_size - w_overlap):(j + 1) * w_size - j * w_overlap,]
                        out_im_name = ''
                        if name_rule == 'pure':
                            out_im_name = out_images + str(total_num) + "." + image_type
                        elif name_rule == 'complete':
                            out_im_name = out_images + path + "_"+str(image_num) + "_"+str(total_num) + "." + image_type
                        else:
                            out_im_name = out_images + path + "_" + str(total_num) + "." + image_type
                        cv2.imwrite(out_im_name, img_crop)
                        im_dict = {}
                        im_dict['file_name'] = out_im_name.split('/')[-1]
                        im_dict['height'] = h_size
                        im_dict['width'] = w_size
                        im_dict['id'] = total_num
                        new_image_list.append(im_dict)
                        for num, ann_list in coco.imgToAnns.items():
                            if ann_list[0]['image_id'] == cur_id:
                                for ann in ann_list:
                                    bbox = ann['bbox']
                                    x1 = bbox[0]
                                    y1 = bbox[1]
                                    x2 = bbox[0] + bbox[2]
                                    y2 = bbox[1] + bbox[3]
                                    if x1 >= j * (w_size - w_overlap) + coco_thre and \
                                            y1 >= k * (h_size - h_overlap) + coco_thre and \
                                            x2 <= j * (w_size - w_overlap) + w_size - coco_thre and \
                                            y2 <= k * (h_size - h_overlap) + h_size - coco_thre:
                                        nx1 = x1 - j * (w_size - w_overlap)
                                        ny1 = y1 - k * (h_size - h_overlap)
                                        nx2 = x2 - (j * (w_size - w_overlap))
                                        ny2 = y2 - (k * (h_size - h_overlap))
                                        new_bbox = [nx1, ny1, nx2 - nx1, ny2 - ny1]
                                        new_ann = ann.copy()
                                        new_ann['bbox'] = new_bbox
                                        new_ann['image_id'] = total_num
                                        new_ann['id'] = len(new_ann_list)
                                        new_ann_list.append(new_ann)
                                        a=1
                                break
                        image_num +=1
                        total_num +=1
                self.split_num.append(image_num)
                image_num = 0
            else:
                for k in range(0,self.clip_h_iter if self.clip_h == 0 else self.clip_h_iter - 1):
                    for j in range(0,self.clip_w_iter  if self.clip_w == 0 else self.clip_w_iter - 1):
                        img_crop = img[k * (h_size - h_overlap):(k + 1) * h_size - k * h_overlap,
                                       j * (w_size - w_overlap):(j + 1) * w_size - j * w_overlap,]

                        if name_rule == 'pure':
                            cv2.imwrite(
                                out_images + str(total_num) + "." + image_type,
                                img_crop)
                        elif name_rule == 'complete':
                            cv2.imwrite(
                                out_images + path + "_" + str(image_num) + "_" + str(total_num) + "." + image_type,
                                img_crop)
                        else:
                            cv2.imwrite(
                                out_images + path + "_" + str(total_num) + "." + image_type,
                                img_crop)
                        image_num +=1
                        total_num +=1
                    if self.clip_w != 0 and self.clip_w / w_size > remove_ratio:
                        img_crop = img[k * (h_size - h_overlap):(k + 1) * h_size - k * h_overlap,
                                       (self.clip_w_iter - 1) * (w_size - w_overlap):(self.clip_w_iter - 1) * (w_size - w_overlap) + self.clip_w,]

                        if name_rule == 'pure':
                            cv2.imwrite(
                                out_images + str(total_num) + "." + image_type,
                                img_crop)
                        elif name_rule == 'complete':
                            cv2.imwrite(
                                out_images + path + "_" + str(image_num) + "_" + str(total_num) + "." + image_type,
                                img_crop)
                        else:
                            cv2.imwrite(
                                out_images + path + "_" + str(total_num) + "." + image_type,
                                img_crop)
                        image_num += 1
                        total_num += 1
                if self.clip_h != 0 and self.clip_h / h_size > remove_ratio:
                    for j in range(0,self.clip_w_iter - 1):
                        img_crop = img[(self.clip_h_iter - 1) * (h_size - h_overlap):(self.clip_h_iter - 1) * (h_size - h_overlap) + self.clip_h,
                                       j * (w_size - w_overlap):(j + 1) * w_size - j * w_overlap,]

                        if name_rule == 'pure':
                            cv2.imwrite(
                                out_images + str(total_num) + "." + image_type,
                                img_crop)
                        elif name_rule == 'complete':
                            cv2.imwrite(
                                out_images + path + "_" + str(image_num) + "_" + str(total_num) + "." + image_type,
                                img_crop)
                        else:
                            cv2.imwrite(
                                out_images + path + "_" + str(total_num) + "." + image_type,
                                img_crop)
                        image_num += 1
                        total_num += 1
                if self.clip_h != 0 and self.clip_w != 0 and self.clip_w / w_size > remove_ratio and self.clip_h / h_size > remove_ratio:
                    img_crop = img[(self.clip_h_iter - 1) * (h_size - h_overlap):(self.clip_h_iter - 1) * (h_size - h_overlap) + self.clip_h,
                                   (self.clip_w_iter - 1) * (w_size - w_overlap):(self.clip_w_iter - 1) * (w_size - w_overlap) + self.clip_w,]

                    if name_rule == 'pure':
                        cv2.imwrite(
                            out_images + str(total_num) + "." + image_type,
                            img_crop)
                    elif name_rule == 'complete':
                        cv2.imwrite(out_images + path + "_" + str(image_num) + "_" + str(total_num) + "." + image_type,
                                    img_crop)
                    else:
                        cv2.imwrite(
                            out_images + path + "_" + str(total_num) + "." + image_type,
                            img_crop)
                    image_num += 1
                    total_num += 1
                self.split_num.append(image_num)
                image_num = 0

        # with open(output_path.split('/')[-1] + '.txt', "w") as f:
        #     for i in range(len(self.split_num)):
        #         f.write(str(self.split_num[i]))
        #         f.write('\n')
        # self.split_num=[]
        out_dict = {}
        out_dict['images'] = new_image_list
        out_dict['type'] = coco.dataset['type']
        out_dict['annotations'] = new_ann_list
        out_dict['categories'] = coco.dataset['categories']
        with open(coco_out_path,'w') as f:
            json.dump(out_dict,f)
        print('Done')

if __name__ == '__main__':
    cropper = COCODatasetImageCropper()
    # cropper(data_path="D:/Dataset/SemanticSegmentation/AerialImageDataset/train/images",
    #         output_path="D:/Dataset/SemanticSegmentation/AerialImageDataset_512/train/images",
    #         w_size=512,
    #         h_size=512,
    #         image_type='tif',
    #         padding_fill=True,
    #         key=lambda name:int(re.findall(r'\d',name)[0]))
    # cropper(data_path='/home/majx/data/赛道1_道路提取和交叉口识别数据集/chusai_release/test/images',
    #         output_path='/home/majx/data/rspic_2022/test/images',
    #         w_size=1024,
    #         h_size=1024,
    #         image_type='tif',
    #         padding_fill=True,
    #         h_overlap=0,
    #         w_overlap=0,
    #         name_rule='pure',
    #         remove_ratio=0,
    #         key=lambda name:int(name.split('/')[-1].split('.')[0]))
    cropper(data_path='/home/majx/data/赛道1_道路提取和交叉口识别数据集/chusai_release/train/images',
            output_path='/home/majx/data/赛道1_道路提取和交叉口识别数据集/chusai_release/train/images_clip',
            w_size=1024,
            h_size=1024,
            image_type='tif',
            padding_fill=True,
            h_overlap=512,
            w_overlap=512,
            name_rule='complete',
            remove_ratio=0,
            coco_json_path='/home/majx/data/赛道1_道路提取和交叉口识别数据集/chusai_release/train/instances_train.json',
            coco_out_path='/home/majx/data/赛道1_道路提取和交叉口识别数据集/chusai_release/train/clip_instances_train.json',
            coco_thre=5,
            key=lambda name:int(name.split('/')[-1].split('.')[0]))
