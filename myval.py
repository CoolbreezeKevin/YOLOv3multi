import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from pycocotools.coco import COCO
import datetime
from model_utils.config import config as default_config
from src.yolo import YOLOV3DarkNet53
import numpy as np
import mindspore as ms
import json
from collections import defaultdict
import cv2
import sys
from src.yolo_dataset import create_test_dataset
from src.transforms import statistic_normalize_img
from tqdm import tqdm
def sofmax(logits):
	e_x = np.exp(logits)
	probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
	return probs

def load_parameters(network, file_name):
    print('load param from {}'.format(file_name))
    param_dict = ms.load_checkpoint(file_name)
    param_dict_new = {}
    for key, values in param_dict.items():
        if key.startswith('moments.'):
            continue
        elif key.startswith('yolo_network.'):
            param_dict_new[key[13:]] = values
        else:
            param_dict_new[key] = values
    ms.load_param_into_net(network, param_dict_new)

class Inference():
    def __init__(self, coco_path='/storage/official/cv/yolov3_darknet53/cutted_coco2.json', 
                instance_test = 'instances_test.json',
                num_classes=2, 
                eval_ignore_threshold=0.01, 
                out_path = '/dataset/testim',
                nms_thresh = 0.7):
        self.instance_test = COCO(instance_test) #json.load(open(instance_test, 'r'))
        self.num_classes=num_classes
        self._coco = COCO(coco_path)
        self.json_file = json.load(open(coco_path, 'r'))
        self.results = {}
        self._img_ids = list(sorted(self._coco.imgs.keys()))
        self.eval_ignore_threshold = eval_ignore_threshold
        self.coco_catIds = self._coco.getCatIds()
        self.det_boxes = []
        self.save_prefix = out_path
        self.nms_thresh = nms_thresh

    def get_img_id(self, file_name):
        images = self.json_file['images']
        for im in images:
            if im['file_name'] == file_name:
                return im['id']
        return False

    def do_nms_for_results(self):
        """Get result boxes."""
        for img_id in self.results:
            for clsi in self.results[img_id]:
                dets = self.results[img_id][clsi]
                dets = np.array(dets)
                keep_index = self._nms(dets, self.nms_thresh)

                keep_box = [{'image_id': int(img_id),
                             'category_id': int(clsi),
                             'bbox': list(dets[i][:4].astype(float)),
                             'score': dets[i][4].astype(float)}
                            for i in keep_index]
                self.det_boxes.extend(keep_box)

    def _nms(self, predicts, threshold):
        """Calculate NMS."""
        # convert xywh -> xmin ymin xmax ymax
        x1 = predicts[:, 0]
        y1 = predicts[:, 1]
        x2 = x1 + predicts[:, 2]
        y2 = y1 + predicts[:, 3]
        scores = predicts[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        reserved_boxes = []
        while order.size > 0:
            i = order[0]
            reserved_boxes.append(i)
            max_x1 = np.maximum(x1[i], x1[order[1:]])
            max_y1 = np.maximum(y1[i], y1[order[1:]])
            min_x2 = np.minimum(x2[i], x2[order[1:]])
            min_y2 = np.minimum(y2[i], y2[order[1:]])

            intersect_w = np.maximum(0.0, min_x2 - max_x1 + 1)
            intersect_h = np.maximum(0.0, min_y2 - max_y1 + 1)
            intersect_area = intersect_w * intersect_h
            ovr = intersect_area / (areas[i] + areas[order[1:]] - intersect_area)

            indexes = np.where(ovr <= threshold)[0]
            order = order[indexes + 1]
        return reserved_boxes

    def write_result(self):
        """Save result to file."""
        import json
        t = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        try:
            self.file_path = self.save_prefix + '/predict' + t + '.json'
            f = open(self.file_path, 'w')
            json.dump(self.det_boxes, f)
        except IOError as e:
            raise RuntimeError("Unable to open json file to dump. What(): {}".format(str(e)))
        else:
            f.close()
            return self.file_path

    def detect(self, outputs, batch, image_shape, image_id):
        """Detect boxes."""
        outputs_num = len(outputs)
        # output [|32, 52, 52, 3, 85| ]
        for batch_id in range(batch):
            for out_id in range(outputs_num):
                # 32, 52, 52, 3, 85
                out_item = outputs[out_id]
                # 52, 52, 3, 85
                out_item_single = out_item[batch_id, :]
                # get number of items in one head, [B, gx, gy, anchors, 5+80]
                dimensions = out_item_single.shape[:-1]
                out_num = 1
                for d in dimensions:
                    out_num *= d
                ori_w, ori_h = image_shape[batch_id]
                img_id = int(image_id[batch_id])

                imgs = self._coco.loadImgs(img_id)
                f_name = imgs[0]['file_name']
                img_id = int(f_name.split("_")[0])
                offset_x, offset_y = int(f_name.split("_")[1]), int(f_name.split("_")[2])
                x = out_item_single[..., 0] * ori_w
                y = out_item_single[..., 1] * ori_h
                w = out_item_single[..., 2] * ori_w
                h = out_item_single[..., 3] * ori_h

                conf = out_item_single[..., 4:5]
                cls_emb = out_item_single[..., 5:]

                cls_argmax = np.expand_dims(np.argmax(cls_emb, axis=-1), axis=-1)
                x = x.reshape(-1)
                y = y.reshape(-1)
                w = w.reshape(-1)
                h = h.reshape(-1)
                cls_emb = cls_emb.reshape(-1, self.num_classes)
                conf = conf.reshape(-1)
                cls_argmax = cls_argmax.reshape(-1)

                x_top_left = x - w / 2.
                y_top_left = y - h / 2.
                # create all False
                flag = np.random.random(cls_emb.shape) > sys.maxsize
                for i in range(flag.shape[0]):
                    c = cls_argmax[i]
                    flag[i, c] = True
                confidence = cls_emb[flag] * conf
                # confidence = confidence/np.max(confidence)
                for x_lefti, y_lefti, wi, hi, confi, clsi in zip(x_top_left, y_top_left, w, h, confidence, cls_argmax):
                    if confi < self.eval_ignore_threshold:
                        continue
                    if img_id not in self.results:
                        self.results[img_id] = defaultdict(list)
                    x_lefti = max(0, x_lefti)
                    y_lefti = max(0, y_lefti)
                    wi = min(wi, ori_w)
                    hi = min(hi, ori_h)
                    # transform catId to match coco
                    coco_clsi = self.coco_catIds[clsi]
                    self.results[img_id][coco_clsi].append([x_lefti+offset_y, y_lefti+offset_x, wi, hi, confi])


class ValDataLoader():
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.imgs = os.listdir(img_dir)
    
    def getim(self,):
        img = self.imgs.pop()
        ary = cv2.imread(os.path.join(self.img_dir, img), -1)
        return ary.astype('float32'), img, self.checkvalid(ary)
    
    def checkvalid(self, img):
        unq = np.unique(img)
        if unq.shape[0]<=2:
            return False
        return True

    def pop(self,):
        if self.imgs:
            ary, img, flag = self.getim()
            ary=statistic_normalize_img(ary)
            return ary.astype('float32'), img, flag
    
    def __str__(self,):
        return self.imgs
    def __len__(self,):
        return len(self.imgs)

def save_seg(seg, im_name, path='/dataset/testim/pred', thred = 0.05):
    if not os.path.exists(path):
        os.mkdir(path)
    if thred:
        seg = seg>=thred
        seg = seg*255
    else:
        seg = seg*255
    seg = seg.astype('uint8')
    cv2.imwrite(os.path.join(path, im_name.replace('.tif', '.png')), seg)


def main():
    ds = ValDataLoader('/dataset/testim/cutted_test2')
    net = YOLOV3DarkNet53(is_training=False)
    load_parameters(net, 'outputs/yolo101/ckpt_0/yolov3_101_1181.ckpt')
    net.set_train(False)
    detection = Inference()
    total = len(ds)
    while len(ds):
        print("Inferencing: {} / {}".format((total-len(ds)), total), end='\r')
        ary, im_name, flag = ds.pop()
        if flag:
            img_id = detection.get_img_id(im_name)

            x = ms.Tensor.from_numpy(ary)
            x = ms.ops.transpose(x, (2,0,1))
            x = ms.ops.expand_dims(x, 0)
            batch, c, w, h = x.shape
            # x = ms.ops.MaxPool(pad_mode="VALID", kernel_size=2, strides=1)(x)
            
            output_big, output_me, output_small, seg_road = net(x)
            


            seg_road = seg_road.asnumpy()
            seg_road = seg_road.squeeze(0)
            seg_road = seg_road.squeeze(0)
            output_big = output_big.asnumpy()
            output_me = output_me.asnumpy()
            
            detection.detect([output_big, output_me], batch, np.array([[w,h]]), [img_id])
            # save_seg(seg_road, im_name)

            x = cv2.pyrDown(ary)
            x = ms.Tensor.from_numpy(x)
            x = ms.ops.transpose(x, (2,0,1))
            x = ms.ops.expand_dims(x, 0)
            output_big, output_me, output_small, seg_road = net(x)
            seg_road = seg_road.asnumpy()
            seg_road = seg_road.squeeze(0)
            seg_road = seg_road.squeeze(0)
            output_big = output_big.asnumpy()
            output_me = output_me.asnumpy()
            
            detection.detect([output_big, output_me], batch, np.array([[w,h]]), [img_id])
            seg_road=cv2.pyrUp(seg_road)
            save_seg(seg_road, im_name)
        else:
            x,y,_ = ary.shape
            seg_road = np.zeros((x,y))
            # pass
            save_seg(seg_road, im_name)
    print('Doing NMS...')
    detection.do_nms_for_results()
    print('Write results...')
    result_file_path = detection.write_result()
    print('Finished !')

if __name__=='__main__':
    coco = COCO('/storage/official/cv/yolov3_darknet53/cutted_coco2.json')
    ds = create_test_dataset('/dataset/testim/cutted_test2', 
        '/storage/official/cv/yolov3_darknet53/cutted_coco2.json')
    from src.yolosimple import YOLOV3DarkNet53
    net = YOLOV3DarkNet53(is_training=False)
    # load_parameters(net, 'outputs/yolo101/ckpt_0/yolov3_101_1181.ckpt')
    load_parameters(net, 'outputs/yolo_only/ckpt_0/yolov3_21_1181.ckpt')
    
    net.set_train(False)
    net.requires_grad=False
    detection = Inference()
    for i, data in tqdm(enumerate(ds.create_dict_iterator(num_epochs=1))):
        image = data["image"]
        image_id = data["img_id"]
        flag = data["flag"]
        batch, c, w, h = image.shape
        image_id = image_id.asnumpy()

        imgs = coco.loadImgs(image_id)
        im_name = imgs[0]['file_name']

        if flag:
            output_big, output_me, _, seg_road = net(image)
            seg_road = seg_road.asnumpy()
            seg_road = seg_road.squeeze(0)
            seg_road = seg_road.squeeze(0)
            output_big = output_big.asnumpy()
            output_me = output_me.asnumpy()
            # save_seg(seg_road, im_name)
            
            detection.detect([output_big, output_me], batch, np.array([[w,h]]), image_id)

            image2 = ms.ops.avg_pool2d(image, 2,2)
            output_big, output_me, _, seg_road = net(image2)
            seg_road = seg_road.asnumpy()
            seg_road = seg_road.squeeze(0)
            seg_road = seg_road.squeeze(0)
            output_big = output_big.asnumpy()
            output_me = output_me.asnumpy()
            detection.detect([output_big, output_me], batch, np.array([[w,h]]), image_id)
        else:
            seg_road = np.zeros((w,h))
            # save_seg(seg_road, im_name)   
    print('Doing NMS...')
    detection.do_nms_for_results()
    print('Write results...')
    result_file_path = detection.write_result()
    print('Finished !')            

            
