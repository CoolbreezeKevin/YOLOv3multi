import albumentations as A
import cv2

######### multiscale has realised in yolo

tranform_test = A.Compose([
    A.Flip(p=0.5),
    A.RandomRotate90(p=0.5)],
    bbox_params=A.BboxParams(format="pascal_voc"),
)

# To Use
# transformed = tranform_test(image="your_image", mask="your_mask", bboxes="your_boxes", )#
