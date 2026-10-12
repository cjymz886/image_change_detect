import sys
sys.path.append('/home/cjy/mystudy/octotools-main/')
import os.path
import time
import warnings
from PIL import Image
import numpy as np
import torch
from torch.jit import TracerWarning
warnings.filterwarnings('ignore', category=TracerWarning)

from icdtools.tools.layer_detector.rfdetr import RFDETRBase
from icdtools.utils import logger,timer_decorator




class LayerDetector(object):
    def __init__(self):
        self.max_pixels = 256*32*32
        self.save_path = '/home/cjy/mystudy/octotools-main/icdtools/logs/layer_detect'
        self.detect_model = self.load_detect_model()

    def load_detect_model(self):
        model_path = "/home/cjy/my_project/change_detect/models/layer_detect_model.pth"
        logger.info(f"加载图层检测模型: {model_path}")
        # model = RFDETRMedium(pretrain_weights=model_path)
        model = RFDETRBase(pretrain_weights=model_path)
        model.optimize_for_inference()
        logger.info("图层检测模型加载并优化完成")
        return model

    def sorted_box(self,data):
        """box 面积排序"""
        box_map = {}
        for i, det in enumerate(data):
            x1, y1, x2, y2 = det
            area = int((x2 - x1) * (y2 - y1))
            assert area > 0
            box_map[i] = area
        box_map = sorted(box_map.items(),key=lambda x:x[1],reverse=False)
        return box_map

    def get_box_mask(self,box,w,h):
        """获取box 对应的mask"""
        x1, y1, x2, y2 = box
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        return mask.astype(bool)

    def crop_image(self,org_image, box):
        """得到切片图层"""
        org_image = np.array(org_image)
        x1, y1, x2, y2 = box
        return org_image[y1:y2, x1:x2]

    def filter_mask_similarity(self,m_list, m, threshold=0.8):
        """过滤相似度很高的图层"""
        m = m.ravel()
        flag = True
        for ml in m_list:
            ml = ml.ravel()
            x = (ml & m).sum()
            y = (ml | m).sum()
            y1 = ml.sum()
            ssim_score = x / y
            ssim_score2 = x / y1
            if ssim_score > threshold or ssim_score2 > threshold:
                flag = False
                break
        return flag

    def layer_detect(self,image):
        """检测图层 by """
        logger.debug("开始图层检测...")
        detections = self.detect_model.predict(image, threshold=0.2)
        width, height = image.size[0], image.size[1]
        xyxy = detections.xyxy
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], a_min=0, a_max=width)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], a_min=0, a_max=height)
        seg_list = xyxy.tolist()
        box_map = self.sorted_box(seg_list)
        layer_res = []
        mask_list = []
        for i, item in enumerate(box_map):
            ind, area = item
            det = seg_list[ind]
            det = [int(d) for d in det]
            seg_mask = self.get_box_mask(det, width, height)
            seg_img = self.crop_image(image, det)
            flag = self.filter_mask_similarity(mask_list, seg_mask)
            if not flag:
                continue
            mask_list.append(seg_mask)
            layer_res.append([seg_img, det])
        logger.debug(f"图层检测完成，检测到 {len(layer_res)} 个图层")
        return layer_res

    def match_boxes(self,box1, d):
        """为boxes1中的每个框找到boxes2中最相似的框"""
        best_iou = 0
        best_match = -1
        for j, item in enumerate(d):
            box2 = item[1]
            iou = self.calculate_iou(box1, box2)
            if iou > best_iou:
                best_iou = iou
                best_match = j
        if best_iou >0.5:
            return best_match
        else:
            return -1

    def get_seg_scale(self,image):
        """如果切片的图很大，也需要进行压缩处理"""
        org_w,org_h = image.size
        if org_w * org_h > self.max_pixels:
            new_image = self.resize_image(image)
            new_w,new_h = new_image.size
            scale_x = new_w / org_w
            scale_y = new_h / org_h
            return new_image,[scale_x,scale_y]
        else:
            return image,[1.0,1.0]

    def multiple_32(self,num: float):
        return int(round(num / 32) * 32)

    def resize_image(self,image):
        """对大图片进行缩放"""
        width, height = image.size
        if width * height > self.max_pixels:
            scale = (width * height / self.max_pixels) ** 0.5
            height, width = int(height / scale), int(width / scale)
            height = self.multiple_32(height)
            width = self.multiple_32(width)
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        return image

    def calculate_iou(self,box1, box2):
        x1, y1, x2, y2 = box1
        a1, b1, a2, b2 = box2

        # 计算交集
        inter_x1 = max(x1, a1)
        inter_y1 = max(y1, b1)
        inter_x2 = min(x2, a2)
        inter_y2 = min(y2, b2)

        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0  # 无交集
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (a2 - a1) * (b2 - b1)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def layer_match(self,d1, d2):
        """图片匹配"""
        logger.debug(f"开始图层匹配，d1有 {len(d1)} 个图层，d2有 {len(d2)} 个图层")
        result = []
        i = 1
        matched_count = 0
        for item in d1:
            seg_img, seg_box = item[0], item[1]
            j = self.match_boxes(seg_box, d2)
            if j==-1:
                logger.debug(f"图层 {i} 未找到匹配项")
                continue
            matched_count += 1
            seg_img2, seg_box2 = d2[j][0], d2[j][1]
            seg1_path = os.path.join(self.save_path,  f"p1_{str(i)}.png")
            seg2_path = os.path.join(self.save_path,  f"p1_{str(i)}.png")

            seg_img = Image.fromarray(seg_img)
            seg_img2 = Image.fromarray(seg_img2)

            seg_img.save(seg1_path)
            seg_img2.save(seg2_path)
            logger.debug(f"图层 {i} 匹配成功，保存到 {seg1_path} 和 {seg2_path}")
            result.append({'seg1': seg1_path, 'seg2': seg2_path, 'box1': seg_box, 'box2': seg_box2})
            del d2[j]
            i += 1
        logger.debug(f"图层匹配完成，成功匹配 {matched_count} 对图层")
        return result

    def detector(self,img1,img2):
        logger.info("stpe2：原图太大，进行分层检测...")
        layer_detect_start = time.time()
        d1 = self.layer_detect(img1)
        d2 = self.layer_detect(img2)
        logger.info(f"图层检测完成，图片1检测到 {len(d1)} 个图层，图片2检测到 {len(d2)} 个图层，耗时: {time.time() - layer_detect_start:.2f}秒")
        logger.info("开始图层匹配...")
        if len(d1) < len(d2):
            layer_res = self.layer_match(d1, d2)
        else:
            layer_res = self.layer_match(d2, d1)
        logger.info(f"图层匹配完成，匹配到 {len(layer_res)} 对图层")
        return layer_res


if __name__ == '__main__':
    img1_path = '/icdtools/test_examples/f1-1.png'
    img2_path = '/icdtools/test_examples/f2-1.png'

    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    ld = LayerDetector()
    res = ld.detector(img1,img2)
    print(res)