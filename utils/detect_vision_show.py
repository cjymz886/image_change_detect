import json
import cv2
import numpy as np
from PIL import Image
import os


def _clip_box(box, w, h):
    """将 box 坐标夹紧到图像范围内，并保证 x1<=x2, y1<=y2"""
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    return [x1, y1, x2, y2]


def convert_box(wres, w1_org, h1_org, w2_org, h2_org,mode='norm1000'):

    """转换模型识别的坐标，得到对应原图的坐标"""
    res = []
    for item in wres:
        s1_box = item['s1_box']
        s2_box = item['s2_box']
        if mode == 'norm1000':
            s1_box = [int(s1_box[0]*w1_org/1000),int(s1_box[1]*h1_org/1000),int(s1_box[2]*w1_org/1000),int(s1_box[3]*h1_org/1000)]
            s2_box = [int(s2_box[0]*w2_org/1000),int(s2_box[1]*h2_org/1000),int(s2_box[2]*w2_org/1000),int(s2_box[3]*h2_org/1000)]
        elif mode == 'norm':
            s1_box = [int(s1_box[0] * w1_org), int(s1_box[1] * h1_org), int(s1_box[2] * w1_org), int(s1_box[3] * h1_org)]
            s2_box = [int(s2_box[0] * w2_org), int(s2_box[1] * h2_org), int(s2_box[2] * w2_org),int(s2_box[3] * h2_org)]

        s1_box = _clip_box(s1_box, w1_org, h1_org)
        s2_box = _clip_box(s2_box, w2_org, h2_org)
        text = item['dres']
        res.append({'s1_box': s1_box, 's2_box': s2_box, 'dres': text})
    return res



def simple_visualize(data, output_path='result.jpg',mode='norm1000' ):
    """
    简化版本：只绘制矩形框，不添加文字
    """
    img1_path = data['img1_path']
    img2_path = data['img2_path']

    # 读取图片
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    w1_org, h1_org = img1.shape[1],img1.shape[0]
    w2_org, h2_org = img2.shape[1],img2.shape[0]

    pred_list = data['dres']
    pred_list = convert_box(pred_list, w1_org, h1_org, w2_org, h2_org, mode=mode)

    colors = [
        (255, 0, 0),  # 红色
        (0, 255, 0),  # 绿色
        (0, 0, 255),  # 蓝色
        (255, 255, 0),  # 青色
    ]

    # 绘制所有框
    for idx, pred in enumerate(pred_list):
        color = colors[idx % len(colors)]

        # 在两张图上绘制
        s1_box = pred['s1_box']
        s2_box = pred['s2_box']

        x1, y1, x2, y2 = s1_box
        cv2.rectangle(img1, (x1, y1), (x2, y2), color, 3)

        x1, y1, x2, y2 = s2_box
        cv2.rectangle(img2, (x1, y1), (x2, y2), color, 3)

    # 水平拼接
    margin = 20  # 边距
    gap = 30  # 图片间距
    top_margin = 80  # 顶部边距（用于标题）
    bottom_margin = 100  # 底部边距（用于描述）

    canvas_width = w1_org + gap + w2_org + margin * 2
    canvas_height = max(h1_org, h2_org) + top_margin + bottom_margin

    # 创建白色画布
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # 计算图片在画布上的位置
    x_offset1 = margin
    x_offset2 = margin + w1_org + gap
    y_offset1 = top_margin + (max(h1_org, h2_org) - h1_org) // 2
    y_offset2 = top_margin + (max(h1_org, h2_org) - h2_org) // 2

    # 将图片放到画布上
    canvas[y_offset1:y_offset1 + h1_org, x_offset1:x_offset1 + w1_org] = img1
    canvas[y_offset2:y_offset2 + h2_org, x_offset2:x_offset2 + w2_org] = img2

    # 保存结果
    result_pil = Image.fromarray(canvas)
    result_pil.save(output_path)




