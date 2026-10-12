"""
图片对差异检测主程序
"""
import sys
# sys.path.append("../")

import logging
import time
import json
from typing import Optional
import re
from PIL import Image
from config.config import Config
from utils.logging import setup_logging
from engine.llm import ChatLLM
from engine.vllm import ChatVLLM
from config.prompts import (change_detect_whole_think_prompt,
                            change_detect_position_think_prompt,
                            change_detect_text_think_prompt,
                            text_similarity_prompt)


setup_logging(log_file='./logs/icdtools.log')

class ICDDetector(object):
    def __init__(
        self,
        cfg = None,
        planner=None,
        memory=None,
        executor=None,
        max_steps: int = 10,
        max_time: int = 300,
        max_tokens: int = 4000,
        root_cache_dir: str = "cache",
    ):
        self.planner = planner
        self.memory = memory
        self.executor = executor
        self.max_steps = max_steps
        self.max_time = max_time
        self.max_tokens = max_tokens
        self.root_cache_dir = root_cache_dir

        self.llm_client = ChatLLM(model_string=cfg.glm_model_name,url=cfg.glm_url,api_key=cfg.glm_key,is_multimodal=True)
        # self.llm_client = ChatVLLM()

    def parse_detect_output(self,output):
        """格式化模型输出"""
        def judge_box(box):
            if len(box) == 4 and box[2] >= box[0] and box[3] >= box[1]:
                return True
            else:
                return False
        p1 = re.compile('存在.*差异')
        p2 = re.compile('<box>(.*)</box>')
        out_list = []
        if not output:
            return out_list
        try:
            if isinstance(output, str):
                output = [output]
            for out_text in output:
                if p1.findall(out_text):
                    for block in out_text.split('\n\n'):
                        s1_box, s2_box, text = [], [], ''
                        for line in block.split('\n'):
                            if '图1坐标：' in line:
                                s1_box = eval(p2.findall(line)[0])
                            if '图2坐标：' in line:
                                s2_box = eval(p2.findall(line)[0])
                            if '解释：' in line:
                                text = line.replace('解释：', '')
                        if s1_box and s2_box and text:
                            if judge_box(s1_box) and judge_box(s2_box):
                                out_list.append({'s1_box': s1_box, 's2_box': s2_box, 'dres': text})
        except Exception as e:
            logging.error(f"parse_detect_output:{e}")
        return out_list

    def detect_whole_diff(self,img1,img2):
        """detect whole different"""
        res = self.llm_client.generate(content=[img1, img2, change_detect_whole_think_prompt],
                                        temperature=1.0,
                                        max_tokens=4096,
                                        top_p=0.95
                                        )
        logging.info(f"llm for whole detect,and output is:{res}")
        res = self.parse_detect_output(res)
        return res

    def detect_position_diff(self,img1,img2):
        """detect position different"""
        res = self.llm_client.generate(content=[img1, img2, change_detect_position_think_prompt],
                                        temperature=1.0,
                                        max_tokens=4096,
                                        top_p=0.95
                                        )
        logging.info(f"llm for position detect,and output is:{res}")
        res = self.parse_detect_output(res)
        return res


    def detect_text_diff(self,img1,img2):
        """detect text different"""
        res = self.llm_client.generate(content=[img1, img2, change_detect_text_think_prompt],
                                        temperature=1.0,
                                        max_tokens=4096,
                                        top_p=0.95
                                        )
        logging.info(f"llm for text detect,and output is:{res}")
        res = self.parse_detect_output(res)
        return res

    def calculate_iou(self, box1, box2):
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

    def parse_text_sim(self,output):
        if not output:
            return []
        try:
            return eval(output)
        except Exception as e:
            logging.error(f"parse_text_sim:{e}")
            return []

    def judge_text_similarity(self,input_text):
        prompt = text_similarity_prompt.format(input_text=input_text)
        output = self.llm_client.generate(content=prompt,
                                        temperature=0.8,
                                        max_tokens=4096,
                                        top_p=0.95
                                        )
        logging.info(f"vllm for judging text similarity,and output is:{output}")
        output = self.parse_text_sim(output)
        return output

    def sort_diff_by_iou(self,mres,threshold=0.1):
        """merge all dectection results"""
        N = len(mres)
        map_res = {}
        remove_list = set()
        for i in range(N-1):
            i_s1_box = mres[i]['s1_box']
            i_s2_box = mres[i]['s2_box']
            map_res[i] = []
            for j in range(i+1,N):
                if j in remove_list:
                    continue
                j_s1_box = mres[j]['s1_box']
                j_s2_box = mres[j]['s2_box']
                iou1 = self.calculate_iou(i_s1_box, j_s1_box, type='inter')
                iou2 = self.calculate_iou(i_s2_box, j_s2_box, type='inter')
                if iou1>threshold and iou2>threshold:
                    map_res[i].append(j)
                    remove_list.append(j)
        if N-1 not in remove_list:
            map_res[N-1] = [N-1]
        return map_res

    def compute_area(self,item):
        s1_box = item['s1_box']
        s2_box = item['s2_box']
        area1 = int((s1_box[2] - s1_box[0]) * (s1_box[3] - s1_box[1]))
        area2 = int((s2_box[2] - s2_box[0]) * (s2_box[3] - s2_box[1]))
        return area1+area2

    def filter_same_diff(self,mres):
        input_text = ''
        for i,item in enumerate(mres):
            text = item['dres']
            input_text += f"S{i}: {text}\n"
        output = self.judge_text_similarity(input_text)
        text_sim_res = self.parse_text_sim(output)
        if not text_sim_res:
            return mres
        fres = []
        for sim_list in text_sim_res:
            if 0<len(sim_list)<=1:
                fres.append(mres[sim_list[0]])
            sel_idx = -1
            max_area = 0
            for idx in sim_list:
                area = self.compute_area(mres[idx])
                if area > max_area:
                    sel_idx = idx
            fres.append(mres[sel_idx])
        return fres

    def detect(self, img1:Image, img2:Image) -> dict:
        """
        detect change between images
        """
        # whole detect
        wres = self.detect_whole_diff(img1,img2)

        # position detect
        pres = self.detect_position_diff(img1,img2)

        # text detect
        tres = self.detect_text_diff(img1, img2)

        #merge detect
        mres = wres + pres + tres
        logging.info(f"the merge detection results :{mres}")

        dres = self.filter_same_diff(mres)
        logging.info(f"the final detection results :{dres}")
        return dres


if __name__ == "__main__":
    img1_path = "/home/cjy/mystudy/octotools-main/icdtools/test_examples/f1-1.png"
    img2_path = "/home/cjy/mystudy/octotools-main/icdtools/test_examples/f1-2.png"
    with open(img1_path, "rb") as f:
        img1 = f.read()

    with open(img2_path, "rb") as f:
        img2 = f.read()

    cfg = Config()
    icd = ICDDetector(cfg)
    res = icd.detect(img1,img2)

