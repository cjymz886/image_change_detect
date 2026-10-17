

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "Please install the openai package by running `pip install openai`, and add 'DEEPSEEK_API_KEY' to your environment variables.")

import sys
sys.path.append("/home/cjy/mystudy/octotools-main/icdtools")
import os
import json
import base64
import platformdirs
from typing import List, Union
from config.config import Config
from engine.base import EngineLM, CachedEngine


class ChatLLM(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."
    def __init__(
            self,
            model_string: str = None,
            url: str = None,
            api_key: str = None,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            is_multimodal: bool = True,
            use_cache: bool = True,
            **kwargs):
        """
        :param model_string:
        :param system_prompt:
        :param is_multimodal:
        """
        self.model_string = model_string
        self.url = url
        self.api_key = api_key
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.is_multimodal = is_multimodal

        if self.use_cache:
            root = platformdirs.user_cache_dir("icdtools")
            cache_path = os.path.join(root, f"cache_vllm_{self.model_string}.db")
            self.image_cache_dir = os.path.join(root, "image_cache")
            os.makedirs(self.image_cache_dir, exist_ok=True)
            super().__init__(cache_path=cache_path)

        try:
            self.client = OpenAI(
                base_url=self.url,
                api_key=self.api_key,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to connect to mdoel server. Please ensure the server is running and try again.")


    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs):
        if isinstance(content, str):
            return self._generate_text(content, system_prompt=system_prompt, **kwargs)

        elif isinstance(content, list):
            if (not self.is_multimodal):
                raise NotImplementedError(f"Multimodal generation is only supported for {self.model_string}.")

            return self._generate_multimodal(content, system_prompt=system_prompt, **kwargs)

    def _generate_text(
            self, prompt, system_prompt=None, temperature=1, max_tokens=4000, top_p=0.95, response_format=None
    ):

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        if self.use_cache:
            cache_key = sys_prompt_arg + prompt
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        # Chat models without structured outputs
        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": prompt},
            ],
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        response = response.choices[0].message.content

        if self.use_cache:
            self._save_cache(cache_key, response)
        return response

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def _format_content(self, content: List[Union[str, bytes]]) -> List[dict]:
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                base64_image = base64.b64encode(item).decode('utf-8')
                img_url = f"data:image/jpeg;base64,{base64_image}"
                formatted_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": img_url,
                    }
                })
            elif isinstance(item, str):
                formatted_content.append({
                    "type": "text",
                    "text": item
                })
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content

    def _generate_multimodal(
            self, content: List[Union[str, bytes]], system_prompt=None, temperature=1.0, max_tokens=4000, top_p=0.95,
            response_format=None
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)

        if self.use_cache:
            cache_key = sys_prompt_arg + json.dumps(formatted_content)
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": formatted_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        response_text = response.choices[0].message.content

        if self.use_cache:
            self._save_cache(cache_key, response_text)
        return response_text


if __name__ == "__main__":
    img1_path = "/home/cjy/mystudy/octotools-main/icdtools/test_examples/f1-1.png"
    img2_path = "/home/cjy/mystudy/octotools-main/icdtools/test_examples/f1-2.png"
    with open(img1_path, "rb") as f:
        img1 = f.read()

    with open(img2_path, "rb") as f:
        img2 = f.read()

    cfg = Config()
    client = ChatLLM(model_string=cfg.kimi_model_name,url=cfg.kimi_url,api_key=cfg.kimi_key,is_multimodal=True)
    prompt = """请你根据输入的两张图片，依照<要求>，识别它们存在的差异。

<输出格式>
结论：共有2处差异

图1坐标：<box>[712,805,922,859]</box>
图2坐标：<box>[726,795,942,857]</box>
解释：颜色差异，第一张图片底部的小猫衣服图案是红色，第二张图片底部的小猫衣服图案是橙色，两张图片存在局部颜色差异

图1坐标：<box>[154,868,830,937]</box>
图2坐标：<box>[154,866,833,937]</box>
解释：标点符号不同，图1文本为“未经蒸煮、干燥、冷冻、水解等处理过程”，图2文本为“未经蒸煮，干燥，冷冻，水解等处理过程”，图1为顿号，图2为逗号

<要求>
1. 请仔细对照两种图片，找出它们在相应位置存在哪些差异；
2. 识别出差异后，请按<输出格式>进行输出，另外说明：结论显示有n处差异，下面就要对应有n个具体差异说明（两张图差异区域的坐标，以及差异解释）
3. 坐标box是按照[x1,y1,x2,y2]形式，其中的具体值是图片的真实绝对坐标；输出格式要严格按照<box>[]</box>格式;
4. 不要进行思考分析，直接输出结果"""

    res = client.generate([img1, img2, prompt])('中国十大美女')
    print(res)


