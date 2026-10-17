# image_change_detect
构建一个两张高度相似图片的差异检测智能工具

# 检测主要思路
利用多种视觉大模型进行差异识别，先整体识别，再按差异类型（位置差异、文本差异等）进行细分化检测，将检测的结果融合进行过滤，过滤后再让模型进行校正判断，得到最终检测结果

# 当前进展
- 实现单个模型的检测流程
- 对检测结果进行可视化检测

# 下步计划
- 并行多个模型进行检测

# 检测可视化样例
![检测样例](https://github.com/cjymz886/image_change_detect/blob/main/logs/vision_results/f1-1_dres.jpg)
