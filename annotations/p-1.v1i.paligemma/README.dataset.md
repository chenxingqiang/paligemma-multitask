# p-1.v1i.paligemma 数据集

## 概述

这是一个用于缺陷检测和分析的数据集，特别关注结构中的空洞和裂缝。数据集包含874张带注释的图像，采用PaliGemma格式标注。

## 数据来源

此数据集由Roboflow提供，发布于2024年6月17日。
链接：https://universe.roboflow.com/rfgd-hvcuq/p-1-xfp7l

## 许可证

数据集采用CC BY 4.0许可证。

## 数据集特点

- 包含874张图像，每张图像均标注了空洞和/或裂缝
- 图像已调整为640x640像素大小（拉伸模式）
- 使用PaliGemma注释格式，包含位置和类型信息
- 注释采用<loc>标签来标记缺陷的精确位置

## 预处理

对每张图像进行了以下预处理：
- 自动调整像素数据方向（去除EXIF方向信息）
- 调整大小至640x640（拉伸）

未应用图像增强技术。

## 使用方法

此数据集可用于：
1. 微调PaliGemma等多模态模型进行缺陷检测
2. 训练专门的计算机视觉模型进行空洞和裂缝检测
3. 研究基于位置的缺陷描述生成

## 数据格式

数据集采用PaliGemma特有的注释格式：
```json
{"image": "image_filename.jpg", "prefix": "detect void ; crack", "suffix": "<loc0486><loc0156><loc0750><loc0592> void"}
```

其中：
- `image`：图像文件名
- `prefix`：检测任务指示
- `suffix`：包含位置标签<locX><locY><locW><locH>和类型标签 