严格配对-可见光-红外严格配对的火点检测数据集，1367张，提供yolo，voc，coco三种标注方式![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/77eef289f7004470ac489ea07e5434cf.png)

图像尺寸:640
类别数量:1类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e29f8e30b300448383befe32ec02ef76.png)

训练集图像数量:1367; 验证集图像数量:0； 测试集图像数量:0
类别名称: 每一类图像数 ，每一类标注数![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fea99a2f56344b23866e28b333e0488a.png)

fire: 1298,1298
image num: 1367![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/470276b79ab74db48fa5df7004c2d161.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/67250e4fceb3451e98fa38bbb6e9ad65.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7b853936625e4d37a9bed59090134c58.png)
# 可见光‑红外配对火点检测数据集详情
## 一、数据集基础信息
- **数据集名称**：可见光‑红外双模态配对火点目标检测数据集
- **图像总数量**：1367对（可见光图 + 对应红外热成像图，严格一一配对）
- **单图分辨率**：640×640
- **标注格式**：YOLO‑TXT + VOC‑XML + COCO‑JSON **三格式全套标注**
- **任务类型**：目标检测(Object Detection)、红外‑可见光跨模态检测、火灾识别
- **适用模型**：YOLOv5/v8/v11、RT‑DETR、Faster‑RCNN、YOLO‑World、双模态融合网络、红外火灾预警算法
- **类别总数**：1类

## 二、类别清单&标注框统计
| 类别ID | 英文标签 | 中文名称 | 含目标图像数量 | 标注框总数 |
|:---:|---|---|---|---|
| 0 | fire | 火点/火焰 | 1298 | 1298 |

>剩余69张图像为**负样本（无火焰背景图）**，用于提升模型抗干扰能力，减少误检。

## 三、数据集划分方案（推荐 7‑2‑1）
>原始数据集未做切分，给出开箱即用划分
- 训练集(train)：957 对（70%）
- 验证集(val)：273 对（20%）
- 测试集(test)：137 对（10%）

### YOLO文件夹目录结构（开箱即用）
```
fire_vis_ir_dataset/
├─images/
│   ├─train/        #可见光图像
│   ├─train_ir/     #配对红外图像
│   ├─val/
│   ├─val_ir/
│   ├─test/
│   └─test_ir/
├─labels/           #YOLO txt标注
│   ├─train/
│   ├─val/
│   └─test/
├─voc_annotations/  #VOC‑XML标注文件
├─coco_annotations/ #COCO‑JSON标注文件
│   ├─train.json
│   ├─val.json
│   └─test.json
└─fire.yaml         #YOLO配置文件
```
#### fire.yaml 完整内容（直接复制使用）
```yaml
path: ./fire_vis_ir_dataset
train: images/train
val: images/val
test: images/test

names:
  0: fire
```

## 四、数据集特点说明
1. **可见光‑红外严格配对**：每一张可见光火焰图像，都有同场景、同视角对应的红外热成像图；适合做多模态图像融合、跨模态目标检测、红外火灾预警研究。
2. **三标注兼容**：YOLO、VOC、COCO三种格式，覆盖几乎所有主流深度学习检测框架，无需格式转换。
3. **样本构成**：1298张含火焰正样本+69张无火焰负样本；场景包含森林火、室内明火、野外篝火、烟气干扰等复杂工况；存在火光反光、烟雾遮挡、弱小火点等难点样本。
4. **应用场景**：森林火情监控、变电站/厂区红外火灾预警、无人机火情巡检、视频烟火检测、安防火灾识别、双模态火灾检测算法研究。

## 五、YOLOv11快速训练代码
```python
from ultralytics import YOLO

#加载预训练权重
model = YOLO("yolo11n.pt")

#单模态（可见光）训练
results = model.train(
    data="fire.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    device=0,
    patience=30,
    mosaic=1.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4
)
```


