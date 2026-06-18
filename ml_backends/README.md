# ML Backends 目录

存放所有 Label Studio ML Backend 服务。

## 目录说明

| 目录 | 任务类型 | 模型 | 端口 | 状态 |
|------|----------|------|------|------|
| image-detection | 目标检测 | YOLOv8n | 9090 | 已配置 |
| text-classification | 文本分类 | DeepSeek V4 Flash | 9091 | 已配置 |
| text-ner | 命名实体识别 | DeepSeek V4 Flash | 9092 | 已配置 |

## 启动命令

激活conda环境
```powershell
conda activate ls-ml
```


```powershell
# 图像检测
cd ml_backends/image-detection
label-studio-ml start . --port 9090

# 文本分类
cd ml_backends/text-classification
label-studio-ml start . --port 9091

# 文本 NER
cd ml_backends/text-ner
label-studio-ml start . --port 9092
```

## Label Studio 配置

在 Label Studio 项目设置 > Machine Learning 中添加模型，URL 对应如下：

| 项目类型 | ML Backend URL |
|----------|----------------|
| 图像检测 | http://localhost:9090 |
| 文本分类 | http://localhost:9091 |
| 文本 NER | http://localhost:9092 |

## 添加新服务

```powershell
cd ml_backends
label-studio-ml init <新服务名称>
```

然后修改 `model.py` 实现对应任务的 `predict()` 方法，并在此 README 中记录端口信息。
