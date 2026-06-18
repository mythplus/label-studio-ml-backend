# Label Studio ML 后端是什么？

Label Studio ML 后端是一个 SDK，可让您打包机器学习代码并将其转换为 Web 服务器。
该 Web 服务器可以连接到正在运行的 [Label Studio](https://labelstud.io/) 实例，以自动化标注任务。

如果您只需要将静态预标注数据加载到 Label Studio 中，运行 ML 后端可能对您来说过于复杂。
相反，您可以 [导入预标注数据](https://labelstud.io/guide/predictions.html)。

# 快速开始

要使用这些模型，请使用 [docker-compose](https://docs.docker.com/compose/install/) 来运行 ML 后端
服务器。

使用以下命令在 `http://localhost:9090` 启动 ML 后端：

```bash
git clone https://github.com/HumanSignal/label-studio-ml-backend.git
cd label-studio-ml-backend/label_studio_ml/examples/{MODEL_NAME}
docker-compose up
```

将 `{MODEL_NAME}` 替换为您要使用的模型名称（见下文）。

## 允许 ML 后端访问 Label Studio 数据

在大多数情况下，您需要设置 `LABEL_STUDIO_URL` 和 `LABEL_STUDIO_API_KEY` 环境变量，以允许 ML 后端访问 Label Studio 中的媒体数据。
[在文档中阅读更多](https://labelstud.io/guide/ml#Allow-the-ML-backend-to-access-Label-Studio-data)。

# 模型

本仓库支持以下模型。其中一些无需任何额外设置即可工作，而另一些
需要设置额外的参数。

查看**必需参数**列，看看是否需要设置任何额外的参数。

- **预标注**列表示该模型是否可用于 Label Studio 中的预标注：
  您可以在打开标注页面或为一组数据运行预测后看到预标注数据。
- **交互模式**列表示该模型是否可用于 Label Studio 中的交互式标注：在标注页面上执行操作时查看
  交互式预测。
- **训练**列表示该模型是否可用于 Label Studio 中的训练：根据
  提交的标注来更新模型状态。

| MODEL_NAME                                                                                 | 描述                                                                                                                                          | 预标注 | 交互模式 | 训练 |  必需参数  | 任意标签或固定标签？                                                   |
|--------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|------------------|----------|----------------------|----------------------------------------------------------------------------|
| [bert_classifier](/label_studio_ml/examples/bert_classifier)                               | 使用 [Huggingface](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#automodelforsequenceclassification) 进行文本分类            | ✅              | ❌                | ✅        | 无                       | 任意|
| [docling](/label_studio_ml/examples/docling)                                               | 通过 [IBM Docling SaaS](https://www.ibm.com/products/docling) (`DoclingServiceClient`) 进行布局分析 → ReactCode 区域                                      | ✅              | ❌                | ❌        | DOCLING_SERVICE_URL, DOCLING_SERVE_API_KEY, LABEL_STUDIO_URL (uploads) | 固定 (布局类别) |
| [easyocr](/label_studio_ml/examples/easyocr)                                               | 自动 OCR。[EasyOCR](https://github.com/JaidedAI/EasyOCR)                                                                                        | ✅              | ❌                | ❌        | 无                       | 固定 (字符)                                                           |
| [flair](/label_studio_ml/examples/flair)                                                   | 使用 [flair](https://flairnlp.github.io/) 进行 NER                                                                                                          | ✅              | ❌                | ❌        | 无                       | 任意|
| [gliner](/label_studio_ml/examples/gliner)                                                 | 使用 [GLiNER](https://huggingface.co/spaces/tomaarsen/gliner_medium-v2.1) 进行 NER                                                                          | ❌              |  ✅  |  ✅  | 无 | 任意|
| [grounding_dino](/label_studio_ml/examples/grounding_dino)                                 | 使用提示词进行目标检测。[详情](https://github.com/IDEA-Research/GroundingDINO)                                                             | ❌              | ✅                | ❌        | 无                       | 任意                                                                  |
| [grounding_sam](/label_studio_ml/examples/grounding_sam) | 使用 [提示词](https://github.com/IDEA-Research/GroundingDINO) 和 [SAM 2](https://github.com/facebookresearch/segment-anything-2) 进行目标检测 |    ❌              | ✅                | ❌        | 无                       | 任意                                                                  |
| [huggingface_llm](/label_studio_ml/examples/huggingface_llm)                               | 使用 [Hugging Face](https://huggingface.co/tasks/text-generation) 进行 LLM 推理                                                                      | ✅              | ❌                | ❌        | 无                       | 任意 |
| [huggingface_ner](/label_studio_ml/examples/huggingface_ner)                               | 使用 [Hugging Face](https://huggingface.co/docs/transformers/en/tasks/token_classification) 进行 NER                                                        | ✅              | ❌                | ✅        | 无                       | 任意 |
| [interactive_substring_matching](/label_studio_ml/examples/interactive_substring_matching) | 简单的关键词搜索                                                                                                                               | ❌              | ✅                | ❌        | 无                       | 任意 |
| [langchain_search_agent](/label_studio_ml/examples/langchain_search_agent)                 | 使用 Google 搜索和 [Langchain](https://langchain.com/) 的 RAG 管道                                                                              | ✅              | ✅                | ✅        | OPENAI_API_KEY, GOOGLE_CSE_ID, GOOGLE_API_KEY | 任意 |
| [llm_interactive](/label_studio_ml/examples/llm_interactive)                               | 使用 [OpenAI](https://platform.openai.com/)、Azure LLM 进行提示工程。                                                                          | ✅              | ✅                | ✅        | OPENAI_API_KEY             | 任意                                                                  |
| [mmdetection](/label_studio_ml/examples/mmdetection-3)                                     | 使用 [OpenMMLab](https://github.com/open-mmlab/mmdetection) 进行目标检测                                                                         | ✅              | ❌                | ❌        | 无                       | 任意 |
| [nemo_asr](/label_studio_ml/examples/nemo_asr)                                             | 使用 [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) 进行语音 ASR                                                                                          | ✅              | ❌                | ❌        | 无                       | 固定 (词汇和字符) |
| [paddleocr](/label_studio_ml/examples/paddleocr)                                           | 使用 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (PP-OCRv5) 进行 OCR                                                                           | ✅              | ❌                | ❌        | 无                       | 固定 (字符)                                                           |
| [segment_anything_2_image](/label_studio_ml/examples/segment_anything_2_image)             | 使用 [SAM 2](https://github.com/facebookresearch/segment-anything-2) 进行图像分割                                                              | ❌              | ✅ | ❌ | 无| 任意|
| [segment_anything_model](/label_studio_ml/examples/segment_anything_model)                 | 使用 [Meta](https://segment-anything.com/) 进行图像分割                                                                                          | ❌              | ✅                |   ❌       | 无                       | 任意                                                                  |
| [sklearn_text_classifier](/label_studio_ml/examples/sklearn_text_classifier)               | 使用 [scikit-learn](https://scikit-learn.org/stable/) 进行文本分类                                                                            | ✅              | ❌                | ✅        | 无                        | 任意 |
| [spacy](/label_studio_ml/examples/spacy)                                                   | 使用 [SpaCy](https://spacy.io/) 进行 NER                                                                                                                    | ✅              | ❌                | ❌        | 无                       | 固定      [(查看文档)](https://spacy.io/usage/linguistic-features) |
| [tesseract](/label_studio_ml/examples/tesseract)                                           | 交互式 OCR。[详情](https://github.com/tesseract-ocr/tesseract)                                                                               | ❌              | ✅                | ❌        | 无                       | 固定 (字符)                                                           |
| [timeseries_segmenter](/label_studio_ml/examples/timeseries_segmenter)             | 使用小型 LSTM 网络进行时间序列分割 | ✅              | ✅                | ✅        | 无   | 固定 |
| [watsonX](/label_studio_ml/exampels/watsonx)| 使用 [WatsonX](https://www.ibm.com/products/watsonx-ai) 进行 LLM 推理并集成 [WatsonX.data](watsonx.data)| ✅ | ✅| ❌ | 无| 任意|
| [yolo](/label_studio_ml/examples/yolo)                                                     | 支持所有 YOLO 任务：[YOLO](https://docs.ultralytics.com/tasks/) | ✅ | ❌ | ❌ | 无 | 任意 |

# (高级用法) 开发您自己的模型

要开始开发自己的 ML 后端，请按照以下说明操作。

## 1. 安装

从仓库下载并安装 `label-studio-ml`：

```bash
git clone https://github.com/HumanSignal/label-studio-ml-backend.git
cd label-studio-ml-backend/
pip install -e .
```

## 2. 创建空的 ML 后端：

```bash
label-studio-ml create my_ml_backend
```

您可以进入 `my_ml_backend` 目录并修改代码以实现自己的推理逻辑。

目录结构应如下所示：

```
my_ml_backend/
├── Dockerfile
├── docker-compose.yml
├── model.py
├── _wsgi.py
├── README.md
└── requirements.txt
```

`Dockefile` 和 `docker-compose.yml` 用于通过 Docker 运行 ML 后端。
`model.py` 是主文件，您可以在其中实现自己的训练和推理逻辑。
`_wsgi.py` 是一个辅助文件，用于通过 Docker 运行 ML 后端（您无需修改它）。
`README.md` 是一个 readme 文件，包含有关如何运行 ML 后端的说明。
`requirements.txt` 是一个包含 Python 依赖项的文件。

## 3. 实现预测逻辑

在您的模型目录中，找到 `model.py` 文件（例如，`my_ml_backend/model.py`）。

`model.py` 文件包含一个继承自 `LabelStudioMLBase` 的类声明。此类提供了 API 方法的包装器，这些方法由 Label Studio 用于与 ML 后端通信。您可以覆盖这些方法来
实现自己的逻辑：

```python
def predict(self, tasks, context, **kwargs):
    """Make predictions for the tasks."""
    return predictions
```

`predict` 方法用于对任务进行预测。它使用以下内容：

- `tasks`：[Label Studio 以 JSON 格式表示的任务](https://labelstud.io/guide/task_format.html)
- `context`：[Label Studio 以 JSON 格式表示的上下文](https://labelstud.io/guide/ml_create#Support-interactive-pre-annotations-in-your-ML-backend) - 用于
  交互式标注场景
- `predictions`：[以 JSON 格式表示的预测数组](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-tasks)

一旦您实现了 `predict` 方法，就可以在 Label Studio 中查看来自已连接 ML 后端的预测。

## 4. 实现训练逻辑（可选）

您还可以实现 `fit` 方法来训练您的模型。`fit` 方法通常用于在
标注数据上训练模型，但它可以用于任何需要数据持久化的任意操作（例如，
将标注数据存储在数据库中、保存模型权重、保留 LLM 提示历史记录等）。

默认情况下，在 Label Studio 中的任何数据操作（如创建新任务或更新
标注）时都会调用 `fit` 方法。您可以在项目设置中的 **Webhooks** 下修改此行为。

要实现 `fit` 方法，您需要覆盖 `model.py` 文件中的 `fit` 方法：

```python
def fit(self, event, data, **kwargs):
    """Train the model on the labeled data."""
    old_model = self.get('old_model')
    # 编写您的逻辑来更新模型
    self.set('new_model', new_model)
```

其中

- `event`：事件类型可以是 `'ANNOTATION_CREATED'`、`'ANNOTATION_UPDATED'` 等。
- `data` 是从事件中接收到的有效负载（查看更多
  关于 [Webhook 事件参考](https://labelstud.io/guide/webhook_reference.html)）

此外，还有两个辅助方法可用于在 ML 后端中存储和检索数据：

- `self.set(key, value)` - 在 ML 后端中存储数据
- `self.get(key)` - 从 ML 后端中检索数据

这两种方法都可以在 ML 后端代码的其他地方使用，例如，在 `predict` 方法中获取新的模型
权重。

### 其他方法和参数

`LabelStudioMLBase` 类中还有其他可用的方法和参数：

- `self.label_config` - 以 XML 字符串形式返回 [Label Studio 标注配置](https://labelstud.io/guide/setup.html)。
- `self.parsed_label_config` - 以 JSON 形式返回 [Label Studio 标注配置](https://labelstud.io/guide/setup.html)。
- `self.model_version` - 返回当前模型版本。
- `self.get_local_path(url, task_id)` - 此辅助函数用于下载和缓存通常存储在 `task['data']` 中的 URL，
并返回其本地路径。该 URL 可以是：LS 上传的文件、LS 本地存储、LS 云存储或任何其他 http(s) URL。

### 不使用 Docker 运行

要不使用 Docker 运行（例如，用于调试目的），您可以使用以下命令：

```bash
label-studio-ml start my_ml_backend
```

### 测试您的 ML 后端

修改 `my_ml_backend/test_api.py` 以确保您的 ML 后端按预期工作。

### 修改端口

要修改端口，请使用 `-p` 参数：

```bash
label-studio-ml start my_ml_backend -p 9091
```

# 将您的 ML 后端部署到 GCP

开始之前：

1. 安装 [gcloud](https://cloud.google.com/sdk/docs/install)。
2. 如果您的账户未 [激活](https://console.cloud.google.com/project/_/billing/enable) 计费，请初始化计费。
3. 初始化 gcloud，输入以下命令并在浏览器中登录：

```bash
gcloud auth login
```

4. 激活您的 Cloud Build API。
5. 找到您的 GCP 项目 ID。
6.（可选）将 `GCP_REGION` 作为默认区域添加到您的 ENV 变量中。

开始部署：

1. 创建您自己的 ML 后端
2. 开始部署到 GCP：

```bash
label-studio-ml deploy gcp {ml-backend-local-dir} \
--from={model-python-script} \
--gcp-project-id {gcp-project-id} \
--label-studio-host {https://app.heartex.com} \
--label-studio-api-key {YOUR-LABEL-STUDIO-API-KEY}
```

3. Label Studio 部署模型后，您可以在控制台中找到模型端点。


# 故障排除

## Windows 上的 Docker 构建故障排除

如果您在 Windows 上运行 `docker-compose up --build` 时遇到类似以下错误：

```
exec /app/start.sh : No such file or directory
exited with code 1
```

此问题很可能是由 Windows 处理文本文件中行尾的方式引起的，这可能会影响
如 `start.sh` 之类的脚本。要解决此问题，请按照以下步骤操作：

### 步骤 1：调整 Git 配置

在克隆仓库之前，请确保您的 Git 配置为在检出文件时不会自动将行尾转换为
Windows 样式 (CRLF)。这可以通过将 `core.autocrlf` 设置为 `false` 来实现。打开 Git Bash
或您首选的终端并执行以下命令：

```
git config --global core.autocrlf false
```

### 步骤 2：重新克隆仓库

如果您在调整 Git 配置之前已经克隆了仓库，则需要重新克隆它以
确保行尾被正确保留：

1. **删除现有的本地仓库。** 确保您已备份任何更改或正在进行的工作。
2. **重新克隆仓库。** 使用标准的 Git clone 命令将仓库克隆到您的本地计算机。

### 步骤 3：构建并运行 Docker 容器

导航到克隆仓库中包含 Dockerfile
和 `docker-compose.yml` 的相应目录。然后，继续使用 Docker 命令：

1. **构建 Docker 容器：** 运行 `docker-compose build` 以基于 `docker-compose.yml` 中指定的配置
   构建 Docker 容器。

2. **启动 Docker 容器：** 构建过程完成后，使用 `docker-compose up` 启动容器。

### 附加说明

- 此解决方案专门针对在 Windows 上由于自动转换行尾而遇到的问题。如果您
  使用的是其他操作系统，此解决方案可能不适用。
- 请记住检查您项目的 `.gitattributes` 文件（如果存在），因为它也会影响 Git 如何处理的
  行尾。

通过执行这些步骤，您应该能够解决与 Docker 在 Windows 上无法识别 `start.sh` 脚本
由于行尾转换相关的问题。


## Docker 镜像中的 Pip 缓存重置故障排除

有时，您希望重置 pip 缓存以确保安装最新版本的依赖项。
例如，Label Studio ML 后端库在 requirements.txt 中用作
`label-studio-ml @ git+https://github.com/HumanSignal/label-studio-ml-backend.git`。假设它
已更新，并且您希望在带有 ML 模型的 docker 镜像中使用最新版本。

您可以使用以下命令从头开始重新构建 docker 镜像：

```bash
docker compose build --no-cache
```

## `Bad Gateway` 和 `Service Unavailable` 错误故障排除

如果您发送多个并发请求，可能会看到这些错误。

请注意，提供的 ML 后端示例以开发模式提供，不支持生产级推理服务。

## ML 后端无法进行简单自动标注或无法看到预测的故障排除

您必须确保 ML 后端可以访问您的 Label Studio 数据。如果不能，您可能会遇到以下问题：

* 服务器日志中出现 `no such file or directory` 错误。
* 在 Label Studio 中加载任务时无法看到预测。
* 您的 ML 后端似乎已正确连接，但似乎无法完成任何任务中的自动标注。

要解决此问题，请确保您已设置 `LABEL_STUDIO_URL` 和 `LABEL_STUDIO_API_KEY` 环境变量。有关更多信息，请参阅 [允许 ML 后端访问 Label Studio 数据](https://labelstud.io/guide/ml#Allow-the-ML-backend-to-access-Label-Studio-data)。
