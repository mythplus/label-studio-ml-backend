# sklearn-text-classifier

基于 scikit-learn 的本地文本分类 ML Backend 示例。

ml_backends/sklearn-text-classifier/
├── model.py                 # 核心：本地训练 + 推理（~160行）
├── _wsgi.py                 # 服务入口
├── Dockerfile               # 容器构建
├── docker-compose.yml       # Docker 编排
├── requirements.txt         # scikit-learn 依赖
├── requirements-base.txt    # label-studio-ml 基础依赖
├── requirements-test.txt    # 测试依赖
├── label_config.xml         # 标注模板配置
├── test_data.json           # 15 条测试数据
└── README.md                # 使用说明



## 特性

- **完全本地运行**：无需联网，不调用任何外部 API
- **真正的模型训练**：`fit()` 使用标注数据训练 scikit-learn 模型并持久化到本地
- **本地推理**：`predict()` 加载本地 `.pkl` 模型文件进行预测

## 快速开始

### 1. 启动 ML Backend 服务

```bash
cd ml_backends/sklearn-text-classifier
label-studio-ml start . -p 9090
```

或使用 Docker：

```bash
docker-compose up --build
```

### 2. 在 Label Studio 中连接模型

在项目设置 → Machine Learning 中添加：

| 设置项 | 值 |
|--------|-----|
| Model URL | `http://localhost:9090` |
| Use for interactive preannotation | ✅ |
| Use for training | ✅ |

### 3. 导入测试数据

使用 `test_data.json` 导入 15 条测试文本。

### 4. 开始标注并训练

1. 打开标注界面，对文本进行 **正面 / 负面 / 中性** 分类标注
2. 提交至少 **10 条** 标注后，模型会自动触发训练
3. 训练完成后，服务日志会显示：`✅ 训练完成！`
4. 之后打开新任务时，会自动显示预标注结果

## 模型文件

训练好的模型保存在当前目录的 `sklearn_model.pkl` 中，包含：

- `pipeline`: scikit-learn Pipeline（TF-IDF + LogisticRegression）
- `label_map`: 标签到数字索引的映射

## 标注模板

使用 `label_config.xml` 配置项目标注界面：

```xml
<View>
  <Text name="text" value="$text"/>
  <Choices name="sentiment" toName="text" choice="single">
    <Choice value="正面"/>
    <Choice value="负面"/>
    <Choice value="中性"/>
  </Choices>
</View>
```

你可以根据需要修改 `<Choice>` 标签来自定义分类类别。

## 技术细节

- **向量化**：`TfidfVectorizer`（max_features=5000, ngram_range=(1,2)）
- **分类器**：`LogisticRegression`（max_iter=1000）
- **最少训练样本**：10 条（可修改 `MIN_TRAIN_SAMPLES`）
