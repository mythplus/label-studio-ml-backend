import os
import pickle
import logging
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from label_studio_ml.model import LabelStudioMLBase

logger = logging.getLogger(__name__)


class SklearnTextClassifier(LabelStudioMLBase):
    """
    基于 scikit-learn 的本地文本分类模型。
    - predict()：加载本地模型文件进行推理（无需联网）
    - fit()：接收标注数据并训练模型，保存到本地 .pkl 文件
    """

    # 最小训练样本数，低于此数量不触发训练
    MIN_TRAIN_SAMPLES = 10

    def __init__(self, model_dir: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)

        # 模型文件保存路径（默认当前目录）
        self.model_dir = model_dir or os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.model_dir, "sklearn_model.pkl")

        # 加载已有模型（如果有）
        self.model: Optional[Pipeline] = None
        self.label_map: Dict[str, int] = {}
        self._load_model()

    # ------------------------------------------------------------------ #
    # 模型持久化
    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        """从本地磁盘加载训练好的模型。"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    saved = pickle.load(f)
                self.model = saved["pipeline"]
                self.label_map = saved.get("label_map", {})
                logger.info(f"✅ 已加载本地模型: {self.model_path}")
            except Exception as e:
                logger.warning(f"⚠️ 加载模型失败，将使用空模型: {e}")
                self.model = None
        else:
            logger.info("ℹ️ 未找到本地模型，等待训练...")
            self.model = None

    def _save_model(self) -> None:
        """将训练好的模型保存到本地磁盘。"""
        os.makedirs(self.model_dir, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(
                {"pipeline": self.model, "label_map": self.label_map},
                f,
            )
        logger.info(f"💾 模型已保存到: {self.model_path}")

    # ------------------------------------------------------------------ #
    # 预测（本地推理，无需联网）
    # ------------------------------------------------------------------ #

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """
        对传入的任务进行本地推理。

        返回格式必须兼容 Label Studio Predictions API：
        https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-tasks
        """
        if self.model is None:
            logger.warning("⏳ 模型尚未训练，无法提供预测")
            return []

        texts = [task["data"].get("text", "") for task in tasks]

        try:
            predictions = self.model.predict(texts)
            probabilities = self.model.predict_proba(texts)
        except Exception as e:
            logger.error(f"推理失败: {e}")
            return []

        # 反转 label_map 用于查找标签名
        idx_to_label = {v: k for k, v in self.label_map.items()}

        results = []
        for pred_idx, probs in zip(predictions, probabilities):
            label = idx_to_label.get(int(pred_idx), str(pred_idx))
            confidence = float(np.max(probs))

            results.append({
                "model_version": self.model_version,
                "score": confidence,
                "result": [
                    {
                        "from_name": "sentiment",
                        "to_name": "text",
                        "type": "choices",
                        "value": {"choices": [label]},
                    }
                ],
            })

        return results

    # ------------------------------------------------------------------ #
    # 训练
    # ------------------------------------------------------------------ #

    def fit(self, event: str, data: Dict, **kwargs) -> Dict:
        """
        接收 Label Studio 的标注事件，提取数据并训练模型。

        event: 如 'ANNOTATION_CREATED', 'ANNOTATION_UPDATED'
        data:  事件负载，包含 task 和 annotation 信息
        """
        if event not in ("ANNOTATION_CREATED", "ANNOTATION_UPDATED"):
            return {}

        # 1. 提取文本和标签（兼容多种数据格式）
        texts, labels = self._extract_training_data(data)

        if len(texts) < self.MIN_TRAIN_SAMPLES:
            logger.info(f"⏳ 标注数据不足（{len(texts)} 条，需 ≥ {self.MIN_TRAIN_SAMPLES}），暂不训练")
            return {}

        # 2. 训练模型
        logger.info(f"🚀 开始训练，样本数: {len(texts)}, 类别数: {len(set(labels))}")
        self._train(texts, labels)

        return {"model_path": self.model_path}

    def _extract_training_data(self, data: Dict) -> tuple:
        """从 Label Studio 的事件数据中提取文本和标签。"""
        texts = []
        labels = []

        # 方式1：标准 webhook 格式 {task: {...}, annotation: {...}}
        if "annotation" in data and "task" in data:
            annotations = [data]
        # 方式2：批量格式 {annotations: [...]}
        elif "annotations" in data:
            annotations = data["annotations"]
        # 方式3：直接是 annotation
        else:
            annotations = [data]

        for ann in annotations:
            # 从 annotation 中获取 task
            task = ann.get("task", ann)
            text = task.get("data", {}).get("text", "") if isinstance(task, dict) else ""
            if not text:
                continue

            # 提取人工标注的结果
            # annotation.result 或 data.result 可能是列表
            results = ann.get("result", []) if isinstance(ann, dict) else []
            if not results and isinstance(task, dict):
                results = task.get("annotations", [{}])[0].get("result", []) if task.get("annotations") else []

            for r in results:
                if r.get("type") == "choices":
                    chosen = r.get("value", {}).get("choices", [])
                    if chosen:
                        labels.append(str(chosen[0]))
                        texts.append(text)
                        break

        return texts, labels

    def _train(self, texts: List[str], labels: List[str]) -> None:
        """使用 scikit-learn 训练文本分类管道并保存。"""
        # 构建标签映射
        unique_labels = sorted(set(labels))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        numeric_labels = [self.label_map[l] for l in labels]

        # 构建管道：TF-IDF + LogisticRegression
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, n_jobs=-1)),
        ])

        self.model.fit(texts, numeric_labels)
        self._save_model()
        logger.info(f"✅ 训练完成！类别: {unique_labels}")
