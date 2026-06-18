import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from label_studio_ml.model import LabelStudioMLBase


class MyTextClassifier(LabelStudioMLBase):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 本地模型文件路径
        self.model_path = os.path.join(self.model_dir, 'model.pkl')
        self.vectorizer_path = os.path.join(self.model_dir, 'vectorizer.pkl')
        
        # 尝试加载已有模型
        self.model = None
        self.vectorizer = None
        self.load_model()

    def load_model(self):
        """加载本地训练好的模型"""
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print(f"✅ 加载本地模型: {self.model_path}")

    def save_model(self):
        """保存模型到本地"""
        os.makedirs(self.model_dir, exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"💾 模型已保存到: {self.model_dir}")

    def predict(self, tasks, context, **kwargs):
        """本地推理（不需要联网）"""
        if self.model is None:
            return []
        
        texts = [task['data']['text'] for task in tasks]
        X = self.vectorizer.transform(texts)
        predictions = self.model.predict(X)
        probs = self.model.predict_proba(X)
        
        results = []
        for pred, prob in zip(predictions, probs):
            results.append({
                'model_version': self.model_version,
                'score': float(max(prob)),
                'result': [{
                    'from_name': 'sentiment',
                    'to_name': 'text',
                    'type': 'choices',
                    'value': {'choices': [str(pred)]}
                }]
            })
        return results

    def fit(self, event, data, **kwargs):
        """
        真正的训练逻辑：
        1. 从 Label Studio 获取标注数据
        2. 训练模型
        3. 保存到本地文件
        """
        if event not in ['ANNOTATION_CREATED', 'ANNOTATION_UPDATED']:
            return
        
        # 从 data 中提取标注数据
        annotations = data.get('annotations', [])
        if len(annotations) < 10:  # 数据太少不训练
            print(f"⏳ 标注数据不足（{len(annotations)}条），暂不训练")
            return
        
        texts = []
        labels = []
        for ann in annotations:
            text = ann['task']['data']['text']
            # 提取人工标注的标签
            for result in ann['result']:
                if result['type'] == 'choices':
                    label = result['value']['choices'][0]
                    texts.append(text)
                    labels.append(label)
        
        print(f"🚀 开始训练，数据量: {len(texts)}")
        
        # 训练
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression()
        self.model.fit(X, labels)
        
        # 保存到本地
        self.save_model()
        print("✅ 训练完成，模型已保存")