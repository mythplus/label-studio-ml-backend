import json
import os
from typing import Dict, List, Optional

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from openai import OpenAI


class NewModel(LabelStudioMLBase):
    """custom ML Backend model"""

    def setup(self):
        """Configure any parameters of your model here."""
        self.set("model_version", "0.0.1")

    def predict(
        self,
        tasks: List[Dict],
        context: Optional[Dict] = None,
        **kwargs,
    ) -> ModelResponse:
        """Write your inference logic here.

        :param tasks: [Label Studio tasks in JSON format]
            (https://labelstud.io/guide/task_format.html)
        :param context: [Label Studio context in JSON format]
            (https://labelstud.io/guide/ml_create#Implement-prediction-logic)
        :return model_response
            ModelResponse(predictions=predictions) with
            predictions: [predictions array in IsoN format]
            (https://labelstud.io/guide/export.html#Label-studio-JsoN-format-of-annotated-tasks)
        """
        predictions = []
        for task in tasks:
            print(f"Predicting for task: {task}")
            llm_results = self.llm_ner(task["data"]["text"])
            results = []
            for llm_result in llm_results:
                results.append({
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                    "value": {
                        "start": llm_result["value"]["start"],
                        "end": llm_result["value"]["end"],
                        "text": llm_result["value"]["text"],
                        "labels": llm_result["value"]["labels"],
                    },
                })
            predictions.append({
                "result": results,
                "score": 1.0,
                "model_version": "latest",
                "model": "qwen3.7-plus",
            })

        print(f"Predictions: {predictions}")
        print(type(predictions))

        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        """This method is called each time an annotation is created or updated.

        You can run your logic here to update the model and persist it to the cache.
        It is not recommended to perform long-running operations here, as it will
        block the main thread. Instead, consider running a separate process or a
        thread (like RQ worker) to perform the training.

        :param event: event type can be ('ANNOTATION_CREATED',
            'ANNOTATION_UPDATED', 'START TRAINING')
        :param data: the payload received from the event
            (check https://labelstud.io/guide/webhook_reference.html)
        """
        old_data = self.get("my_data")
        old_model_version = self.get("model_version")
        print(f"old data: {old_data}")
        print(f"old model version: {old_model_version}")

        self.set("my_data", "my_new_data_value")
        self.set("model_version", "my_new_model_version")
        print(f'New data: {self.get("my_data")}')
        print(f"New model version: {self.get('model_version')}")
        print("fit() completed successfully.")

    def llm_ner(self, text: str) -> List[dict]:
        """Use LLM to perform NER on the input text.

        Return the results in the format required by Label Studio.
        """
        client = OpenAI(
            api_key="sk-ws-H.REPYRXI.kmny.MEYCIQDu_mMSM__daCXFCwxwNvKfOkWwamUdbmhfF13DnVg_cwIhAIeL34XyrcH4lwJOqGZVKPFzUaAQ29DQtpWHjJ8mhgOf",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        system_prompt = """\
# 任务: 命名实体抽取

你需要从输入文本中抽取实体，仅支持以下4种实体标签:
1. PER: 人物
2. ORG: 组织/机构/公司
3. LOC: 地理位置/地址
4. MISC: 其他不属于以上三类的实体

# 严格输出规则
1. 必须返回标准JSON格式，无任何多余文字、解释、说明
2. 外层必须包含 data 数组
3. 数组中每个元素结构:
{
    "value": {
        "start": 起始索引(数字，从0开始),
        "end": 结束索引(数字，左闭右开),
        "text": "实体文本",
        "labels": ["实体标签"]
    }
}
4. start 是实体在原文中的第一个字符索引
5. end 是实体最后一个字符的索引+1(左闭右开)
6. 无实体时返回 {"data": []}
7. 禁止添加任何注释、描述、markdown、代码块，只输出纯JSON
8. 输出的JSON必须可被 json.loads() 解析

现在对以下文本执行实体抽取:
{{文本}}
"""

        completion = client.chat.completions.create(
            model="qwen3.7-plus",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        )

        result = completion.model_dump_json()
        data = json.loads(result)
        llm_result = data["choices"][0]["message"]["content"]
        json_result = json.loads(llm_result)
        print(f"LLM NER result: {json_result}")
        return json_result["data"]
  