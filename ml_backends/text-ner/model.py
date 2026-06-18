from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
import json
from openai import OpenAI


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model for text classification"""

    def setup(self):
        """Configure any parameters of your model here"""
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
            predictions: [predictions array in JSON format]
            (https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        predictions = []
        for task in tasks:
            print(f"Predicting for task: {task}")
            llm_result = self.llm_classify(task["data"]["text"])
            results = [{
                "from_name": "sentiment",
                "to_name": "text",
                "type": "choices",
                "value": {
                    "choices": [llm_result],
                },
            }]
            predictions.append({
                "result": results,
                "score": 1.0,
                "model_version": self.get("model_version"),
                "model": "qwen3.7-plus",
            })

        print(f"Predictions: {predictions}")
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
        print(f"Old data: {old_data}")
        print(f"Old model version: {old_model_version}")

        self.set("my_data", "my_new_data_value")
        self.set("model_version", "my_new_model_version")
        print(f'New data: {self.get("my_data")}')
        print(f"New model version: {self.get('model_version')}")
        print("fit() completed successfully.")

    def llm_classify(self, text: str) -> str:
        """Use LLM to classify the sentiment of the input text.

        Returns the classification result (Positive or Negative).
        """
        client = OpenAI(
            api_key="sk-ws-H.REPYRXI.kmny.MEYCIQDu_mMSM__daCXFCwxwNvKfOkWwamUdbmhfF13DnVg_cwIhAIeL34XyrcH4lwJOqGZVKPFzUaAQ29DQtpWHjJ8mhgOf",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        system_prompt = """\
# Task: Sentiment Classification

You need to classify the sentiment of the input text into one of the following categories:
- Positive
- Negative

# Strict Output Rules
1. Return ONLY the category name, e.g. "Positive" or "Negative"
2. Do NOT add any explanation, description, or extra text
3. The output must be exactly one word: Positive or Negative

Now classify the following text:
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
        classification = data["choices"][0]["message"]["content"].strip()
        print(f"LLM classification result: {classification}")
        return classification
