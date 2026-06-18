from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from ultralytics import YOLO
from label_studio_ml.utils import get_single_tag_keys
from io import BytesIO
from PIL import Image
# Label Studio 访问地址
LS_URL = 'http://localhost:8080'

# Label Studio 的 API Token
LS_API_TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA4ODc3NjIxMSwiaWF0IjoxNzgxNTc2MjExLCJqdGkiOiI1NjNhZWZiOWQxMDY0YjA5YTJhMGM2ODcwNjU5NmJmMiIsInVzZXJfaWQiOjF9.5kV8CVvT1RXheR4AUExWNY-14WWFpfNak9zsFli-r7M'

class NewModel(LabelStudioMLBase):

    """custom ML Backend model"""
    
    def setup(self):

        """Configure any parameters of your model here"""
        self.set("model_version", "0.0.1")
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config,'RectangleLabels','Image')
        self.model = YOLO("yolov8n.pt")
        self.labels = self.model.names

    def predict(self, tasks: List[Dict], context: Optional[Dict]= None, **kwargs)-> ModelResponse:
        # header = {"Authorization":"Token "+ LS_API_TOKEN}
        # image =Image.open(BytesIo(requests.get(

        # LS_URL + task['data']['image']  , headers=header).content))

        url = tasks[0]['data']['image']

        image_path = self.get_local_path(url=url, ls_host=LS_URL, task_id=tasks[0]['id'])

        image = Image.open(image_path)

        original_width, original_height = image.size

        predictions = []
        score = 0
        i = 0

        results = self.model.predict(image, conf=0.5)

        for result in results:
            for i, prediction in enumerate(result.boxes):
                xyxy = prediction.xyxy[0].tolist()
                predictions.append({
                    "id": str(i),
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "type": "rectanglelabels",
                    "score": prediction.conf.item(),
                    "original_width": original_width,
                    "original_height": original_height,
                    "value": {
                        "x": (xyxy[0] / original_width) * 100,
                        "y": (xyxy[1] / original_height) * 100,
                        "width": ((xyxy[2] - xyxy[0]) / original_width) * 100,
                        "height": ((xyxy[3] - xyxy[1]) / original_height) * 100,
                        "rotation": 0,
                        "rectanglelabels": [self.labels[int(prediction.cls.item())]]
                    }
                })
                score += prediction.conf.item()

        print(f"prediction score is {score:.3f}.")

        final_predictions=[{

            "result": predictions,

            "score":score/(i+1),

            "model_version": self.get("model_version")

        }]
        return ModelResponse(predictions=final_predictions)

