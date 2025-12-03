import json
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class RitualContinuityDataset:
    def __init__(self, data_path):
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        conversations = item["conversations"]

        # 取出 Question
        question = None
        for conv in conversations:
            if conv.get("from") == "human" and "Question:" in conv["value"]:
                question = conv["value"].split("Question:")[-1].strip()

        # ground_truth
        gt_obj = conversations[-1]
        gt_label = None
        try:
            val = gt_obj["value"]
            if isinstance(val, str):
                gt_label = json.loads(val).get("Ritual continuity level")
            elif isinstance(val, dict):
                gt_label = val.get("Ritual continuity level")
        except Exception:
            gt_label = None

        return {
            "data_id": item.get("data_id", str(idx)),
            "text_inputs": question,
            "image_inputs": {
                "video_path": item["video"][0],
                "fps": 1,
                "max_frames": 180
            },
            "ground_truth": {"Ritual continuity level": gt_label}
        }


class RitualContinuityBenchmark:
    def __init__(self, label_name="Ritual continuity level"):
        self.label_name = label_name
        # Updated to High / Low 2-class
        self.label_map = {"high": 1, "low": 0}
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

    def _label_to_id(self, value):
        value = (value or "").strip().lower()
        return self.label_map.get(value, -1)

    def evaluate(self, results):
        y_true, y_pred, infos = [], [], []
        for entry in results:
            data_id = entry.get("data_id")
            ground_truth = entry.get("ground_truth", {})
            prediction = entry.get("prediction", {})

            gt_label = self._label_to_id(ground_truth.get(self.label_name))
            pred_label = self._label_to_id(prediction.get(self.label_name))
            if gt_label == -1 or pred_label == -1:
                continue
            y_true.append(gt_label)
            y_pred.append(pred_label)
            infos.append({
                "data_id": data_id,
                "ground_truth": ground_truth.get(self.label_name),
                "prediction": prediction.get(self.label_name),
                "correct": gt_label == pred_label
            })

        # 混淆矩陣：1=high, 0=low
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

        metrics = {
            "Label order": ["high", "low"],
            "Confusion Matrix": cm.tolist(),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(
                y_true, y_pred,
                average='binary',
                zero_division=0,
                pos_label=1  # high 為正樣本
            ),
            "Recall": recall_score(
                y_true, y_pred,
                average='binary',
                zero_division=0,
                pos_label=1
            ),
            "F1 Score": f1_score(
                y_true, y_pred,
                average='binary',
                zero_division=0,
                pos_label=1
            ),
        }
        return metrics, infos

    def process_response(self, response):
        try:
            if isinstance(response, dict):
                return response
            elif response.strip().startswith("{"):
                return json.loads(response)
            else:
                import re
                # Updated regex to match only high or low
                match = re.search(r'(high|low)', response.lower())
                if match:
                    return {self.label_name: match.group(1)}
                else:
                    return {self.label_name: None}
        except Exception:
            return {self.label_name: None}
