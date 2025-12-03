import json
import re
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ================================
# 📘 Dataset
# ================================
class EEGroupSolidarityDataset:
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

        # 選擇 Question
        question = None
        for conv in conversations:
            if conv.get("from") == "human" and "Question:" in conv["value"]:
                question = conv["value"].split("Question:")[-1].strip()
                break

        gt_obj = conversations[-1]
        gt_label = {}
        try:
            val = gt_obj["value"]
            if isinstance(val, str):
                gt_label = json.loads(val)
            elif isinstance(val, dict):
                gt_label = val
        except Exception:
            gt_label = {}

        return {
            "data_id": item.get("data_id", str(idx)),
            "text_inputs": question,
            "image_inputs": {
                "video_path": item["video"][0],
                "fps": 1,
                "max_frames": 180
            },
            "ground_truth": gt_label
        }

# ================================
# 🧠 Benchmark
# ================================
class EEGroupSolidarityBenchmark:
    def __init__(self):
        # 每個欄位自己的分類設定，基於需求已調整
        self.metric_cfg = {
            "Emotional Energy (EE)": {
                "labels": ["low", "medium", "high"],
                "map": {"low": 0, "medium": 1, "high": 2}
            },
            "Group Solidarity Participation (GSp)": {
                "labels": ["low", "medium", "high"],
                "map": {"low": 0, "medium": 1, "high": 2}
            },
            "Group Solidarity Cohesion (GSc)": {  # updated
                "labels": ["low", "medium", "high"],
                "map": {"low": 0, "medium": 1, "high": 2}
            },
            "Group Symbols (GSy)": {
                "labels": ["low", "high"],
                "map": {"low": 0, "high": 1}
            },
        }

    def _label_to_id(self, value, field):
        """將字串類別轉 mapping ID"""
        cfg = self.metric_cfg[field]
        value = (value or "").strip().lower()
        return cfg["map"].get(value, -1)

    def evaluate(self, results):
        metrics = {}
        infos = {field: [] for field in self.metric_cfg}

        for field in self.metric_cfg:
            y_true, y_pred = [], []
            for entry in results:
                ground_truth = entry.get("ground_truth", {})
                prediction = entry.get("prediction", {})
                gt_label = self._label_to_id(ground_truth.get(field), field)
                pred_label = self._label_to_id(prediction.get(field), field)

                if gt_label == -1 or pred_label == -1:
                    continue

                y_true.append(gt_label)
                y_pred.append(pred_label)

                infos[field].append({
                    "data_id": entry.get("data_id"),
                    "ground_truth": ground_truth.get(field),
                    "prediction": prediction.get(field),
                    "correct": gt_label == pred_label
                })

            labels = list(range(len(self.metric_cfg[field]["labels"])))
            metrics[field] = {
                "Label order": self.metric_cfg[field]["labels"],
                "Confusion Matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
                "Recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
                "F1 Score": f1_score(y_true, y_pred, average='macro', zero_division=0),
            }

        return metrics, infos

    def process_response(self, response):
        """解析模型輸出為結構化 dict 格式"""
        try:
            if isinstance(response, dict):  # 模型有時會直接輸出 dict
                return response
            elif response.strip().startswith("{"):  # JSON 格式
                return json.loads(response)
            else:
                # 非 JSON 格式，使用 regex 抽取
                result = {}
                for field in self.metric_cfg:
                    pattern = field.split("(")[0].strip()  # ex: "Emotional Energy"
                    # 支援 low / medium / high
                    m = re.search(
                        rf"{pattern}.*?:.*?(low|medium|high)",
                        response.lower()
                    )
                    if m:
                        result[field] = m.group(1)
                return result
        except Exception:
            return {}
