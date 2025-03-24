import json
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset

class Benchmark1(Dataset):
    def __init__(self, data_root, processor, num_splits=1, split_idx=0, fps=1, max_frames=180):
        self.data = []
        self.processor = processor
        self.fps = fps
        self.max_frames = max_frames

        # 讀取 JSONL 檔案
        input_file = os.path.join(data_root)
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)

        # 分片數據（如果是多 GPU）
        self.data = self.data[split_idx::num_splits]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return {
            "data_ids": [item["data_id"]],
            "text_inputs": [item["text_inputs"]],
            "image_inputs": item["image_inputs"],
            "ground_truth": json.loads(item["ground_truth"])  # 解析 Ground Truth
        }

    def process_response(self, data_id, response):
        """處理模型回應，確保格式符合 JSON"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"Semantic Similarity Level": None, "Sentiment Consistency Level": None}

    def evaluate(self, results):
        """評估模型輸出，計算混淆矩陣、準確率、精確率、召回率、F1 分數"""

        # 存儲 Ground Truth 和 Prediction
        y_true_similarity, y_pred_similarity = [], []
        y_true_sentiment, y_pred_sentiment = [], []

        infos = []

        for entry in results:
            data_id = entry["data_id"]
            response = entry["response"]
            prediction = entry["prediction"]
            ground_truth = entry.get("ground_truth", {})

            gt_similarity = 1 if ground_truth.get("Semantic Similarity Level") == "high" else 0
            gt_sentiment = 1 if ground_truth.get("Sentiment Consistency Level") == "high" else 0

            pred_similarity = 1 if prediction.get("Semantic Similarity Level") == "high" else 0
            pred_sentiment = 1 if prediction.get("Sentiment Consistency Level") == "high" else 0

            y_true_similarity.append(gt_similarity)
            y_pred_similarity.append(pred_similarity)

            y_true_sentiment.append(gt_sentiment)
            y_pred_sentiment.append(pred_sentiment)

            infos.append({
                "data_id": data_id,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "semantic_correct": gt_similarity == pred_similarity,
                "sentiment_correct": gt_sentiment == pred_sentiment
            })

        # 計算混淆矩陣
        cm_similarity = confusion_matrix(y_true_similarity, y_pred_similarity)
        cm_sentiment = confusion_matrix(y_true_sentiment, y_pred_sentiment)

        # 計算評估指標
        metrics = {
            "Semantic Similarity": {
                "Confusion Matrix": cm_similarity.tolist(),
                "Accuracy": accuracy_score(y_true_similarity, y_pred_similarity),
                "Precision": precision_score(y_true_similarity, y_pred_similarity, zero_division=0),
                "Recall": recall_score(y_true_similarity, y_pred_similarity, zero_division=0),
                "F1 Score": f1_score(y_true_similarity, y_pred_similarity, zero_division=0),
            },
            "Sentiment Consistency": {
                "Confusion Matrix": cm_sentiment.tolist(),
                "Accuracy": accuracy_score(y_true_sentiment, y_pred_sentiment),
                "Precision": precision_score(y_true_sentiment, y_pred_sentiment, zero_division=0),
                "Recall": recall_score(y_true_sentiment, y_pred_sentiment, zero_division=0),
                "F1 Score": f1_score(y_true_sentiment, y_pred_sentiment, zero_division=0),
            }
        }

        return metrics, infos