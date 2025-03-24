import json
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class ProfitabilityEvaluator:
    def __init__(self, data_root):
        self.data = []
        
        # 讀取 JSONL 檔案
        input_file = os.path.join(data_root)
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)

    def process_response(self, response):
        """處理模型回應，確保格式符合 JSON"""
        return response
        
    def evaluate(self, results):
        """評估模型輸出，計算混淆矩陣、準確率、精確率、召回率、F1 分數"""
        
        y_true_profit, y_pred_profit = [], []
        infos = []

        for entry in results:
            data_id = entry["data_id"]
            prediction = entry["prediction"]
            ground_truth = entry.get("ground_truth", {})

            gt_profit = 1 if ground_truth == "high" else 0
            pred_profit = 1 if prediction == "high" else 0

            y_true_profit.append(gt_profit)
            y_pred_profit.append(pred_profit)

            infos.append({
                "data_id": data_id,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "profit_correct": gt_profit == pred_profit
            })

        # 計算混淆矩陣
        cm_profit = confusion_matrix(y_true_profit, y_pred_profit)

        # 計算評估指標
        metrics = {
            "Profitability Prediction": {
                "Confusion Matrix": cm_profit.tolist(),
                "Accuracy": accuracy_score(y_true_profit, y_pred_profit),
                "Precision": precision_score(y_true_profit, y_pred_profit, zero_division=0),
                "Recall": recall_score(y_true_profit, y_pred_profit, zero_division=0),
                "F1 Score": f1_score(y_true_profit, y_pred_profit, zero_division=0),
            }
        }

        return metrics, infos