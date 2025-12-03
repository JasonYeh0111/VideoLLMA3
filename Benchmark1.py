# import json
# import os
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# class SentimentSimilarityDataset:
#     def __init__(self, data_path):
#         self.data = []
#         with open(data_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 self.data.append(json.loads(line))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         # 取 Question
#         conversations = item["conversations"]
#         # 拿最後一個 human 有 Question，gpt 那一個是 ground truth
#         question = None
#         for conv in conversations:
#             if conv.get("from") == "human":
#                 if "Question:" in conv["value"]:
#                     # 通常這就是你的 prompt
#                     question = conv["value"].split("Question:")[-1].strip()
#         # ground_truth 格式
#         gt_obj = conversations[-1]
#         gt_label = None
#         try:
#             # 可能是 {"Sentiment Similarity Level": "high"}
#             val = gt_obj["value"]
#             if isinstance(val, str):
#                 gt_label = json.loads(val).get("Sentiment Similarity Level")
#             elif isinstance(val, dict):
#                 gt_label = val.get("Sentiment Similarity Level")
#         except Exception:
#             gt_label = None
#         # 返回統一格式
#         return {
#             "data_id": item.get("data_id", str(idx)),
#             "text_inputs": question,
#             "image_inputs": {
#                 "video_path": item["video"][0],
#                 "fps": 1,
#                 "max_frames": 180
#             },
#             "ground_truth": {"Sentiment Similarity Level": gt_label}
#         }

# class SentimentSimilarityBenchmark:
#     def __init__(self, label_name="Sentiment Similarity Level"):
#         self.label_name = label_name
#         self.label_map = {"high": 2, "medium": 1, "low": 0}
#         self.inv_label_map = {v: k for k, v in self.label_map.items()}

#     def _label_to_id(self, value):
#         value = (value or "").strip().lower()
#         return self.label_map.get(value, -1)

#     def evaluate(self, results):
#         y_true, y_pred, infos = [], [], []
#         for entry in results:
#             data_id = entry.get("data_id")
#             ground_truth = entry.get("ground_truth", {})
#             prediction = entry.get("prediction", {})

#             gt_label = self._label_to_id(ground_truth.get(self.label_name))
#             pred_label = self._label_to_id(prediction.get(self.label_name))
#             if gt_label == -1 or pred_label == -1:
#                 continue
#             y_true.append(gt_label)
#             y_pred.append(pred_label)
#             infos.append({
#                 "data_id": data_id,
#                 "ground_truth": ground_truth.get(self.label_name),
#                 "prediction": prediction.get(self.label_name),
#                 "correct": gt_label == pred_label
#             })
#         cm = confusion_matrix(y_true, y_pred, labels=[2, 1, 0])
#         metrics = {
#             "Label order": ["high", "medium", "low"],
#             "Confusion Matrix": cm.tolist(),
#             "Accuracy": accuracy_score(y_true, y_pred),
#             "Precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
#             "Recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
#             "F1 Score": f1_score(y_true, y_pred, average='macro', zero_division=0),
#         }
#         return metrics, infos

#     def process_response(self, response):
#         try:
#             if isinstance(response, dict):
#                 return response
#             elif response.strip().startswith("{"):
#                 return json.loads(response)
#             else:
#                 import re
#                 match = re.search(r'(high|medium|low)', response.lower())
#                 if match:
#                     return {self.label_name: match.group(1)}
#                 else:
#                     return {self.label_name: None}
#         except Exception:
#             return {self.label_name: None}

# import json
# import os
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# import re

# class SentimentSimilarityDataset:
#     def __init__(self, data_path):
#         self.data = []
#         with open(data_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 self.data.append(json.loads(line))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         conversations = item["conversations"]

#         # 1) 抓 Question（最後一個有人寫 Question: 的 human）
#         question = None
#         for conv in conversations:
#             if conv.get("from") == "human" and "Question:" in conv.get("value", ""):
#                 question = conv["value"].split("Question:", 1)[-1].strip()

#         # 2) ground truth = 最後一個 assistant / gpt 的 value
#         gt_obj = conversations[-1]
#         val = gt_obj.get("value", None)
#         gt_label = None

#         if isinstance(val, str):
#             # 你的標註現在就是 "high" / "medium" / "low" 這種字串
#             s = val.strip().lower()
#             if s in {"high", "medium", "low"}:
#                 gt_label = s
#             else:
#                 # 如果未來有 "Sentiment Similarity Level: high" 這種格式，就用 regex 抓
#                 m = re.search(r"(high|medium|low)", s)
#                 if m:
#                     gt_label = m.group(1)
#         elif isinstance(val, dict):
#             # 若之後又換回 JSON 形式，這邊仍然支援
#             gt_label = (val.get("Sentiment Similarity Level") or "").strip().lower() or None

#         return {
#             "data_id": item.get("data_id", str(idx)),
#             "text_inputs": question,
#             "image_inputs": {
#                 "video_path": item["video"][0],
#                 "fps": 1,
#                 "max_frames": 60
#             },
#             "ground_truth": {"Sentiment Similarity Level": gt_label}
#         }


# class SentimentSimilarityBenchmark:
#     def __init__(self, label_name="Sentiment Similarity Level"):
#         self.label_name = label_name
#         self.label_map = {"high": 2, "medium": 1, "low": 0}
#         self.inv_label_map = {v: k for k, v in self.label_map.items()}

#     def _label_to_id(self, value):
#         if value is None:
#             return -1
#         s = str(value).strip().lower()
#         return self.label_map.get(s, -1)

#     def evaluate(self, results):
#         y_true, y_pred, infos = [], [], []

#         for entry in results:
#             data_id = entry.get("data_id")
#             ground_truth = entry.get("ground_truth", {})
#             prediction = entry.get("prediction", {})

#             gt_raw = ground_truth.get(self.label_name)
#             pred_raw = prediction.get(self.label_name)

#             gt_label = self._label_to_id(gt_raw)
#             pred_label = self._label_to_id(pred_raw)

#             # 無法解析的 label 就跳過
#             if gt_label == -1 or pred_label == -1:
#                 continue

#             y_true.append(gt_label)
#             y_pred.append(pred_label)
#             infos.append({
#                 "data_id": data_id,
#                 "ground_truth": gt_raw,
#                 "prediction": pred_raw,
#                 "correct": gt_label == pred_label
#             })

#         cm = confusion_matrix(y_true, y_pred, labels=[2, 1, 0])
#         metrics = {
#             "Label order": ["high", "medium", "low"],
#             "Confusion Matrix": cm.tolist(),
#             "Accuracy": accuracy_score(y_true, y_pred),
#             "Precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
#             "Recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
#             "F1 Score": f1_score(y_true, y_pred, average='macro', zero_division=0),
#         }
#         return metrics, infos

#     def process_response(self, response):
#         """
#         把 model 的 raw output 統一轉成:
#         { "Sentiment Similarity Level": "high/medium/low" }
#         目前你的輸出是單純 'high' / 'medium' / 'low' 字串，
#         這裡會優先處理那種情況。
#         """
#         try:
#             # 如果本來就已經是 dict（例如你自己包好的）
#             if isinstance(response, dict):
#                 return response

#             if not isinstance(response, str):
#                 response = str(response)

#             s = response.strip()

#             # 1) 先處理最常見：直接輸出 'high' / 'medium' / 'low'
#             lower = s.lower()
#             if lower in {"high", "medium", "low"}:
#                 return {self.label_name: lower}

#             # 2) 如果是 JSON 字串
#             if s.startswith("{"):
#                 try:
#                     obj = json.loads(s)
#                     # 如果裡面直接是 {'Sentiment Similarity Level': 'high'}
#                     if self.label_name in obj:
#                         val = obj[self.label_name]
#                     else:
#                         # 或者 {'label': 'high'} 之類，再用 regex 找
#                         txt = json.dumps(obj).lower()
#                         m = re.search(r"(high|medium|low)", txt)
#                         val = m.group(1) if m else None
#                     return {self.label_name: val}
#                 except Exception:
#                     pass

#             # 3) 一般文字，裡面包含 high / medium / low 就抓出來
#             m = re.search(r"(high|medium|low)", lower)
#             if m:
#                 return {self.label_name: m.group(1)}

#             # 4) 完全沒找到
#             return {self.label_name: None}

#         except Exception:
#             return {self.label_name: None}

##### 二分類

import json
import re
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class SentimentSimilarityDataset:
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

        # 取 Question
        question = None
        for conv in conversations:
            if conv.get("from") == "human" and "Question:" in conv.get("value", ""):
                question = conv["value"].split("Question:")[-1].strip()

        # 取 Ground Truth
        gt_obj = conversations[-1]
        gt_label = None
        try:
            val = gt_obj.get("value")
            if isinstance(val, str):
                gt_label = json.loads(val).get("Sentiment Similarity Level")
            elif isinstance(val, dict):
                gt_label = val.get("Sentiment Similarity Level")
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
            "ground_truth": {"Sentiment Similarity Level": gt_label}
        }

class SentimentSimilarityBenchmark:
    def __init__(self, label_name="Sentiment Similarity Level"):
        self.label_name = label_name
        # 二分類：只保留 high 和 low，忽略 medium
        self.label_map = {"high": 1, "low": 0}
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

    def _label_to_id(self, value):
        if value is None:
            return -1
        value = (value or "").strip().lower()
        # 只接受 high / low，medium 或其他都視為無效
        return self.label_map.get(value, -1)

    # def process_response(self, response):
    #     try:
    #         if isinstance(response, dict):
    #             return response
    #         elif response.strip().startswith("{"):
    #             return json.loads(response)
    #         else:
    #             import re
    #             # 僅接受 high 或 low，忽略 medium
    #             match = re.search(r'\b(high|low)\b', response.lower())
    #             if match:
    #                 return {self.label_name: match.group(1)}
    #             else:
    #                 return {self.label_name: None}
    #     except Exception:
    #         return {self.label_name: None}

    def process_response(self, response):

    # 最後一定要回傳這種格式：{ self.label_name: "high" or "low" or None }

        try:
            # 1) 如果本來就是 dict，直接處理
            if isinstance(response, dict):
                candidate = response
            else:
                text = str(response).strip()

                # 2) 先嘗試把它當 JSON 解析（包含「不完整但前半段是合法 JSON」的情況）
                candidate = None
                if text.startswith("{"):
                    # 從後面往前截斷，找到第一個能成功 json.loads 的位置
                    for cut in range(len(text), 0, -1):
                        segment = text[:cut]
                        try:
                            candidate = json.loads(segment)
                            break
                        except Exception:
                            continue

                # 如果完全 parse 不出 JSON，就 regex 硬抓 high/low
                if candidate is None:
                    m = re.search(r'\b(high|low)\b', text.lower())
                    if m:
                        return {self.label_name: m.group(1)}
                    else:
                        return {self.label_name: None}

            # 3) 現在有一個 dict candidate，裡面可能還有巢狀字串 JSON
            val = candidate.get(self.label_name)

            # 4) 連續解開巢狀 JSON：一直試著把字串當 JSON 解析，直到失敗或拿到不是字典
            while isinstance(val, str) and self.label_name in val:
                try:
                    inner = json.loads(val)
                    if isinstance(inner, dict):
                        val = inner.get(self.label_name, val)
                    else:
                        break
                except Exception:
                    break

            # 5) 正常情況：val 就是 "high" 或 "low"
            if isinstance(val, str):
                v = val.strip().lower()
                if v in ("high", "low"):
                    return {self.label_name: v}

            # 6) 如果 val 還是很奇怪，就對整個 candidate 做 regex
            text_all = json.dumps(candidate, ensure_ascii=False)
            m = re.search(r'\b(high|low)\b', text_all.lower())
            if m:
                return {self.label_name: m.group(1)}

            return {self.label_name: None}

        except Exception:
            # 7) 最後防呆：對原始 response 做 regex
            text = str(response)
            m = re.search(r'\b(high|low)\b', text.lower())
            if m:
                return {self.label_name: m.group(1)}
            return {self.label_name: None}


    def evaluate(self, results):
        y_true, y_pred, infos = [], [], []

        for entry in results:
            data_id = entry.get("data_id")
            ground_truth = entry.get("ground_truth", {})
            prediction = entry.get("prediction", {})

            gt_label = self._label_to_id(ground_truth.get(self.label_name))
            pred_label = self._label_to_id(prediction.get(self.label_name))

            # 忽略無效資料（包括 medium）
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

        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])  # High=1, Low=0

        metrics = {
            "Label order": ["high", "low"],
            "Confusion Matrix": cm.tolist(),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average='binary', zero_division=0),
            "Recall": recall_score(y_true, y_pred, average='binary', zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, average='binary', zero_division=0),
        }
        return metrics, infos

