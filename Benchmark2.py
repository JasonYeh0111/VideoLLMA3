# import json
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from rouge import Rouge
# from bert_score import score as bert_score

# class TextGenEvaluator:
#     def __init__(self, data_path):
#         self.data = []
#         self.model = SentenceTransformer("all-MiniLM-L6-v2")
#         self.rouge = Rouge()
#         # 讀取 jsonl，每行一筆
#         with open(data_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 item = json.loads(line)
#                 self.data.append(item)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         conversations = item["conversations"]
#         # 通常倒數第1個 gpt 是 ground truth，倒數第2個 human 是 prompt
#         prompt = None
#         for conv in reversed(conversations):
#             if conv.get("from") == "human":
#                 prompt = conv["value"]
#                 break
#         gt_obj = conversations[-1]
#         gt_str = ""
#         if isinstance(gt_obj["value"], str):
#             try:
#                 # 若是 json 格式
#                 gt_str = json.loads(gt_obj["value"])
#                 # 若真的是字串，不是 dict 直接抓
#                 if isinstance(gt_str, dict):
#                     # 自動選一個最長字串當 reference
#                     gt_str = max(gt_str.values(), key=lambda x: len(str(x)))
#                 else:
#                     gt_str = str(gt_str)
#             except Exception:
#                 gt_str = gt_obj["value"]
#         else:
#             gt_str = str(gt_obj["value"])
#         return {
#             "data_id": item.get("data_id", str(idx)),
#             "text_inputs": prompt,
#             "image_inputs": {
#                 "video_path": item["video"][0],
#                 "fps": 1,
#                 "max_frames": 180
#             },
#             "ground_truth": gt_str.strip()
#         }

#     def process_response(self, response):
#         # 不管什麼格式，直接 return（純文字即可）
#         if isinstance(response, dict):
#             # 如果模型真的生成 dict，只抓第一個 key
#             val = list(response.values())[0]
#             return val if isinstance(val, str) else str(val)
#         return response.strip()

#     def evaluate(self, results):
#         similarities = []
#         bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [], [], [], []
#         rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
#         bert_scores = []
#         infos = []

#         smooth_fn = SmoothingFunction().method1

#         for entry in results:
#             data_id = entry["data_id"]
#             prediction = entry["prediction"]
#             ground_truth = entry["ground_truth"]

#             # 跳過空值
#             if not prediction or not ground_truth:
#                 continue

#             # Cosine Semantic Similarity
#             pred_embedding = self.model.encode(prediction, convert_to_tensor=True)
#             gt_embedding = self.model.encode(ground_truth, convert_to_tensor=True)
#             similarity = cosine_similarity(
#                 pred_embedding.cpu().numpy().reshape(1, -1),
#                 gt_embedding.cpu().numpy().reshape(1, -1)
#             )[0][0]
#             similarities.append(similarity)

#             # BLEU
#             reference = [ground_truth.split()]
#             candidate = prediction.split()
#             bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smooth_fn)
#             bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn)
#             bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth_fn)
#             bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)
#             bleu1_scores.append(bleu1)
#             bleu2_scores.append(bleu2)
#             bleu3_scores.append(bleu3)
#             bleu4_scores.append(bleu4)

#             # ROUGE
#             rouge_scores = self.rouge.get_scores(prediction, ground_truth)[0]
#             rouge1_scores.append(rouge_scores["rouge-1"]["f"])
#             rouge2_scores.append(rouge_scores["rouge-2"]["f"])
#             rougeL_scores.append(rouge_scores["rouge-l"]["f"])

#             # BERTScore
#             P, R, F1 = bert_score([prediction], [ground_truth], lang="en", rescale_with_baseline=True)
#             bert_scores.append(F1.mean().item())

#             infos.append({
#                 "data_id": data_id,
#                 "ground_truth": ground_truth,
#                 "prediction": prediction,
#                 "similarity_score": similarity,
#                 "bleu1_score": bleu1,
#                 "bleu2_score": bleu2,
#                 "bleu3_score": bleu3,
#                 "bleu4_score": bleu4,
#                 "rouge_1_score": rouge_scores["rouge-1"]["f"],
#                 "rouge_2_score": rouge_scores["rouge-2"]["f"],
#                 "rouge_l_score": rouge_scores["rouge-l"]["f"],
#                 "bert_score": F1.mean().item()
#             })

#         # 平均
#         metrics = {
#             "Average Semantic Similarity": float(np.mean(similarities)) if similarities else None,
#             "Average BLEU-1 Score": float(np.mean(bleu1_scores)) if bleu1_scores else None,
#             "Average BLEU-2 Score": float(np.mean(bleu2_scores)) if bleu2_scores else None,
#             "Average BLEU-3 Score": float(np.mean(bleu3_scores)) if bleu3_scores else None,
#             "Average BLEU-4 Score": float(np.mean(bleu4_scores)) if bleu4_scores else None,
#             "Average ROUGE-1 Score": float(np.mean(rouge1_scores)) if rouge1_scores else None,
#             "Average ROUGE-2 Score": float(np.mean(rouge2_scores)) if rouge2_scores else None,
#             "Average ROUGE-L Score": float(np.mean(rougeL_scores)) if rougeL_scores else None,
#             "Average BERTScore": float(np.mean(bert_scores)) if bert_scores else None
#         }

#         return metrics, infos

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from bert_score import score as bert_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import re


class TextGenEvaluator:
    """
    專門評估：
    - Streamer Emotion: angry / happy / sad  (三分類)
    - Emotion Intensity: high / medium / low (三分類)
    - Emotion Reason: 文字生成品質 (semantic similarity, BLEU, ROUGE, BERTScore)
    """

    def __init__(self, data_path):
        self.data = []
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.rouge = Rouge()

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        conversations = item["conversations"]

        # 取最後一個 human (有 Question 的 prompt)
        prompt = None
        for conv in reversed(conversations):
            if conv.get("from") == "human":
                prompt = conv["value"]
                break

        # ground truth 是最後一個 gpt 的 value
        gt_obj = conversations[-1]
        gt_val = gt_obj.get("value", "")
        gt_emo = None
        gt_reason = None
        gt_intensity = None

        # value 可能是 string 裡面又包 JSON，要拆兩層
        if isinstance(gt_val, str):
            try:
                parsed = json.loads(gt_val)
                if isinstance(parsed, dict):
                    gt_emo = parsed.get("Streamer Emotion")
                    gt_reason = parsed.get("Emotion Reason")
                    gt_intensity = parsed.get("Emotion Intensity")
                else:
                    # 不是 dict，就當成整塊 reason 字串
                    gt_reason = str(parsed)
            except Exception:
                # parse 失敗就當成整段文字當 reason
                gt_reason = gt_val
        elif isinstance(gt_val, dict):
            gt_emo = gt_val.get("Streamer Emotion")
            gt_reason = gt_val.get("Emotion Reason")
            gt_intensity = gt_val.get("Emotion Intensity")
        else:
            gt_reason = str(gt_val)

        return {
            "data_id": item.get("data_id", str(idx)),
            "text_inputs": prompt,
            "image_inputs": {
                "video_path": item["video"][0],
                "fps": 1,
                "max_frames": 180
            },
            "ground_truth": {
                "Streamer Emotion": (gt_emo or "").strip() if isinstance(gt_emo, str) else gt_emo,
                "Emotion Reason": (gt_reason or "").strip() if isinstance(gt_reason, str) else gt_reason,
                "Emotion Intensity": (gt_intensity or "").strip() if isinstance(gt_intensity, str) else gt_intensity,
            }
        }

    # -----------------------
    # 解析模型輸出 response
    # -----------------------
    def process_response(self, response):
        """
        將模型輸出（string 或 dict）統一轉成：
        {
            "Streamer Emotion": "...",
            "Emotion Reason": "...",
            "Emotion Intensity": "..."
        }
        """
        # 如果本來就是 dict
        if isinstance(response, dict):
            emo = response.get("Streamer Emotion")
            reason = response.get("Emotion Reason")
            intensity = response.get("Emotion Intensity")
            return {
                "Streamer Emotion": (emo or "").strip() if isinstance(emo, str) else emo,
                "Emotion Reason": (reason or "").strip() if isinstance(reason, str) else reason,
                "Emotion Intensity": (intensity or "").strip() if isinstance(intensity, str) else intensity,
            }

        if not isinstance(response, str):
            response = str(response)

        text = response.strip()

        # 1) 優先嘗試 JSON parse
        parsed = None
        # 有些模型會輸出前後多餘文字，所以切第一個 '{' 到最後一個 '}' 試試看
        if "{" in text and "}" in text:
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                json_str = text[start:end]
                parsed = json.loads(json_str)
            except Exception:
                parsed = None

        emo = None
        reason = None
        intensity = None

        if isinstance(parsed, dict):
            emo = parsed.get("Streamer Emotion")
            reason = parsed.get("Emotion Reason")
            intensity = parsed.get("Emotion Intensity")
        else:
            # 2) fallback: 用 regex 抓 Emotion / Intensity
            emo_match = re.search(
                r'"Streamer Emotion"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE
            )
            if emo_match:
                emo = emo_match.group(1).strip()

            intensity_match = re.search(
                r'"Emotion Intensity"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE
            )
            if intensity_match:
                intensity = intensity_match.group(1).strip()

            # Reason 比較長，直接抓整段或嘗試中間片段都 ok，這裡簡化處理：
            reason_match = re.search(
                r'"Emotion Reason"\s*:\s*"(.+?)"', text, flags=re.IGNORECASE
            )
            if reason_match:
                reason = reason_match.group(1).strip()
            else:
                # 實在抓不到就把整段 response 當 reason
                reason = text

        return {
            "Streamer Emotion": (emo or "").strip() if isinstance(emo, str) else emo,
            "Emotion Reason": (reason or "").strip() if isinstance(reason, str) else reason,
            "Emotion Intensity": (intensity or "").strip() if isinstance(intensity, str) else intensity,
        }

    # -----------------------
    # Label 映射
    # -----------------------
    def _emo_to_id(self, value):
        """
        Streamer Emotion: angry / happy / sad
        """
        if not isinstance(value, str):
            return -1
        v = value.strip().lower()
        mapping = {"angry": 0, "happy": 1, "sad": 2}
        return mapping.get(v, -1)

    def _intensity_to_id(self, value):
        """
        Emotion Intensity: low / medium / high
        """
        if not isinstance(value, str):
            return -1
        v = value.strip().lower()
        mapping = {"low": 0, "medium": 1, "high": 2}
        return mapping.get(v, -1)

    # -----------------------
    # 主評估函式
    # -----------------------
    def evaluate(self, results):
        """
        results: list of dict, 每個元素格式：
        {
            "data_id": ...,
            "prediction": <dict from process_response>,
            "ground_truth": {
                "Streamer Emotion": ...,
                "Emotion Reason": ...,
                "Emotion Intensity": ...
            },
            ...
        }
        """

        # ===== 1) 分類指標：Streamer Emotion =====
        emo_y_true, emo_y_pred = [], []

        # ===== 2) 分類指標：Emotion Intensity =====
        int_y_true, int_y_pred = [], []

        # ===== 3) 生成指標：Emotion Reason =====
        similarities = []
        bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [], [], [], []
        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
        bert_scores = []
        reason_infos = []

        smooth_fn = SmoothingFunction().method1

        for entry in results:
            data_id = entry["data_id"]
            gt = entry.get("ground_truth", {}) or {}
            pred = entry.get("prediction", {}) or {}

            # -------- Streamer Emotion (3-class) --------
            gt_emo = gt.get("Streamer Emotion")
            pred_emo = pred.get("Streamer Emotion")
            gt_emo_id = self._emo_to_id(gt_emo)
            pred_emo_id = self._emo_to_id(pred_emo)
            if gt_emo_id != -1 and pred_emo_id != -1:
                emo_y_true.append(gt_emo_id)
                emo_y_pred.append(pred_emo_id)

            # -------- Emotion Intensity (3-class) --------
            gt_int = gt.get("Emotion Intensity")
            pred_int = pred.get("Emotion Intensity")
            gt_int_id = self._intensity_to_id(gt_int)
            pred_int_id = self._intensity_to_id(pred_int)
            if gt_int_id != -1 and pred_int_id != -1:
                int_y_true.append(gt_int_id)
                int_y_pred.append(pred_int_id)

            # -------- Emotion Reason (text generation) --------
            gt_reason = gt.get("Emotion Reason")
            pred_reason = pred.get("Emotion Reason")

            if not isinstance(gt_reason, str) or not isinstance(pred_reason, str):
                continue
            if not gt_reason.strip() or not pred_reason.strip():
                continue

            # Semantic Similarity
            pred_emb = self.model.encode(pred_reason, convert_to_tensor=True)
            gt_emb = self.model.encode(gt_reason, convert_to_tensor=True)
            sim = cosine_similarity(
                pred_emb.cpu().numpy().reshape(1, -1),
                gt_emb.cpu().numpy().reshape(1, -1)
            )[0][0]
            similarities.append(sim)

            # BLEU
            reference = [gt_reason.split()]
            candidate = pred_reason.split()
            bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smooth_fn)
            bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn)
            bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth_fn)
            bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)
            bleu1_scores.append(bleu1)
            bleu2_scores.append(bleu2)
            bleu3_scores.append(bleu3)
            bleu4_scores.append(bleu4)

            # ROUGE
            rouge_scores = self.rouge.get_scores(pred_reason, gt_reason)[0]
            rouge1_scores.append(rouge_scores["rouge-1"]["f"])
            rouge2_scores.append(rouge_scores["rouge-2"]["f"])
            rougeL_scores.append(rouge_scores["rouge-l"]["f"])

            # BERTScore
            P, R, F1 = bert_score([pred_reason], [gt_reason], lang="en", rescale_with_baseline=True)
            bert_scores.append(F1.mean().item())

            reason_infos.append({
                "data_id": data_id,
                "ground_truth": gt_reason,
                "prediction": pred_reason,
                "similarity_score": sim,
                "bleu1_score": bleu1,
                "bleu2_score": bleu2,
                "bleu3_score": bleu3,
                "bleu4_score": bleu4,
                "rouge_1_score": rouge_scores["rouge-1"]["f"],
                "rouge_2_score": rouge_scores["rouge-2"]["f"],
                "rouge_l_score": rouge_scores["rouge-l"]["f"],
                "bert_score": F1.mean().item()
            })

        # ===== 整理指標 =====
        metrics = {}

        # Streamer Emotion
        if emo_y_true:
            emo_cm = confusion_matrix(emo_y_true, emo_y_pred, labels=[0, 1, 2])
            metrics["Streamer Emotion"] = {
                "Label order": ["angry", "happy", "sad"],
                "Confusion Matrix": emo_cm.tolist(),
                "Accuracy": accuracy_score(emo_y_true, emo_y_pred),
                "Precision": precision_score(emo_y_true, emo_y_pred, average='macro', zero_division=0),
                "Recall": recall_score(emo_y_true, emo_y_pred, average='macro', zero_division=0),
                "F1 Score": f1_score(emo_y_true, emo_y_pred, average='macro', zero_division=0),
            }
        else:
            metrics["Streamer Emotion"] = None

        # Emotion Intensity
        if int_y_true:
            int_cm = confusion_matrix(int_y_true, int_y_pred, labels=[0, 1, 2])
            metrics["Emotion Intensity"] = {
                "Label order": ["low", "medium", "high"],
                "Confusion Matrix": int_cm.tolist(),
                "Accuracy": accuracy_score(int_y_true, int_y_pred),
                "Precision": precision_score(int_y_true, int_y_pred, average='macro', zero_division=0),
                "Recall": recall_score(int_y_true, int_y_pred, average='macro', zero_division=0),
                "F1 Score": f1_score(int_y_true, int_y_pred, average='macro', zero_division=0),
            }
        else:
            metrics["Emotion Intensity"] = None

        # Emotion Reason text generation
        if similarities:
            metrics["Emotion Reason"] = {
                "Average Semantic Similarity": float(np.mean(similarities)),
                "Average BLEU-1 Score": float(np.mean(bleu1_scores)),
                "Average BLEU-2 Score": float(np.mean(bleu2_scores)),
                "Average BLEU-3 Score": float(np.mean(bleu3_scores)),
                "Average BLEU-4 Score": float(np.mean(bleu4_scores)),
                "Average ROUGE-1 Score": float(np.mean(rouge1_scores)),
                "Average ROUGE-2 Score": float(np.mean(rouge2_scores)),
                "Average ROUGE-L Score": float(np.mean(rougeL_scores)),
                "Average BERTScore": float(np.mean(bert_scores)),
            }
        else:
            metrics["Emotion Reason"] = None

        return metrics, reason_infos
