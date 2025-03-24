import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from bert_score import score as bert_score

class ProfitabilityEvaluator:
    def __init__(self, data_root):
        self.data = []
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.rouge = Rouge()
        
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
        """評估模型輸出，計算與 ground_truth 的語意相似性、BLEU、ROUGE 和 BERTScore"""
        similarities = []
        bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [], [], [], []
        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
        bert_scores = []
        infos = []
        
        smooth_fn = SmoothingFunction().method1  # 平滑 BLEU 避免零分數

        for entry in results:
            data_id = entry["data_id"]
            prediction = entry["prediction"]
            ground_truth = entry.get("ground_truth", "")

            # 計算語意相似度
            pred_embedding = self.model.encode(prediction, convert_to_tensor=True)
            gt_embedding = self.model.encode(ground_truth, convert_to_tensor=True)
            similarity = cosine_similarity(pred_embedding.cpu().numpy().reshape(1, -1), gt_embedding.cpu().numpy().reshape(1, -1))[0][0]
            similarities.append(similarity)

            # 計算不同 n-gram BLEU 分數
            reference = [ground_truth.split()]
            candidate = prediction.split()
            bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smooth_fn)
            bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn)
            bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth_fn)
            bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)
            
            bleu1_scores.append(bleu1)
            bleu2_scores.append(bleu2)
            bleu3_scores.append(bleu3)
            bleu4_scores.append(bleu4)
            
            # 計算 ROUGE-1, ROUGE-2, ROUGE-L
            rouge_scores = self.rouge.get_scores(prediction, ground_truth)[0]
            rouge1_scores.append(rouge_scores["rouge-1"]["f"])
            rouge2_scores.append(rouge_scores["rouge-2"]["f"])
            rougeL_scores.append(rouge_scores["rouge-l"]["f"])
            
            # 計算 BERTScore
            P, R, F1 = bert_score([prediction], [ground_truth], lang="en", rescale_with_baseline=True)
            bert_scores.append(F1.mean().item())

            infos.append({
                "data_id": data_id,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "similarity_score": similarity,
                "bleu1_score": bleu1,
                "bleu2_score": bleu2,
                "bleu3_score": bleu3,
                "bleu4_score": bleu4,
                "rouge_1_score": rouge_scores["rouge-1"]["f"],
                "rouge_2_score": rouge_scores["rouge-2"]["f"],
                "rouge_l_score": rouge_scores["rouge-l"]["f"],
                "bert_score": F1.mean().item()
            })

        # 計算平均分數
        metrics = {
            "Average Semantic Similarity": np.mean(similarities),
            "Average BLEU-1 Score": np.mean(bleu1_scores),
            "Average BLEU-2 Score": np.mean(bleu2_scores),
            "Average BLEU-3 Score": np.mean(bleu3_scores),
            "Average BLEU-4 Score": np.mean(bleu4_scores),
            "Average ROUGE-1 Score": np.mean(rouge1_scores),
            "Average ROUGE-2 Score": np.mean(rouge2_scores),
            "Average ROUGE-L Score": np.mean(rougeL_scores),
            "Average BERTScore": np.mean(bert_scores)
        }

        return metrics, infos
