import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from Benchmark3 import ProfitabilityEvaluator
import json
import os
import numpy as np

# 設置模型路徑
model_path = "../finetuneModel3"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": "cuda:0"},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

@torch.inference_mode()
def infer(conversation):
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=1024, pad_token_id=151643)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response

# 初始化 ProfitabilityEvaluator
data_root = "/workspace/data_root/eval3.jsonl"
evaluator = ProfitabilityEvaluator(data_root)

# 存儲結果
results = []

for entry in evaluator.data:
    data_id = entry["data_id"]
    text_inputs = entry["text_inputs"]
    image_inputs = entry["image_inputs"]
    ground_truth = entry["ground_truth"]
    print(data_id)

    assert isinstance(image_inputs, dict), f"image_inputs 應該是 dict, 但得到 {type(image_inputs)}"

    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": image_inputs},
                {"type": "text", "text": text_inputs},
            ]
        },
    ]
    
    # **模型推理**
    with torch.no_grad():
        response = infer(conversation)
        print(response)

    # 解析回應
    prediction = evaluator.process_response(response)

    # 收集結果
    results.append({
        "data_id": data_id,
        "response": response,
        "prediction": prediction,
        "ground_truth": ground_truth
    })

# **執行相似度評估**
metrics, infos = evaluator.evaluate(results)

output_file = "./result/finetune/stage3result.json"

metrics = {k: float(v) if isinstance(v, np.float32) else v for k, v in metrics.items()}

# **輸出結果到 JSON 檔案**
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({"results": results, "metrics": metrics}, f, indent=4, ensure_ascii=False)


# **輸出結果到 JSON 檔案**
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({"results": results, "metrics": metrics}, f, indent=4, ensure_ascii=False)

print(f"Results saved to {os.path.abspath(output_file)}")

# **輸出相似度評估結果**
print(json.dumps(metrics, indent=4))
