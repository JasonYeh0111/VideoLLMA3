import torch
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from Benchmark2 import ProfitabilityEvaluator
import json
import os


# NOTE: transformers==4.46.3 is recommended for this script
model_path = "../finetuneModel2"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": "cuda:0"},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

#print(model)

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
    output_ids = model.generate(**inputs, max_new_tokens=1024,pad_token_id=151643)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response


# 初始化 MyBenchmark 資料集
data_root = "/workspace/data_root/eval2.jsonl"  # 請替換成你的資料夾路徑
evaluator = ProfitabilityEvaluator(data_root)

# # 你的模型（這裡假設是一個 PyTorch 模型）
# model = torch.load("/path/to/your/model.pth")  # 請替換成你的模型權重
# model.eval()  # 設置為評估模式

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

# **執行評估**
metrics, infos = evaluator.evaluate(results)

output_file = "./result/finetune/stage2Result.json"

# **輸出結果到 JSON 檔案**
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({"results": results, "metrics": metrics}, f, indent=4, ensure_ascii=False)

print(f"Results saved to {os.path.abspath(output_file)}")

# **輸出結果**
print(json.dumps(metrics, indent=4))