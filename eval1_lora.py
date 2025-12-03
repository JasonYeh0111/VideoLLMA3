from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
import torch
from Benchmark1 import SentimentSimilarityDataset, SentimentSimilarityBenchmark
import os
import json
from tqdm import tqdm

# ===============================
# 1. 路徑設定（請填入你的）
# ===============================
base_model_path = "/workspace/videollama3_2b_local"  # 原始模型
lora_path = "/workspace/videollama3_qwen2.5_2b/stage_4_lora"  # 你的 LoRA checkpoint（含 adapter_model.bin）
non_lora_path = "/workspace/videollama3_qwen2.5_2b/stage_4_lora/non_lora_trainables.bin"  # projector/encoder 微調（若存在則載入）

device = "cuda:0"

# ===============================
# 2. 載入 Base Model
# ===============================
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

processor = AutoProcessor.from_pretrained(
    base_model_path,
    trust_remote_code=True
)

# ===============================
# 3. 載入 LoRA（adapter_model.bin）
# ===============================
model = PeftModel.from_pretrained(
    model,
    lora_path,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
)

print("✔ LoRA adapter loaded.")

# ===============================
# 4. 載入 non_lora_trainables.bin（若存在）
# ===============================
if os.path.exists(non_lora_path):
    nl = torch.load(non_lora_path, map_location=device)
    model.load_state_dict(nl, strict=False)
    print("✔ non-LoRA projector/encoder weights loaded.")
else:
    print("⚠ No non_lora_trainables.bin found — only LoRA applied.")

model.eval()

# ===============================
# 5. 推論 function（保持你的原版邏輯）
# ===============================
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

    output_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        pad_token_id=processor.tokenizer.pad_token_id or 151643,
    )
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response

# ===============================
# 6. Dataset + Benchmark
# ===============================
data_root = "/workspace/data_root/task1_binary_test.jsonl"
dataset = SentimentSimilarityDataset(data_root)
benchmark = SentimentSimilarityBenchmark()

results = []

# ===============================
# 7. Evaluate
# ===============================
for idx in tqdm(range(len(dataset)), desc="Evaluating Dataset"):
    sample = dataset[idx]
    data_id = sample["data_id"]
    text_inputs = sample["text_inputs"]
    image_inputs = sample["image_inputs"]
    ground_truth = sample["ground_truth"]

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

    response = infer(conversation)
    #print("response:", response)

    prediction = benchmark.process_response(response)
    results.append({
        "data_id": data_id,
        "response": response,
        "prediction": prediction,
        "ground_truth": ground_truth
    })

# ===============================
# 8. Save results
# ===============================
metrics, infos = benchmark.evaluate(results)
output_file = "./result/finetune/stage1Results_lora.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump({"results": results, "metrics": metrics}, f, indent=4, ensure_ascii=False)

print(f"Results saved to {os.path.abspath(output_file)}")
print(json.dumps(metrics, indent=4, ensure_ascii=False))
