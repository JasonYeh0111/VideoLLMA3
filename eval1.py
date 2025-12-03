from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from Benchmark1 import SentimentSimilarityDataset,SentimentSimilarityBenchmark
import os
import json
from tqdm import tqdm


model_path = "../finetune"
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


data_root = "/workspace/data_root/task1_binary_test.jsonl"
dataset = SentimentSimilarityDataset(data_root)
benchmark = SentimentSimilarityBenchmark()

results = []
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
    with torch.no_grad():
        response = infer(conversation)
        #print("response:", response)
    prediction = benchmark.process_response(response)
    results.append({
        "data_id": data_id,
        "response": response,
        "prediction": prediction,
        "ground_truth": ground_truth
    })

metrics, infos = benchmark.evaluate(results)
output_file = "./result/finetune/stage1Results.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({"results": results, "metrics": metrics}, f, indent=4, ensure_ascii=False)

print(f"Results saved to {os.path.abspath(output_file)}")
print(json.dumps(metrics, indent=4, ensure_ascii=False))
