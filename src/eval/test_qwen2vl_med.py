import os 
import argparse


from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, GenerationConfig, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re

parser = argparse.ArgumentParser(description="Run Qwen2-VL medical evaluation")
parser.add_argument("--modality", choices=["CT", "Ultrasound", "MRI", "Xray", "Dermoscopy", "Microscopy", "Fundus"], required=True)
parser.add_argument("--prompt_type", choices=["simple", "complex"], required=True)
parser.add_argument("--model_path", required=True)
parser.add_argument("--bsz", type=int, default=1)
parser.add_argument("--output_path", required=True)
parser.add_argument("--prompt_path", required=True)
parser.add_argument("--base_path", required=True, help="Base path for images, corresponds to BATH_PATH in code")
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--do_sample", type=lambda x: str(x).lower() == "true", default=False)
parser.add_argument("--temperature", type=float, default=1.0)
args = parser.parse_args()

Modality = args.modality
Prompt_type = args.prompt_type
MODEL_PATH = args.model_path
BSZ = args.bsz
OUTPUT_PATH = args.output_path
PROMPT_PATH = args.prompt_path
BATH_PATH = args.base_path

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

temp_generation_config = GenerationConfig(
    max_new_tokens=args.max_new_tokens,
    do_sample=args.do_sample,  
    temperature=args.temperature, 
    num_return_sequences=1,
    pad_token_id=151643,
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

# processor.image_processor.max_pixels = 500000

data = []

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# data = dataset[300:600]
if Modality == 'MRI':
    data = dataset[:300]
elif Modality == 'CT':
    data = dataset[300:600]
elif Modality == 'Xray':
    data = dataset[600:]

if Prompt_type == 'simple':
    QUESTION_TEMPLATE = """
    {Question} 
    Your task: provide the correct single-letter choice (A, B, C, D,...).
    """
elif Prompt_type == 'complex':
    QUESTION_TEMPLATE = """
    {Question} 
    Your task: 
    1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags. 
    2. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.
    3. No extra information or text outside of these tags.
    """

messages = []


def convert_example_SFT(example):
    """
    correct example into "messages" 
    eg:
    {
      "system": "You are a helpful assistant.",
      "conversations": [
          {"from": "user", "value": "How many objects are included in this image?",
           "image_path": "/path/to/image.png"},
          {"from": "assistant", "value": "<think>\nI can see 10 objects\n</think>\n<answer>\n10\n</answer>"}
      ]
    }
    """
    messages = []
    if "system" in example:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": example["system"]}],
        })
    else:
        SYSTEM_PROMPT = (
    # """ A conversation between User and Assistant. The user asks a multiple choice question, and the Assistant answers with the correct single-letter choice (A, B, C, D,...)."""
    """ A conversation between User and Assistant. The user asks a multiple choice question, and the Assistant should
    1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags. 
    2. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.
    3. No extra information or text outside of these tags.."""
        
        )
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        })

for i in data:
    message = [{
        "role": "user",
        "content": [
            {
                "type": "image", 
                # "image": f"file://{i['image_path']}"
                # using joined path
                "image": f"file://{BATH_PATH}/{i['image'][0]}"
            },
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(Question=i['problem'])
            }
        ]
    }]
    messages.append(message)



all_outputs = []  # List to store all answers

# Process data in batches
for i in tqdm(range(0, len(messages), BSZ)):
    batch_messages = messages[i:i + BSZ]
    
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
    
    image_inputs, video_inputs = process_vision_info(batch_messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=512, do_sample=False, generation_config=temp_generation_config)
    # generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=512, do_sample=False)

    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    all_outputs.extend(batch_output_text)
    print(f"Processed batch {i//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}")

def extract_letter_answer(output_str):
    # Try to find the letter within <answer> tags, if not found, return None
    answer_pattern = r'<answer>\s*([A-Z])\s*</answer>'
    match = re.search(answer_pattern, output_str)

    if match:
        return match.group(1)  # Return the letter
    return None


final_output = []
correct_number = 0

for input_example, model_output in zip(data,all_outputs):
    original_output = model_output
    ground_truth = input_example['solution']
    if Prompt_type == 'complex':
        model_answer = extract_letter_answer(original_output)
    elif Prompt_type == 'simple':
        model_answer = original_output
    else:
        raise ValueError("Prompt_type must be either 'complex' or 'simple'")
    
    # Create a result dictionary for this example
    result = {
        'question': input_example,
        'ground_truth': ground_truth,
        'model_output': original_output,
        'extracted_answer': model_answer
    }
    final_output.append(result)
    
    # Count correct answers
    if model_answer is not None and model_answer == ground_truth:
        correct_number += 1

# Calculate and print accuracy
accuracy = correct_number / len(data) * 100
print(f"\nAccuracy: {accuracy:.2f}%")

# Save results to a JSON file
output_path = OUTPUT_PATH
with open(output_path, "w") as f:
    json.dump({
        'accuracy': accuracy,
        'results': final_output
    }, f, indent=2)

print(f"Results saved to {output_path}")





