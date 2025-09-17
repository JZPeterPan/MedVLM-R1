# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections import Counter

from io import BytesIO
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk, Image, concatenate_datasets
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from PIL import Image as PILImage



@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
        dataset_selection (`str`):
            Dataset selection for medical imaging. Possible values: 'MR', 'CT', 'XRAY', 'COMBINED'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    dataset_selection: str = field(
        default="MR",
        metadata={"help": "Dataset selection for medical imaging. Possible values: 'MR', 'CT', 'XRAY', 'COMBINED'"},
    )


def accuracy_reward(completions, solution, image_ids, problems, **kwargs):
    """
    Reward function for multiple-choice answers:
      - 1.0 if exactly one occurrence of the correct letter (A–J) and no other text.
      - 0.5 if exactly one occurrence of the correct letter plus extra text.
      - 0.0 otherwise (no letters, wrong letter, multiple letters, repeated letters, etc).
    """
    rewards = []

    for comp, sol, image_id, problem in zip(completions, solution, image_ids, problems):
        content = comp[0]["content"]  # typical structure: [{ "role":"assistant", "content": "..."}]
        reward = 0.0

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        # Extract ground truth letter
        sol_match = re.search(r"<answer>(.*?)</answer>", sol, flags=re.DOTALL|re.IGNORECASE)
        if sol_match:
            ground_truth = sol_match.group(1).strip()
        else:
            ground_truth = sol.strip()

        # Extract student's answer
        content_match = re.search(r"<answer>(.*?)</answer>", content, flags=re.DOTALL|re.IGNORECASE)
        if content_match:
            student_answer = content_match.group(1).strip()
        else:
            student_answer = ""  # no <answer> => 0.0

        # Prepare comparisons
        gt_letter = re.sub(r"\s+", "", ground_truth.upper())
        sa_str = student_answer.strip()  
        sa_upper = sa_str.upper()

        if not sa_upper:
            reward = 0.0
        else:
            letters_found = re.findall(r"[A-J]", sa_upper)
            distinct_letters = set(letter.upper() for letter in letters_found)

            if len(letters_found) == 1 and len(distinct_letters) == 1:
                found_letter = distinct_letters.pop()
                if found_letter == gt_letter:
                    # Check leftover text
                    leftover = re.sub(r"(?i)" + found_letter, "", sa_str, count=1)
                    if leftover.strip():
                        # Has extra text
                        reward = 0.5
                    else:
                        # Perfect single-letter match
                        reward = 1.0
                else:
                    # Single letter but not correct
                    reward = 0.0
            else:
                # no letters, multiple letters, repeated letter, etc.
                reward = 0.0
        
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Image ID: {image_id}\n")
                f.write(f"Problem: {problem}\n\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]


    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]


    if os.environ.get("DATASET") == "Huatuo" or os.environ.get("DATASET") == "ISIC":
        # for medical multi-choice questions
        QUESTION_TEMPLATE = """
        {Question} 
        Your task: 
        1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags. 
        2. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.
        3. No extra information or text outside of these tags.
        """
    else:
        QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."
    
    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
    if  os.environ.get("DATASET") == 'VQA-RAD':
        # todo: potential bugs for CPU not GPU training
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)['train']

        def transform_example(example):
            # Rename 'question' to 'problem'
            example['problem'] = example.pop('question')
            # Create 'solution' column from 'answer'
            example['solution'] = f"<answer> {example['answer']} </answer>"
            return example

        dataset = dataset.map(transform_example)
        dataset = dataset.remove_columns('answer')

    elif os.environ.get("DATASET") == 'SLAKE':
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, split='train').select(range(2000))
        base_image_dir = f'{script_args.dataset_name}/SLAKE/imgs'


        def transform_example(example):
            # Construct the full image path
            image_path = os.path.join(base_image_dir, example['img_name'])
            
            # Load the image
            example['image'] = PILImage.open(image_path).convert('RGBA')
            
            # Rename 'question' to 'problem'
            example['problem'] = example.pop('question')
            
            # Create 'solution' column from 'answer'
            example['solution'] = f"<answer> {example['answer']} </answer>"
            
            return example
        
        dataset = dataset.filter(lambda example: example['answer_type'] == 'CLOSED' and example['modality'] == 'CT')
        # Apply the transformation to the dataset
        dataset = dataset.map(transform_example, num_proc=2)
        # Remove the original columns that are no longer needed
        dataset = dataset.remove_columns(['img_id', 'img_name', 'answer', 'q_lang', 'location', 'modality', 'answer_type', 'base_type', 'content_type', 'triple', 'qid'])
    
    elif os.environ.get("DATASET") == 'Huatuo':
        dataset = load_dataset("FreedomIntelligence/Medical_Multimodal_Evaluation_Data", split="test")

        dataset_mr = dataset.filter(lambda x: x["subset"] == "MR (Mag-netic Resonance Imaging)").select(range(500))
        dataset_ct = dataset.filter(lambda x: x["subset"] == "CT(Computed Tomography)").select(range(500))
        dataset_xray = dataset.filter(lambda x: x["subset"] == "X-Ray").select(range(500))

        # Select dataset based on the parameter
        if script_args.dataset_selection == "MR":
            dataset = dataset_mr
        elif script_args.dataset_selection == "CT":
            dataset = dataset_ct
        elif script_args.dataset_selection == "XRAY":
            dataset = dataset_xray
        elif script_args.dataset_selection == "COMBINED":
            dataset = concatenate_datasets([dataset_mr, dataset_ct, dataset_xray])
        else:
            raise ValueError(f"Invalid dataset_selection: {script_args.dataset_selection}. Must be one of: 'MR', 'CT', 'XRAY', 'COMBINED'")
        
        # shuffle the dataset
        dataset = dataset.shuffle(seed=42)



        print("dataset Huatuo loaded")
        def create_problem_solution(example, base_image_path):
            """
            1) Create a new column 'problem' by combining the question and options into a multiple-choice format.
            2) Convert the 'answer' to its corresponding letter (A, B, C, ...) and store it in 'solution'.
            3) Load the image from base_image_path and store the actual image object in 'image'.
            4) Remove all other keys, keeping only 'image', 'problem', 'solution'.
            """
            # We'll support up to 10 options (A-J).
            letter_map = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
            
            # Combine question + labeled options
            labeled_options = [f"{letter_map[i]}) {opt}" for i, opt in enumerate(example["options"])]
            example['problem'] = example["question"] + "\n" + "\n".join(labeled_options)
            
            # Convert text answer to letter
            correct_index = example["options"].index(example["answer"])
            example["solution"] = letter_map[correct_index]
            
            # Load the actual image from disk

            example['image_id'] = example["image"][0]
            image_path = os.path.join(base_image_path, example["image"][0])
            example["image"] = PILImage.open(image_path).convert("RGB")
            return example
        
        base_image_path = script_args.dataset_name
        dataset = dataset.map(lambda x: create_problem_solution(x, base_image_path), num_proc=2)
        dataset = dataset.remove_columns(["question", "dataset", "subset", "answer"])
        dataset = dataset.cast_column("image", Image()) # todo: to optimize 
    
    elif os.environ.get("DATASET") == 'ISIC':
        dataset = load_dataset(f"{script_args.dataset_name}/ISIC-2017", split="train").select(range(1600))
        print("dataset ISIC loaded")

        def filter_by_pixels(example):
            # Get width, height from the PIL Image
            width, height = example["image"].size
            # Keep only those whose pixel count is under 1e6
            return (width * height) < 1_000_000

        def add_problem_and_solution(example):
            """
            Given an example of the form:
            {
            'image': <some PIL image>,
            'label': <integer: 0,1,2>
            }
            we add two columns:
            - 'problem': A fixed multiple-choice question about skin cancer.
            - 'solution': The corresponding letter ('A','B','C') derived from 'label'.
            """
            # Fixed question text
            problem_text = (
                """Which type of skin cancer is it?\n
                A) melanoma\n
                B) nevus\n
                C) seborrheic_keratosis"""
            )

            # Map label -> letter
            label_map = ["A", "B", "C"]
            # Derive the solution letter based on example["label"]
            solution_letter = label_map[example["label"]]

            # Update the example with new columns
            example["problem"] = problem_text
            example["solution"] = solution_letter
            return example

        dataset = dataset.filter(filter_by_pixels)
        dataset = dataset.map(add_problem_and_solution)
        dataset = dataset.remove_columns(["label"])

    
    if "image" in dataset.features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    
    trainer_cls = Qwen2VLGRPOTrainer


    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
