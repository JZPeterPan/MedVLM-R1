# MedVLM-R1: Incentivizing Medical Reasoning Capability of Vision-Language Models via Reinforcement Learning


[![arXiv](https://img.shields.io/badge/arXiv-2502.19634-b31b1b.svg)](https://arxiv.org/abs/2502.19634)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](https://huggingface.co/JZPeterPan/MedVLM-R1)

## Installation

Run the setup script to configure the environment:

```bash
bash setup.sh
```

This script will:
- Create conda environment `medvlm-r1`
- Install necessary dependencies
- Configure the open-r1-multimodal framework

## Quick Start

### Run Demo

Use the Jupyter notebook to quickly experience the model:

```bash
jupyter notebook demo.ipynb
```

The demo includes:
- Model loading
- Medical image VQA examples
- Inference process demonstration

### Example Output

The model generates structured reasoning process:

```
<think>
    The image is a magnetic resonance imaging (MRI) scan of a knee joint. The scan shows a chondral abnormality, which is a type of cartilage damage. This is evident from the irregular shape and the presence of a defect in the cartilage.
</think>

<answer>A</answer>
```

## Dataset Download

### Training and Testing Datasets

Download the HuatuoGPT-Vision dataset via Hugging Face CLI:

```bash
# 1) Install Hugging Face CLI (if not already)
pip install -U "huggingface_hub[cli]"

# 2) (Optional) Login if the dataset requires auth
# huggingface-cli login

# 3) Download the dataset to a local directory
# Replace <TARGET_DIR> with your local path, e.g., /data/datasets/PubMedVision
hf download FreedomIntelligence/PubMedVision \
  --repo-type dataset \
  --local-dir <TARGET_DIR> \
  --local-dir-use-symlinks False \
  --include "*"

# After download, set <DATASET_PATH_ROOT>=<TARGET_DIR> in your scripts
```

The dataset contains:
- MRI, CT, X-ray medical images
- Corresponding visual question-answer pairs
- Multi-modal medical reasoning tasks

## Training and Testing

### Training

Run the training script:

```bash
bash train_script.sh
```

**Note**: Please update the following paths in the script:
- `<DATASET_NAME>`: Dataset name
- `<GPU_NUM>`: Number of GPUs
- `<LOG_PATH>`: Log output path
- `<HF_CACHE_DIR>`: Hugging Face cache directory
- `<WANDB_ENTITY>`: Weights & Biases entity
- `<WANDB_PROJECT>`: Project name
- `<OUTPUT_DIR_ROOT>`: Output directory root path
- `<MODEL_REPO_OR_DIR>`: Model path
- `<DATASET_PATH_ROOT>`: Dataset root path
- `<MASTER_ADDR>`: Master node address
- `<MASTER_PORT>`: Master node port

### Testing

Run the testing script:

```bash
bash test_script.sh
```

**Note**: Please update the following paths in the script:
- `<HF_CACHE_DIR>`: Hugging Face cache directory
- `<CUDA_DEVICES>`: CUDA devices
- `<MODEL_REPO_OR_DIR>`: Model path
- `<DATASET_PATH_ROOT>`: Dataset root path
- `<OUTPUT_DIR>`: Output directory

### Testing Configuration

The testing script supports the following parameters:
- `MODALITY`: Modality type (MRI, CT, Ultrasound, Xray, Dermoscopy, Microscopy, Fundus)
- `PROMPT_TYPE`: Prompt type (simple, complex)
- `BSZ`: Batch size
- `MAX_NEW_TOKENS`: Maximum new tokens to generate
- `DO_SAMPLE`: Whether to sample
- `TEMPERATURE`: Temperature parameter

## Project Structure

```
r1-v-med/
├── demo.ipynb                    # Demo notebook
├── setup.sh                      # Setup script
├── train_script.sh               # Training script
├── test_script.sh                # Testing script
├── MRI_CT_XRAY_300each_dataset.json  # Test dataset
├── images/                       # Example images
│   ├── successful_cases/         # Successful cases
│   └── failure_cases/            # Failure cases
└── src/
    ├── eval/                     # Evaluation code
    │   └── test_qwen2vl_med.py   # Testing script
    ├── distill_r1/               # R1 distillation related
    └── open-r1-multimodal/       # Based framework
        └── src/open_r1/
            ├── grpo.py           # GRPO training code
            └── trainer/
                └── grpo_trainer.py  # GRPO trainer
```

## Acknowledgement

### Citation

If you find our work helpful, please cite:

```bibtex
@article{pan2025medvlm,
  title={MedVLM-R1: Incentivizing Medical Reasoning Capability of Vision-Language Models (VLMs) via Reinforcement Learning},
  author={Pan, Jiazhen and Liu, Che and Wu, Junde and Liu, Fenglin and Zhu, Jiayuan and Li, Hongwei Bran and Chen, Chen and Ouyang, Cheng and Rueckert, Daniel},
  journal={arXiv preprint arXiv:2502.19634},
  year={2025}
}
```

### Base Frameworks

Our code is based on the following open-source projects:

- **open-r1-multimodal**: [https://github.com/EvolvingLMMs-Lab/](https://github.com/EvolvingLMMs-Lab/)
- **R1-V**: [https://github.com/StarsfieldAI/R1-V](https://github.com/StarsfieldAI/R1-V)

Thanks to these excellent open-source projects for providing a solid foundation for our research.

## License

This project is licensed under the Apache 2.0 License.