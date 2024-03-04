# Fine-tuning LLaMA 2 on the OpenOrca Dataset

This project aims to fine-tune the LLaMA 2 model, specifically `NousResearch/Llama-2-7b-chat-hf` from Hugging Face, on a new dataset named `new_dataset` for generating insights or predictions specific to the dataset's domain. The fine-tuned model will be saved as `Llama-2-7b-chat-finetune`.

## Project Configuration

### Model and Dataset

- **Model to Train**: `NousResearch/Llama-2-7b-chat-hf` from the Hugging Face hub.
- **Instruction Dataset**: `new_dataset` - A curated dataset for fine-tuning.
- **Fine-tuned Model Name**: `Llama-2-7b-chat-finetune`.

### QLoRA Parameters

QLoRA (Query, key, value Low-Rank Adaptation) parameters are crucial for the adaptation process, providing a balance between performance and computational efficiency.

- **LoRA Attention Dimension (`lora_r`)**: 64
- **Alpha Parameter for LoRA Scaling (`lora_alpha`)**: 16
- **Dropout Probability for LoRA Layers (`lora_dropout`)**: 0.1

### bitsandbytes Parameters

Optimizing model loading and computation precision using bitsandbytes.

- **4-bit Precision Base Model Loading (`use_4bit`)**: True
- **Compute dtype for 4-bit Base Models (`bnb_4bit_compute_dtype`)**: `float16`
- **Quantization Type (`bnb_4bit_quant_type`)**: `nf4`
- **Nested Quantization (`use_nested_quant`)**: False

### TrainingArguments Parameters

Configuration for the training process.

- **Output Directory (`output_dir`)**: `./results`
- **Number of Training Epochs (`num_train_epochs`)**: 1
- **FP16/BF16 Training (`fp16`, `bf16`)**: False
- **Batch Size per GPU (`per_device_train_batch_size`, `per_device_eval_batch_size`)**: 4
- **Gradient Accumulation Steps (`gradient_accumulation_steps`)**: 1
- **Gradient Checkpointing (`gradient_checkpointing`)**: True
- **Maximum Gradient Norm (`max_grad_norm`)**: 0.3
- **Initial Learning Rate (`learning_rate`)**: 2e-4
- **Weight Decay (`weight_decay`)**: 0.001
- **Optimizer (`optim`)**: `paged_adamw_32bit`
- **Learning Rate Schedule (`lr_scheduler_type`)**: `cosine`
- **Max Training Steps (`max_steps`)**: -1
- **Warmup Ratio (`warmup_ratio`)**: 0.03
- **Group by Length (`group_by_length`)**: True
- **Save Steps (`save_steps`)**: 0
- **Logging Steps (`logging_steps`)**: 25

### SFT Parameters

Sparse Fine-tuning (SFT) parameters for efficient training.

- **Maximum Sequence Length (`max_seq_length`)**: None
- **Packing (`packing`)**: False
- **Device Map (`device_map`)**: `{"": 0}`

## Setup and Training

To start fine-tuning the LLaMA 2 model with the above configurations, ensure you have the necessary environment and dependencies set up. Follow the instructions below:

1. **Environment Setup**: Ensure you have a Python environment with necessary libraries installed, including `transformers` and `bitsandbytes` for optimization.

2. **Dataset Preparation**: Prepare your `new_dataset` following the guidelines provided in the dataset documentation. Ensure it's formatted correctly for training.

3. **Training**: Run the training script with the specified configurations. Make sure to adjust paths and parameters according to your setup.

## Evaluation and Usage

After training, evaluate the fine-tuned model on a separate validation set to assess its performance. Use the model for generating predictions or insights as per the project's objective.

For detailed usage, refer to the training and evaluation scripts provided in the repository.
