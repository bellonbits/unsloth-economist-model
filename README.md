# Economist Fine-Tuning with Unsloth

This repository contains a comprehensive notebook for fine-tuning language models using the Unsloth framework, specifically optimized for generating Economist-style content. The implementation leverages advanced quantization techniques and memory-efficient training methods to enable high-quality model customization on limited hardware resources.

## 🚀 Overview

The project demonstrates the complete workflow for fine-tuning the `Llama-3.2-3B-Instruct` model using Unsloth's optimized framework. The approach combines 4-bit quantization with efficient training algorithms to produce a specialized model capable of generating professional economic journalism and analysis.

## ✨ Key Features

- **Base Model**: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` (4-bit quantized version)
- **Framework**: Unsloth for efficient fine-tuning with significant speed improvements
- **Context Length**: 2048 tokens for comprehensive document processing
- **Quantization**: 4-bit quantization for dramatic memory reduction
- **Hardware Compatibility**: Optimized for Tesla T4, V100 (Float16) and Ampere+ architectures (Bfloat16)
- **Memory Efficiency**: Reduced VRAM requirements while maintaining model quality

## 🔧 Installation

### Prerequisites

Ensure you have a CUDA-compatible GPU with sufficient VRAM (minimum 8GB recommended).

### Dependencies

Install the required packages using the following commands:

```bash
# Core dependencies with specific version constraints
pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo

# Additional required packages
pip install sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer

# Unsloth framework (install without dependencies to avoid conflicts)
pip install --no-deps unsloth
```

### Alternative Installation (Conda)

```bash
conda create -n economist-ft python=3.10
conda activate economist-ft
# Follow pip installation steps above
```

## 📚 Usage

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd economist-finetuning-unsloth
   ```

2. **Open the notebook**:
   ```bash
   jupyter notebook economist_finetuning.ipynb
   ```

3. **Follow the notebook sections**:
   - Environment setup and dependency installation
   - Model loading with optimized configuration
   - Dataset preparation for Economist-style content
   - Fine-tuning process with custom parameters
   - Model evaluation and validation
   - Export and deployment options

### Notebook Structure

The notebook is organized into the following key sections:

1. **Environment Setup**: Installing and configuring all required dependencies
2. **Model Loading**: Loading the quantized Llama model with Unsloth optimizations
3. **Data Preparation**: Processing and formatting training data for economic content
4. **Training Configuration**: Setting hyperparameters and training options
5. **Fine-tuning Process**: Running the optimized training loop
6. **Evaluation**: Testing model performance on validation data
7. **Model Export**: Saving the fine-tuned model for deployment

## ⚡ Performance Characteristics

### Memory Efficiency
- **4-bit Quantization**: Reduces model size by ~75% compared to full precision
- **Optimized Attention**: Unsloth's attention mechanisms reduce memory overhead
- **Gradient Checkpointing**: Further memory savings during training

### Training Speed
- **2x Faster Training**: Unsloth optimizations provide significant speedup
- **Efficient Data Loading**: Optimized data pipeline reduces I/O bottlenecks
- **Dynamic Batching**: Automatic batch size optimization for available hardware

### Hardware Requirements
- **Minimum**: 8GB VRAM (Tesla T4, RTX 3070)
- **Recommended**: 16GB VRAM (RTX 4080, V100)
- **Optimal**: 24GB+ VRAM (RTX 4090, A100)

## 🎯 Applications

This fine-tuned model excels at generating:

### Economic Journalism
- **News Articles**: Financial market analysis and economic trend reporting
- **Opinion Pieces**: Editorial content on economic policy and market developments
- **Data Interpretation**: Converting complex economic data into accessible narratives

### Business Analysis
- **Market Research**: Industry analysis and competitive landscape reports
- **Financial Commentary**: Investment insights and market forecasting
- **Corporate Analysis**: Company performance evaluation and strategic assessments

### Academic Writing
- **Research Papers**: Economic research methodology and findings presentation
- **Policy Analysis**: Government policy evaluation and recommendation frameworks
- **Educational Content**: Economics concepts explanation and case study development

## 🔬 Technical Details

### Model Architecture
- **Base Model**: Llama-3.2-3B-Instruct with 3 billion parameters
- **Quantization**: 4-bit NF4 quantization with double quantization
- **LoRA Configuration**: Low-rank adaptation for efficient fine-tuning
- **Attention Optimization**: Flash Attention 2.0 for memory efficiency

### Training Configuration
```python
# Example configuration
max_seq_length = 2048
dtype = None  # Auto-detection for optimal performance
load_in_4bit = True
batch_size = 2  # Adjust based on available VRAM
gradient_accumulation_steps = 4
learning_rate = 2e-4
num_train_epochs = 3
```

## 📊 Evaluation Metrics

The model's performance is evaluated using:

- **Perplexity**: Measures model's confidence in predictions
- **BLEU Score**: Evaluates generated text quality against references
- **Economic Content Accuracy**: Custom metrics for domain-specific evaluation
- **Style Consistency**: Assessment of writing style alignment with Economist standards

## 🚦 Getting Started Guide

### Step 1: Environment Setup
```python
# Import required libraries
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
```

### Step 2: Model Loading
```python
# Load the quantized model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    max_seq_length=2048,
    dtype=None,  # Auto-detect
    load_in_4bit=True,
)
```

### Step 3: Fine-tuning Setup
```python
# Configure LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0.1,
)
```

## 🛠️ Troubleshooting

### Common Issues

**Memory Errors**:
- Reduce batch size to 1
- Enable gradient checkpointing
- Use smaller sequence lengths

**Installation Issues**:
- Ensure CUDA compatibility
- Use virtual environment to avoid conflicts
- Install packages in specified order

**Training Instability**:
- Lower learning rate
- Increase warmup steps
- Use gradient clipping

## 📈 Future Enhancements

- **Multi-language Support**: Extend to international economic publications
- **Real-time Data Integration**: Incorporate live market data
- **Advanced Evaluation**: Implement domain-expert evaluation protocols
- **Model Compression**: Further optimization for deployment efficiency

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any enhancements.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Unsloth Team**: For providing the efficient fine-tuning framework
- **Meta AI**: For the Llama model architecture
- **Hugging Face**: For the transformers library and model hosting
- **The Economist**: For inspiration in economic journalism standards

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation in the notebook
- Review the troubleshooting section above

---

**Note**: This implementation uses 4-bit quantization to enable training on consumer-grade hardware while maintaining high-quality text generation capabilities. The model is specifically optimized for economic and financial content generation, making it ideal for applications in business journalism, market analysis, and academic economic writing.
