# 📝 Text Summarizer - Advanced NLP Pipeline

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![MLOps](https://img.shields.io/badge/MLOps-Ready-purple.svg)

An end-to-end **Text Summarization** project built with modern MLOps practices using **Hugging Face Transformers**, **PEGASUS model**, and modular architecture. This project demonstrates production-ready machine learning pipeline with proper configuration management, logging, and reproducible experiments.

## 🌟 Features

- **🚀 State-of-the-art Model**: Fine-tuned PEGASUS model for high-quality text summarization
- **📊 Comprehensive Pipeline**: Complete MLOps pipeline from data ingestion to model evaluation
- **🔧 Modular Architecture**: Clean, maintainable code with proper separation of concerns
- **📈 Model Evaluation**: Automated ROUGE score evaluation for performance tracking
- **🐳 Docker Ready**: Containerized application for easy deployment
- **📝 Extensive Logging**: Comprehensive logging for debugging and monitoring
- **⚙️ Configuration Management**: YAML-based configuration for easy experimentation

## 🏗️ Project Architecture

```
textsummarizer/
├── 📁 src/textSummarizer/
│   ├── 🔧 components/          # Core ML components
│   ├── ⚙️  config/             # Configuration management
│   ├── 📋 entity/              # Data classes and entities
│   ├── 🔄 pipeline/            # ML pipelines
│   ├── 🛠️  utils/              # Utility functions
│   └── 📊 logging/             # Logging configuration
├── 📁 research/                # Jupyter notebooks for experimentation
├── 📁 config/                  # YAML configuration files
├── 🐳 Dockerfile              # Container configuration
├── 📋 requirements.txt         # Python dependencies
└── 🚀 app.py                   # Streamlit web application
```

## 📊 Dataset

The project uses the **SAMSum Dataset** - a collection of chat conversations with human-annotated summaries, perfect for training conversational text summarization models.

- **Training Samples**: ~14K conversations
- **Validation Samples**: ~819 conversations  
- **Test Samples**: ~819 conversations

## 🤖 Model Details

- **Base Model**: `google/pegasus-cnn_dailymail`
- **Architecture**: PEGASUS (Pre-training with Extracted Gap-sentences for Abstractive SUmmarization Sequence-to-sequence)
- **Fine-tuning**: Domain-specific fine-tuning on conversational data
- **Evaluation Metrics**: ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### 1. Clone the Repository
```bash
git clone https://github.com/danialasim/textsummarizer.git
cd textsummarizer
```

### 2. Create Virtual Environment
```bash
python -m venv text-sum-env
source text-sum-env/bin/activate  # On Windows: text-sum-env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install the Package
```bash
pip install -e .
```

## 🚀 Quick Start

### Running the Complete Pipeline
```bash
python main.py
```

### Individual Pipeline Stages
```bash
# Data Ingestion
python -c "from src.textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline; DataIngestionTrainingPipeline().main()"

# Data Transformation  
python -c "from src.textSummarizer.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline; DataTransformationTrainingPipeline().main()"

# Model Training
python -c "from src.textSummarizer.pipeline.stage_03_model_trainer import ModelTrainerTrainingPipeline; ModelTrainerTrainingPipeline().main()"

# Model Evaluation
python -c "from src.textSummarizer.pipeline.stage_04_model_evaluation import ModelEvaluationTrainingPipeline; ModelEvaluationTrainingPipeline().main()"
```

### Web Application
```bash
streamlit run app.py
```

## 📋 Configuration

The project uses YAML configuration files for easy experimentation:

### `config/config.yaml`
```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: https://github.com/krishnaik06/datasets/raw/refs/heads/main/summarizer-data.zip
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion

model_trainer:
  root_dir: artifacts/model_trainer
  data_path: artifacts/data_transformation/samsum_dataset
  model_ckpt: google/pegasus-cnn_dailymail
```

### `params.yaml`
```yaml
TrainingArguments:
  num_train_epochs: 3
  warmup_steps: 1000
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  weight_decay: 0.01
  logging_steps: 50
  evaluation_strategy: steps
  eval_steps: 200
  save_steps: 200
  gradient_accumulation_steps: 8
  learning_rate: 5e-5
```

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t text-summarizer .
```

### Run Container
```bash
docker run -p 8501:8501 text-summarizer
```

## 📊 Model Performance

The fine-tuned model achieves the following ROUGE scores on the test set:

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.47+ |
| ROUGE-2 | 0.24+ |
| ROUGE-L | 0.39+ |
| ROUGE-Lsum | 0.39+ |

## 🔬 Research & Experimentation

The `research/` directory contains Jupyter notebooks for:

- **1_data_ingestion.ipynb**: Data loading and preprocessing experiments
- **2_data_transformation.ipynb**: Tokenization and data preparation
- **3_model_trainer.ipynb**: Model fine-tuning experiments
- **textsummarizer.ipynb**: Complete end-to-end experimentation

## 📝 Usage Examples

### Python API
```python
from src.textSummarizer.pipeline.prediction import PredictionPipeline

# Initialize pipeline
predictor = PredictionPipeline()

# Your text to summarize
text = """
Hannah: Hey, do you have Betty's number?
Amanda: Lemme check
Hannah: <file_gif>
Amanda: Sorry, can't find it.
Amanda: Ask Larry
Amanda: He called her last time we were at the park together
Hannah: I don't know him well
Hannah: <file_gif>
Amanda: Don't be shy, he's very nice
Hannah: If you say so..
Amanda: Better yet, I can introduce you to him
Hannah: Hannah: That's sweet of you
Amanda: I'm sure he'll be happy to give you Betty's number
Amanda: or her BeReal username
Hannah: Yeah, I'd prefer that actually
Amanda: Great! I'll text him now
"""

# Generate summary
summary = predictor.predict(text)
print(f"Summary: {summary}")
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for the amazing Transformers library.
- **Google Research** for the PEGASUS model
- **Samsung** for the SAMSum dataset
- **MLOps Community** for best practices and inspiration

## 📞 Contact

**Danial Asim** - [GitHub](https://github.com/danialasim)

Project Link: [https://github.com/danialasim/textsummarizer](https://github.com/danialasim/textsummarizer)

---

⭐ **Star this repository if you found it helpful!** ⭐