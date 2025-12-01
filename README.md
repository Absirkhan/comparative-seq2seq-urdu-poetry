# Comparative Analysis of Neural Architectures for Urdu Poetry Generation

## Project Overview

This project implements a comprehensive comparison of neural sequence-to-sequence architectures for generating classical Urdu poetry. We evaluate **9 combinations** of 3 architectures (RNN, LSTM, Transformer) paired with 3 optimizers (Adam, RMSprop, SGD), plus additional hyperparameter tuning experiments.

## Dataset

- **Source**: [ReySajju742/Urdu-Poetry-Dataset](https://huggingface.co/datasets/ReySajju742/Urdu-Poetry-Dataset) (HuggingFace)
- **Size**: 1,323 classical Urdu poems from renowned poets (Mirza Ghalib, Allama Iqbal)
- **Split**: 80% training, 10% validation, 10% test
- **Language**: Urdu (UTF-8 encoded)

## Model Architectures

### 1. Simple RNN
- 2 layers with 256 units each
- Embedding dimension: 256
- Dropout: 0.2

### 2. LSTM
- 2 layers with 256 units each
- Embedding dimension: 256
- Dropout: 0.2

### 3. Transformer
- 4 attention heads
- 2 transformer blocks
- Feed-forward network dimension: 512
- Embedding dimension: 256
- Dropout: 0.2

## Optimizers

1. **Adam**: Learning rate = 0.001
2. **RMSprop**: Learning rate = 0.001
3. **SGD**: Learning rate = 0.01, Momentum = 0.9

---

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- GPU recommended (8GB+ VRAM) for faster training
- Google Colab (recommended) or local environment with GPU support

### Option 1: Google Colab (Recommended)

1. Upload the notebook to Google Colab:
   - Go to [Google Colab](https://colab.research.google.com/)
   - Upload `urdu_poetry_comparison.ipynb`
   - Enable GPU: Runtime → Change runtime type → GPU

2. Dependencies are automatically installed in the notebook

### Option 2: Local Installation

1. Clone or download this repository

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Launch Jupyter:
```bash
jupyter notebook urdu_poetry_comparison.ipynb
```

---

## How to Run the Code

### Step-by-Step Execution

1. **Open the notebook**: `urdu_poetry_comparison.ipynb`

2. **Run cells sequentially** from top to bottom:
   - **Section 1**: Setup & Installation (installs dependencies)
   - **Section 2**: Data Loading & Preprocessing (loads Urdu poetry dataset)
   - **Section 3**: Model Definition (defines RNN, LSTM, Transformer architectures)
   - **Section 4**: Training Loop (trains all 9 model-optimizer combinations)
   - **Section 5**: Hyperparameter Tuning (tests 8 different configurations)
   - **Section 6**: Evaluation & Visualization (generates metrics and plots)
   - **Section 7**: Text Generation (generates sample poetry from each model)

3. **Execution time**:
   - Full training (9 models): ~3-5 hours on Google Colab GPU
   - Hyperparameter tuning: ~1-2 additional hours
   - CPU execution: Not recommended (10-20x slower)

4. **Memory management**: The notebook includes automatic garbage collection between model trainings to prevent memory issues

### Key Features
- Early stopping and learning rate scheduling
- Model checkpointing (saves best models automatically)
- Progressive results saving (CSV updated after each model)
- Comprehensive visualizations
- Reproducible results (seed=42)

---

## Expected Outputs

### Directory Structure After Running

```
comparative-seq2seq-urdu-poetry/
├── csvs/                              # CSV results and data
│   ├── final_results_all_models.csv   # Main results: all 9 models
│   ├── training_results_progress.csv  # Intermediate training progress
│   ├── hyperparameter_tuning_results.csv  # Hyperparameter experiments
│   ├── poetry_generations_all_models.csv  # Generated text samples
│   ├── best_poetry_samples.csv        # Top quality generations
│   ├── worst_poetry_samples.csv       # Lowest quality generations
│   ├── top_10_poetry_samples.csv      # Top 10 samples by quality
│   └── most_creative_samples.csv      # Most creative (diverse) samples
│
├── visualizations/                    # Generated plots
│   ├── perplexity_comparison.png      # Bar chart: perplexity for all 9 models
│   ├── training_time_comparison.png   # Bar chart: training time analysis
│   ├── perplexity_heatmap.png         # Heatmap: architecture × optimizer
│   ├── all_loss_curves.png            # Training curves (9 subplots)
│   ├── comprehensive_comparison.png   # Multi-metric comparison dashboard
│   ├── vocabulary_diversity_by_model.png  # Vocabulary diversity analysis
│   ├── quality_ratings_by_model.png   # Generated text quality ratings
│   ├── temperature_effect_on_quality.png  # Temperature parameter analysis
│   ├── hyperparameter_tuning_comparison.png  # Hyperparameter results
│   ├── hyperparameter_sensitivity_analysis.png  # Sensitivity analysis
│   └── hyperparameter_tradeoff_analysis.png  # Performance vs time tradeoff
│
├── urdu_poetry_comparison.ipynb       # Main notebook
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

### CSV Outputs

#### 1. final_results_all_models.csv
Contains comprehensive metrics for all 9 models:
- Model name and optimizer
- Train/validation/test loss
- Test accuracy
- **Perplexity** (primary metric: lower is better)
- Training time (minutes)
- Best epoch
- Total parameters

**Sample output:**
```
model_name,optimizer,test_loss,test_accuracy,perplexity,training_time_minutes
RNN,SGD,6.256,0.0876,520.93,2.91
LSTM,RMSprop,6.375,0.0858,586.87,3.68
LSTM,Adam,6.580,0.0682,720.27,2.56
...
```

#### 2. hyperparameter_tuning_results.csv
Results from 8 hyperparameter experiments:
- Experiment ID and hypothesis
- Parameter tested (learning rate, dropout, layers, batch size)
- Perplexity change and improvement percentage
- Training time change
- Conclusion (accept/reject hypothesis)

#### 3. poetry_generations_all_models.csv
Generated poetry samples from all models:
- Model name and optimizer
- Seed text (Urdu input)
- Generated text (Urdu output)
- Generation parameters (temperature, num_words)
- Quality metrics (vocabulary diversity, repetition rate)

### Visualization Outputs

#### Main Performance Plots

1. **perplexity_comparison.png**: Bar chart showing perplexity for all 9 models (lower is better)
2. **perplexity_heatmap.png**: 3×3 heatmap showing architecture-optimizer combinations
3. **training_time_comparison.png**: Training time for each model
4. **all_loss_curves.png**: 9 subplots showing training/validation loss curves
5. **comprehensive_comparison.png**: Multi-panel dashboard with perplexity, accuracy, time, and parameters

#### Text Quality Analysis

6. **vocabulary_diversity_by_model.png**: Vocabulary richness of generated text
7. **quality_ratings_by_model.png**: Overall quality scores by model
8. **temperature_effect_on_quality.png**: How temperature affects generation quality

#### Hyperparameter Analysis

9. **hyperparameter_tuning_comparison.png**: Results of 8 hyperparameter experiments
10. **hyperparameter_sensitivity_analysis.png**: Sensitivity to different parameters
11. **hyperparameter_tradeoff_analysis.png**: Performance vs. training time tradeoffs

### Key Metrics Explained

- **Perplexity**: exp(loss) - measures how well the model predicts text (lower = better)
  - < 500: Excellent
  - 500-700: Good
  - 700-900: Moderate
  - > 900: Poor

- **Vocabulary Diversity**: unique_words / total_words (higher = more creative)

- **Repetition Rate**: Frequency of repeated bigrams (lower = better)

### Sample Results

Based on the actual run, the best performing models were:
1. **RNN + SGD**: Perplexity = 520.93 (best overall)
2. **LSTM + RMSprop**: Perplexity = 586.87
3. **LSTM + Adam**: Perplexity = 720.27

---

## Research Questions Addressed

1. Which architecture performs best for Urdu poetry generation?
2. How do different optimizers affect model performance?
3. What is the trade-off between model complexity and performance?
4. How does training time correlate with model quality?
5. What hyperparameters have the most impact on performance?

---

## Technical Details

### Evaluation Metrics

**Quantitative Metrics:**
- Perplexity (primary metric) = exp(loss)
- Training/Validation/Test Loss
- Accuracy
- Training Time (minutes)
- Best Epoch

**Text Quality Metrics:**
- Vocabulary Diversity: unique_words / total_words
- Repetition Rate: bigram repetition frequency
- Total words and unique words count

### Training Configuration

- **Epochs**: Variable (with early stopping)
- **Batch Size**: 128
- **Sequence Length**: Max 50 words
- **Vocabulary Size**: ~10,000 most frequent words
- **Early Stopping**: Patience = 5 epochs
- **Learning Rate Scheduling**: ReduceLROnPlateau (factor=0.5, patience=3)

---

## Troubleshooting

**Issue**: Out of memory errors
- **Solution**: Reduce batch size or use Google Colab with GPU

**Issue**: Slow training
- **Solution**: Enable GPU in Colab (Runtime → Change runtime type → GPU)

**Issue**: Package installation errors
- **Solution**: Ensure Python >= 3.8, upgrade pip: `pip install --upgrade pip`

**Issue**: Urdu text not displaying correctly
- **Solution**: Ensure UTF-8 encoding, use Urdu-compatible fonts in visualization

---

## Dependencies

See [requirements.txt](requirements.txt) for full list:
- tensorflow >= 2.13.0
- datasets >= 2.14.0
- transformers >= 4.30.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- tqdm >= 4.65.0

---

## Contributing

This is an academic project for Deep Learning coursework. Feel free to fork and experiment with:
- Different hyperparameters
- Additional architectures (GRU, Bidirectional LSTM)
- Temperature variations for text generation
- Extended training epochs
- Fine-tuning on specific poets

---

## References

- Dataset: [Urdu-Poetry-Dataset on HuggingFace](https://huggingface.co/datasets/ReySajju742/Urdu-Poetry-Dataset)
- Transformer Architecture: "Attention Is All You Need" (Vaswani et al., 2017)
- LSTM: "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)

---

## License

MIT License - See LICENSE file for details

---

## Author

**Course**: Deep Learning (7th Semester)
**Date**: December 2025

---

**Note**: This project is designed to run efficiently on Google Colab with GPU acceleration. For local execution, ensure adequate GPU memory (8GB+ recommended).
