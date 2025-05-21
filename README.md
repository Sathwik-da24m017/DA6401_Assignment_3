# DA6401_Assignment_3

# Sathwik Pentela DA24M017

# Dakshina Transliteration System (RNN Seq2Seq)

This project provides an end-to-end implementation of a character-level sequence-to-sequence transliteration system using PyTorch. The system translates words written in the Latin script into their corresponding forms in Indian native scripts (e.g., Telugu). It includes both vanilla encoder-decoder models and attention-based variants.

## Project Structure

The repository is organized as follows:

```
├── data_loader.py             # Dataset loading, preprocessing, tokenization
├── models.py                  # Model definitions: Vanilla Encoder-Decoder, Bahdanau Attention
├── train_vanilla.py           # Training loop with Weights & Biases for vanilla Seq2Seq
├── train_attention.py         # Training loop for attention-based Seq2Seq
├── evaluate.py                # Greedy decoding and test evaluation
├── visualize.py               # Attention heatmaps using matplotlib
├── interactive_vis.py         # Interactive HTML-based attention visualizer
├── main.py                    # Script-based orchestrator for training, evaluation, and visualization
├── main.ipynb                 # Notebook version of the entire workflow (training, evaluation, attention)
├── README.md                  # Project overview and usage instructions
├── dakshina_dataset_v1.0/     # Folder containing dataset files
├── *.pth                      # Trained model checkpoints
├── predictions_vanilla/       # Vanilla model predictions on test set
├── predictions_attention/     # Attention model predictions on test set
├── *.html                     # Attention visualization outputs
```

## Environment Setup

To get started, ensure the following dependencies are installed:

```bash
pip install torch pandas matplotlib seaborn wandb
```

## Dataset Requirements

Download the [Dakshina dataset](https://github.com/google-research-datasets/dakshina) from Google Research. Extract the required Telugu lexicon files:

- `te.translit.sampled.train.tsv`
- `te.translit.sampled.dev.tsv`
- `te.translit.sampled.test.tsv`

Ensure the files are placed under:

```
dakshina_dataset_v1.0/te/lexicons/
```

## How to Run the Project

### Option 1: Use Jupyter Notebook

Use `main.ipynb` to run the full assignment in a step-by-step, interactive manner. It includes:
- Data preprocessing
- Training (vanilla and attention)
- Evaluation
- Static and interactive attention visualization

This is the recommended mode if you want to demonstrate each step manually.

### Option 2: Use Python Scripts

For CLI automation, use the modular `.py` files:

- `train_vanilla.py` or `train_attention.py` for training
- `evaluate.py` for test prediction
- `visualize.py` for generating heatmaps
- `interactive_vis.py` for generating HTML attention explorers
- `main.py` for running the full pipeline from terminal

Example usage:

```bash
python main.py --model_type attention --checkpoint best_model_attn.pth --eval --visualize --interactive
```

## Training the Models

### Option A: Hyperparameter Sweep with Weights & Biases

#### Vanilla Encoder-Decoder Model

```python
from train_vanilla import sweep_config, run_training
import wandb
sweep_id = wandb.sweep(sweep_config, project="DA6401_Transliteration")
wandb.agent(sweep_id, function=lambda: run_training(...), count=30)
```

#### Attention-Based Encoder-Decoder Model

```python
from train_attention import sweep_config, run_attention_training
import wandb
sweep_id = wandb.sweep(sweep_config, project="DA6401_Attention")
wandb.agent(sweep_id, function=lambda: run_attention_training(...), count=25)
```

### Option B: Direct Training with Best Configuration

```python
best_config = {
    "embedding_dim": 64,
    "hidden_dim": 256,
    "num_layers": 3,
    "dropout": 0.5,
    "cell_type": "LSTM",
    "lr": 0.001,
    "teacher_forcing": 0.5,
    "epochs": 15
}

import wandb
wandb.init(project="DA6401", name="vanilla_best")
run_training(best_config)
```

## Evaluating the Model

Use the `evaluate.py` module to run inference on the test dataset and save predictions to a CSV file:

```python
from evaluate import run_test_evaluation
run_test_evaluation(model, test_pairs, input2idx, target2idx, idx2target, output_csv="test_predictions.csv")
```

The CSV file will contain the following columns:

```
Input (Roman), Ground Truth (Telugu), Prediction (Telugu), Match
```

## Visualizing Attention Weights

### Static Visualization using Heatmaps

```python
from visualize import plot_attention_grid
plot_attention_grid(model, test_samples, input2idx, target2idx, idx2target)
```

This will generate 3x3 grid heatmaps for a few sample inputs and save them as `.png` files.

### Interactive Visualization using HTML

```python
from interactive_vis import save_interactive_attention_html
save_interactive_attention_html(examples_dict, filename="attn_visualization.html")
```

This will produce an HTML file with token-level hoverable highlights. You can embed this HTML file in your W&B report or host it online.

## Notes

- The attention visualizations only apply to models trained with attention mechanisms.
- Accuracy is computed as the number of exact match predictions over total predictions.
- Models are saved automatically as `best_model.pth` or `best_model_attn.pth` during training.


