import torch
import torch.nn as nn
import wandb
from data_loader import PAD_TOKEN
from models import Encoder, AttentionDecoder, Seq2Seq

# ------------------------------
# Compute accuracy ignoring PAD
# ------------------------------
def compute_accuracy(preds, targets, pad_idx):
    preds = preds.argmax(dim=-1)
    mask = targets != pad_idx
    correct = (preds == targets) & mask
    return correct.sum().item() / mask.sum().item()

# ------------------------------
# Train model for one epoch
# ------------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device, pad_idx, teacher_forcing):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()

        outputs, _ = model(src, tgt, teacher_forcing_ratio=teacher_forcing)
        outputs = outputs[:, 1:].reshape(-1, outputs.size(-1))
        tgt = tgt[:, 1:].reshape(-1)

        loss = criterion(outputs, tgt)
        acc = compute_accuracy(outputs, tgt, pad_idx)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc

    return total_loss / len(dataloader), total_acc / len(dataloader)

# ------------------------------
# Evaluation on val/test set
# ------------------------------
def evaluate(model, dataloader, criterion, device, pad_idx):
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            outputs, _ = model(src, tgt, teacher_forcing_ratio=0.0)

            outputs = outputs[:, 1:].reshape(-1, outputs.size(-1))
            tgt = tgt[:, 1:].reshape(-1)

            loss = criterion(outputs, tgt)
            acc = compute_accuracy(outputs, tgt, pad_idx)

            total_loss += loss.item()
            total_acc += acc

    return total_loss / len(dataloader), total_acc / len(dataloader)

# ------------------------------
# Full training run with W&B logging
# ------------------------------
def run_attention_training(config=None, train_loader=None, dev_loader=None, input_vocab=None, target_vocab=None):
    with wandb.init(config=config):
        config = wandb.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        encoder = Encoder(
            vocab_size=len(input_vocab),
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            cell_type=config.cell_type
        )

        decoder = AttentionDecoder(
            vocab_size=len(target_vocab),
            embedding_dim=config.embedding_dim,
            enc_hidden_dim=config.hidden_dim,
            dec_hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            cell_type=config.cell_type
        )

        model = Seq2Seq(encoder, decoder, device, use_attention=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=target_vocab[PAD_TOKEN])

        best_val_acc = 0.0

        for epoch in range(config.epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, target_vocab[PAD_TOKEN], config.teacher_forcing)
            val_loss, val_acc = evaluate(model, dev_loader, criterion, device, target_vocab[PAD_TOKEN])

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc
            })

            print(f"[Epoch {epoch + 1:02}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "best_model_attn.pth")

# ------------------------------
# Sweep Configuration Example
# ------------------------------
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "embedding_dim": {"values": [32, 64, 128, 256]},
        "hidden_dim": {"values": [32, 64, 128, 256]},
        "dropout": {"values": [0.0, 0.2, 0.4, 0.5]},
        "cell_type": {"values": ["RNN", "GRU", "LSTM"]},
        "lr": {"values": [0.001, 0.0005, 0.0001]},
        "teacher_forcing": {"values": [0.5, 0.7]},
        "epochs": {"value": 10}
    }
}

# Usage:
# sweep_id = wandb.sweep(sweep_config, project="TransliterationAttention")
# wandb.agent(sweep_id, function=lambda: run_attention_training(...), count=25)
