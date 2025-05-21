import torch
import pandas as pd
from models import Seq2Seq
from data_loader import encode_sequence, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN

# ------------------------------
# Predict a sequence for one input sample
# ------------------------------
def greedy_decode(model, src_tensor, target2idx, idx2target, max_len=30):
    model.eval()
    device = next(model.parameters()).device
    src_tensor = src_tensor.unsqueeze(0).to(device)  # [1, src_len]

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        input_token = torch.tensor([target2idx[SOS_TOKEN]], device=device)  # Start with <sos>
        outputs = []

        for _ in range(max_len):
            if model.use_attention:
                output, hidden, _ = model.decoder(input_token, hidden, encoder_outputs)
            else:
                output, hidden = model.decoder(input_token, hidden)

            pred_token = output.argmax(1).item()
            if idx2target[pred_token] == EOS_TOKEN:
                break
            outputs.append(idx2target[pred_token])
            input_token = torch.tensor([pred_token], device=device)

    return ''.join(outputs)

# ------------------------------
# Run predictions on test set and export
# ------------------------------
def run_test_evaluation(model, test_pairs, input2idx, target2idx, idx2target, output_csv="test_predictions.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    results = []
    for src_text, tgt_text in test_pairs:
        encoded = encode_sequence(src_text, input2idx)
        src_tensor = torch.tensor(encoded, dtype=torch.long)
        pred = greedy_decode(model, src_tensor, target2idx, idx2target)
        results.append([src_text, tgt_text, pred, pred == tgt_text])

    df = pd.DataFrame(results, columns=["Input (Roman)", "Ground Truth (Telugu)", "Prediction (Telugu)", "Match"])
    df.to_csv(output_csv, index=False)
    print(f"✅ Predictions saved to {output_csv} | Accuracy: {df['Match'].mean()*100:.2f}%")
    return df
