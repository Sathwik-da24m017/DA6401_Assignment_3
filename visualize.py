import torch
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Plot a single attention heatmap
# ------------------------------
def plot_attention_heatmap(attn_weights, input_tokens, output_tokens, title=None, save_path=None):
    plt.figure(figsize=(10, 6))
    sns.heatmap(attn_weights, xticklabels=input_tokens, yticklabels=output_tokens, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.xlabel("Input (Roman)")
    plt.ylabel("Output (Telugu)")
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# ------------------------------
# Generate attention weights for given input
# ------------------------------
def extract_attention(model, input_text, input2idx, target2idx, idx2target, max_len=30):
    model.eval()
    device = next(model.parameters()).device
    src_tensor = torch.tensor([input2idx[SOS_TOKEN]] + [input2idx[c] for c in input_text] + [input2idx[EOS_TOKEN]], device=device).unsqueeze(0)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        input_token = torch.tensor([target2idx[SOS_TOKEN]], device=device)

        output_tokens = []
        attention_scores = []

        for _ in range(max_len):
            output, hidden, attn_weights = model.decoder(input_token, hidden, encoder_outputs)
            pred_token = output.argmax(1).item()
            token = idx2target[pred_token]
            if token == EOS_TOKEN:
                break
            output_tokens.append(token)
            attention_scores.append(attn_weights.squeeze(0).cpu().numpy())
            input_token = torch.tensor([pred_token], device=device)

        input_tokens = [SOS_TOKEN] + list(input_text) + [EOS_TOKEN]
        return input_tokens, output_tokens, torch.stack(attention_scores).numpy()

# ------------------------------
# Show 3x3 grid of attention examples
# ------------------------------
def plot_attention_grid(model, test_samples, input2idx, target2idx, idx2target, out_dir="attn_grid"):
    import os
    os.makedirs(out_dir, exist_ok=True)

    for idx, (input_text, _) in enumerate(test_samples[:9]):
        input_tok, output_tok, attn_mat = extract_attention(model, input_text, input2idx, target2idx, idx2target)
        save_path = f"{out_dir}/attention_{idx+1}.png"
        plot_attention_heatmap(attn_mat, input_tok, output_tok, title=f"Sample {idx+1}: {input_text}", save_path=save_path)

    print(f"✅ Saved 3x3 attention visualizations in {out_dir}/")