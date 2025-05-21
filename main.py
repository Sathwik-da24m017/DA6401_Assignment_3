import argparse
import torch
from data_loader import load_data, build_vocab, tensorify, get_dataloaders
from models import Encoder, Decoder, AttentionDecoder, Seq2Seq
from evaluate import run_test_evaluation
from visualize import plot_attention_grid
from interactive_vis import save_interactive_attention_html

# ------------------------------
# Load vocabularies and data
# ------------------------------
def prepare_data():
    BASE = "dakshina_dataset_v1.0/te/lexicons/"
    train = load_data(BASE + "te.translit.sampled.train.tsv")
    dev = load_data(BASE + "te.translit.sampled.dev.tsv")
    test = load_data(BASE + "te.translit.sampled.test.tsv")
    
    input_vocab, input2idx, idx2input = build_vocab(train, is_input=True)
    target_vocab, target2idx, idx2target = build_vocab(train, is_input=False)
    loaders = get_dataloaders(train, dev, test, input2idx, target2idx, batch_size=64)
    return train, dev, test, loaders, input2idx, target2idx, idx2target, input_vocab, target_vocab

# ------------------------------
# Main execution logic
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["vanilla", "attention"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    # Load data and vocab
    _, _, test_pairs, (train_loader, dev_loader, test_loader), input2idx, target2idx, idx2target, input_vocab, target_vocab = prepare_data()

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_type == "vanilla":
        encoder = Encoder(len(input_vocab), 64, 256, num_layers=3, dropout=0.5, cell_type="LSTM")
        decoder = Decoder(len(target_vocab), 64, 256, num_layers=3, dropout=0.5, cell_type="LSTM")
        model = Seq2Seq(encoder, decoder, device)
    else:
        encoder = Encoder(len(input_vocab), 256, 256, dropout=0.2, cell_type="LSTM")
        decoder = AttentionDecoder(len(target_vocab), 256, 256, 256, dropout=0.2, cell_type="LSTM")
        model = Seq2Seq(encoder, decoder, device, use_attention=True)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    # Evaluation
    if args.eval:
        run_test_evaluation(model, test_pairs, input2idx, target2idx, idx2target)

    # Static attention visualization
    if args.visualize and args.model_type == "attention":
        plot_attention_grid(model, test_pairs[:9], input2idx, target2idx, idx2target)

    # Interactive visualization
    if args.interactive and args.model_type == "attention":
        from visual_utils import get_multiple_attn_examples
        examples = get_multiple_attn_examples(model, test_pairs[:5], input2idx, target2idx, idx2target)
        save_interactive_attention_html(examples, filename="attention_all.html")

if __name__ == "__main__":
    main()