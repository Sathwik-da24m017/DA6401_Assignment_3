import torch
import torch.nn as nn

# ----------------------------------------------------
# Encoder for Vanilla and Attention-Based Seq2Seq
# ----------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.0, cell_type="LSTM"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        rnn_cls = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[cell_type]
        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.cell_type = cell_type

    def forward(self, x):
        embedded = self.embedding(x)                          # [batch, src_len, emb_dim]
        outputs, hidden = self.rnn(embedded)                  # outputs: [batch, src_len, hidden_dim]
        return outputs, hidden                                # both needed for attention

# ----------------------------------------------------
# Decoder for Vanilla Seq2Seq Model
# ----------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.0, cell_type="LSTM"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        rnn_cls = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[cell_type]
        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.cell_type = cell_type

    def forward(self, x, hidden):
        x = x.unsqueeze(1)                            # [batch] -> [batch, 1]
        embedded = self.embedding(x)                  # [batch, 1, emb_dim]
        output, hidden = self.rnn(embedded, hidden)   # output: [batch, 1, hidden_dim]
        prediction = self.fc_out(output.squeeze(1))   # [batch, vocab_size]
        return prediction, hidden

# ----------------------------------------------------
# Attention Mechanism (Bahdanau Style)
# ----------------------------------------------------
class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.attn = nn.Linear(enc_dim + dec_dim, dec_dim)
        self.v = nn.Linear(dec_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch, dec_dim]
        # encoder_outputs: [batch, src_len, enc_dim]
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)

        hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch, src_len, dec_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)                       # [batch, src_len]
        return torch.softmax(attention, dim=1)                     # [batch, src_len]

# ----------------------------------------------------
# Decoder with Attention
# ----------------------------------------------------
class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_dim, dec_dim, dropout=0.0, cell_type="LSTM"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = BahdanauAttention(enc_dim, dec_dim)

        self.rnn_input_dim = embedding_dim + enc_dim
        rnn_cls = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[cell_type]
        self.rnn = rnn_cls(self.rnn_input_dim, dec_dim, batch_first=True)
        self.fc_out = nn.Linear(enc_dim + dec_dim + embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.cell_type = cell_type

    def forward(self, x, hidden, encoder_outputs):
        x = x.unsqueeze(1)                               # [batch] -> [batch, 1]
        embedded = self.dropout(self.embedding(x))      # [batch, 1, emb_dim]

        # Get last layer hidden state
        dec_hidden = hidden[0][-1] if self.cell_type == "LSTM" else hidden[-1]
        attn_weights = self.attention(dec_hidden, encoder_outputs)     # [batch, src_len]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, enc_dim]

        rnn_input = torch.cat((embedded, context), dim=2)              # [batch, 1, emb+context]
        output, hidden = self.rnn(rnn_input, hidden)

        output = output.squeeze(1)     # [batch, dec_dim]
        context = context.squeeze(1)   # [batch, enc_dim]
        embedded = embedded.squeeze(1)

        logits = self.fc_out(torch.cat((output, context, embedded), dim=1))  # [batch, vocab_size]
        return logits, hidden, attn_weights

# ----------------------------------------------------
# Seq2Seq Wrapper for both Vanilla and Attention-Based models
# ----------------------------------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, use_attention=False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.use_attention = use_attention

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)
        attentions = torch.zeros(batch_size, trg_len, src.size(1)).to(self.device) if self.use_attention else None

        encoder_outputs, hidden = self.encoder(src)
        input_token = trg[:, 0]  # start with <sos>

        for t in range(1, trg_len):
            if self.use_attention:
                output, hidden, attn_weights = self.decoder(input_token, hidden, encoder_outputs)
                attentions[:, t] = attn_weights
            else:
                output, hidden = self.decoder(input_token, hidden)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input_token = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return (outputs, attentions) if self.use_attention else outputs
