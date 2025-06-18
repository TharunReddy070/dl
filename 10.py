import torch
import torch.nn as nn
import torch.optim as optim

text = "hello world, this is a simple text generation using LSTMs."

# character vocabulary
chars = sorted(set(text))  # Unique characters
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# Convert text into sequences
seq_length = 10
data_X, data_Y = [], []
for i in range(len(text) - seq_length):
    input_seq = text[i:i+seq_length]
    target_char = text[i+seq_length]
    data_X.append([char_to_idx[c] for c in input_seq])
    data_Y.append(char_to_idx[target_char])

# Convert to PyTorch tensors
X_train = torch.tensor(data_X, dtype=torch.long)
y_train = torch.tensor(data_Y, dtype=torch.long)


class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Take last output for prediction
        return out

vocab_size = len(chars)
embed_size = 16
hidden_size = 128
model = TextLSTM(vocab_size, embed_size, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 50
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()


def generate_text(start_seq, length=50):
    model.eval()
    input_seq = [char_to_idx[c] for c in start_seq]
    input_tensor = torch.tensor([input_seq], dtype=torch.long)
    generated_text = start_seq
    for _ in range(length):
        with torch.no_grad():
            output = model(input_tensor)
            predicted_idx = torch.argmax(output, dim=1).item()
            predicted_char = idx_to_char[predicted_idx]
            generated_text += predicted_char
            input_tensor = torch.tensor([[*input_seq[1:], predicted_idx]], dtype=torch.long)
            input_seq = input_seq[1:] + [predicted_idx]
    return generated_text


print("\nGenerated Text:")
print(generate_text("hello wor", 50))