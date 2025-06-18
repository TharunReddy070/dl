import torch
import torch.nn as nn
import torch.optim as optim


class SimpleRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.rnn(x)           # RNN output: (batch, seq_len, hidden)
        out = out[:, -1, :]            # Take the output of the last time step
        out = self.fc(out)             # Dense layer
        return out

# Hyperparameters
input_size = 8      # Features per time step
hidden_size = 32    # Number of RNN hidden units
output_size = 1     # Binary classification (use >1 for multi-class)
seq_len = 10
batch_size = 16

# Generate random input and target data
X = torch.randn(batch_size, seq_len, input_size)  # Shape: (batch, seq_len, input_size)
y = torch.randint(0, 2, (batch_size, 1)).float()   # Binary targets
print(X)
print(y)

model = SimpleRNNModel(input_size, hidden_size, output_size)
criterion = nn.BCEWithLogitsLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")