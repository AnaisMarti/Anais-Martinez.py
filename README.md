# Anais-Martinez.py
```Python
import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[i] for i in range(10)], dtype=torch.float32)
y = torch.tensor([[i % 2] for i in range(10)], dtype=torch.float32)

class ParidadMLP(nn.Module):
    def __init__(self):
        super(ParidadMLP, self).__init__()
        self.hidden = nn.Linear(1, 5)
        self.relu = nn.ReLU()
        self.output = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

model = ParidadMLP()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    predictions = model(X).round()
    print("Predicciones:", predictions.view(-1).tolist())
    print("Esperado    :", y.view(-1).tolist())
```
# Rama-1
```Python
class ParidadMLP(nn.Module):
    def __init__(self):
        super(ParidadMLP, self).__init__()
        self.hidden = nn.Linear(1, 3)  # <- Cambiado a 3 neuronas
        self.relu = nn.ReLU()
        self.output = nn.Linear(3, 1)  # <- Ajustado a 3
        self.sigmoid = nn.Sigmoid()
    # ... (el resto del código igual al main)
```
