
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 1. Adatgenerálás
BATCH_SIZE = 32
TIME_STEPS = 128
FEATURES = 4
NUM_CLASSES = 7
EPOCHS = 300
LEARNING_RATE = 0.01

# Bemeneti adatok
X_dummy = torch.randn(BATCH_SIZE, TIME_STEPS, FEATURES)

# Címkék generálása (integer formátumban a CrossEntropyLoss-hoz)
y_dummy_int = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

# One-hot enkódolt célváltozók (opcionális, a modell tanításához nem kell)
y_dummy_onehot = nn.functional.one_hot(y_dummy_int, num_classes=NUM_CLASSES).float()

print(f"Bemeneti adat (X) formátuma: {X_dummy.shape}")
print(f"Célváltozó (y) formátuma (one-hot): {y_dummy_onehot.shape}")
print("-" * 30)

# 2. Modellépítés (LSTM)
class LSTMOverfitModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMOverfitModel, self).__init__()
        # LSTM réteg dropout nélkül
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=0.0  # Regularizáció kikapcsolása
        )
        # Kimeneti réteg
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Az LSTM kimenete: output, (hidden_state, cell_state)
        # Nekünk csak az utolsó időpontban kapott kimenet kell a teljes szekvenciából.
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        
        # Átadás a teljesen összekötött rétegnek
        output = self.fc(last_time_step_out)
        return output

# Modell inicializálása
model = LSTMOverfitModel(input_size=FEATURES, hidden_size=32, num_classes=NUM_CLASSES)
print("A modell felépítése:")
print(model)
print("-" * 30)

# 3. Túltanítás (Overfitting)
# Loss és optimizer definiálása
# A CrossEntropyLoss magában foglalja a Softmax aktivációt, ezért a modell végére nem kell külön Softmax réteg.
# A CrossEntropyLoss integer (nem one-hot) címkéket vár bemenetként.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss_history = []

print("Tanítás megkezdése...")
for epoch in range(EPOCHS):
    model.train() 
    
    # Nullázzuk a gradienseket
    optimizer.zero_grad()
    
    # Előrecsatolás
    outputs = model(X_dummy)
    
    # Loss számítása
    loss = criterion(outputs, y_dummy_int)
    
    # Visszaterjesztés és optimalizálás
    loss.backward()
    optimizer.step()
    
    # Loss érték mentése a vizualizációhoz
    loss_history.append(loss.item())
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")
        if loss.item() < 0.001:
            print(f"\nA loss elérte a 0.001 alatti értéket a(z) {epoch+1}. epoch-ban. A tanítás leáll.")
            # Hogy a grafikon szép legyen, feltöltjük a maradék epochot az utolsó értékkel
            remaining_epochs = EPOCHS - (epoch + 1)
            if remaining_epochs > 0:
                loss_history.extend([loss.item()] * remaining_epochs)
            break

print("A tanítás befejeződött.")
print("-" * 30)


# 4. Vizualizáció
print("Loss görbe kirajzolása...")
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.title('Training Loss az Epochok során')
plt.xlabel('Epoch')
plt.ylabel('Loss (CrossEntropy)')
plt.grid(True)
plt.yscale('log') # Logaritmikus skála a jobb láthatóságért
plt.axhline(y=0.001, color='r', linestyle='--', label='Cél (Loss < 0.001)')
plt.legend()
plt.savefig('loss_curve.png')

print("\nA script futása befejeződött. A grafikont a 'loss_curve.png' fájlba mentettem.")

