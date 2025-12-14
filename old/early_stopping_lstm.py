import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os

# -- Konfiguráció --
EPOCHS = 500  # Magasabb epoch szám, hogy az early stopping be tudjon avatkozni
LEARNING_RATE = 0.01
PATIENCE = 32 # Türelem az early stoppinghoz
BEST_MODEL_PATH = "best_model.pth"
DATA_PATH = "data/processed"

# 1. Adatok betöltése és előkészítése

# Adatok betöltése .npy fájlokból
print("Adatok betöltése a 'data/processed' könyvtárból...")
try:
    X_train_np = np.load(os.path.join(DATA_PATH, 'bullflag_train_data.npy'), allow_pickle=True)
    y_train_np = np.load(os.path.join(DATA_PATH, 'bullflag_train_labels.npy'), allow_pickle=True)
    X_val_np = np.load(os.path.join(DATA_PATH, 'bullflag_val_data.npy'), allow_pickle=True)
    y_val_np = np.load(os.path.join(DATA_PATH, 'bullflag_val_labels.npy'), allow_pickle=True)
    X_test_np = np.load(os.path.join(DATA_PATH, 'bullflag_test_data.npy'), allow_pickle=True)
    y_test_np = np.load(os.path.join(DATA_PATH, 'bullflag_test_labels.npy'), allow_pickle=True)
except FileNotFoundError as e:
    print(f"Hiba: Nem található a szükséges adatfájl. Ellenőrizd a '{DATA_PATH}' könyvtárat.")
    print(f"Részletek: {e}")
    exit()

# Címkék integerre való leképezése
all_labels = np.concatenate((y_train_np, y_val_np, y_test_np))
unique_labels = np.unique(all_labels)
label_to_int = {label: i for i, label in enumerate(unique_labels)}
# int_to_label = {i: label for i, label in enumerate(unique_labels)} # Opcionális, későbbi elemzéshez

print("Címkék leképezése:")
print(label_to_int)

# Modell paraméterek dinamikus beállítása
NUM_CLASSES = len(label_to_int)

# Feldolgozó függvények
def process_features(np_array_x):
    """Bemeneti jellemzők (X) feldolgozása: paddolt tenzorrá alakítás."""
    if np_array_x.dtype == 'object':
        x_tensors = [torch.from_numpy(seq).float() for seq in np_array_x]
        x_padded = torch.nn.utils.rnn.pad_sequence(x_tensors, batch_first=True, padding_value=0.0)
    else:
        x_padded = torch.from_numpy(np_array_x).float()
    return x_padded

def process_labels(np_array_y, mapping):
    """Címkék (y) feldolgozása: string -> integer, majd tenzorrá alakítás."""
    y_int = np.array([mapping[label] for label in np_array_y])
    return torch.from_numpy(y_int).long()

# Adatok átalakítása a modell számára megfelelő formátumra
X_train = process_features(X_train_np)
y_train = process_labels(y_train_np, label_to_int)

X_val = process_features(X_val_np)
y_val = process_labels(y_val_np, label_to_int)

X_test = process_features(X_test_np)
y_test = process_labels(y_test_np, label_to_int)

# Jellemzők számának kinyerése
_, TIME_STEPS, FEATURES = X_train.shape

# -- Vége: Adatok betöltése és előkészítése --

print("\nAdat formátumok (feldolgozás után):")
print(f"Training X: {X_train.shape}, y: {y_train.shape}")
print(f"Validation X: {X_val.shape}, y: {y_val.shape}")
print(f"Test X: {X_test.shape}, y: {y_test.shape}")
print(f"Osztályok száma: {NUM_CLASSES}, Jellemzők száma: {FEATURES}")
print("-" * 30)

from torch.utils.data import TensorDataset, DataLoader

# -- Modell, Loss, Optimizer --
model = LSTMOverfitModel(input_size=FEATURES, hidden_size=32, num_classes=NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -- DataLoaderek létrehozása a mini-batch tanításhoz --
BATCH_SIZE = 16
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Tanítás Early Stoppinggal és Mini-Batchekkel
train_loss_history = []
val_loss_history = []
best_val_loss = float('inf')
epochs_no_improve = 0
best_epoch = 0

print("Tanítás megkezdése Early Stoppinggal és Mini-Batchekkel...")
for epoch in range(EPOCHS):
    # --- Training Fázis ---
    model.train()
    running_train_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
    
    epoch_train_loss = running_train_loss / len(train_loader)
    train_loss_history.append(epoch_train_loss)

    # --- Validation Fázis ---
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            
    epoch_val_loss = running_val_loss / len(val_loader)
    val_loss_history.append(epoch_val_loss)

    if (epoch + 1) % 5 == 0: # Gyakoribb kiíratás a több lépés miatt
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    # --- Early Stopping Logika ---
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        epochs_no_improve = 0
        best_epoch = epoch + 1
    else:
        epochs_no_improve += 1

    if epochs_no_improve == PATIENCE:
        print(f"\nEarly stopping a(z) {epoch+1}. epoch-ban. A legjobb epoch: {best_epoch}.")
        print(f"A legjobb validation loss: {best_val_loss:.4f}")
        break

print("A tanítás befejeződött.")
print("-" * 30)


# 4. Kiértékelés a Test adatokon a legjobb modellel
print("Kiértékelés a test adatokon a legjobb modellel...")
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1-Score (Weighted): {f1:.4f}")
print("-" * 30)


# 5. Vizualizáció
print("Loss görbék kirajzolása...")
plt.figure(figsize=(12, 7))
plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.axvline(best_epoch - 1, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
plt.title('Training & Validation Loss (Mini-Batch)')
plt.xlabel('Epoch')
plt.ylabel('Loss (CrossEntropy)')
# plt.yscale('log') # Log scale might not be ideal if loss gets very low or validation loss fluctuates
plt.legend()
plt.grid(True)
plt.savefig('early_stopping_loss.png')

print("A grafikon elmentve: 'early_stopping_loss.png'.")

# Takarítás: töröljük a mentett modellfájlt
if os.path.s.path.exists(BEST_MODEL_PATH):
    os.remove(BEST_MODEL_PATH)

