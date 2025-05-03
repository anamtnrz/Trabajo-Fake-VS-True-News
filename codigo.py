#IMPORTAR LIBRERÍAS

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


#CARGAR ARCHIVOS

from google.colab import files
uploaded = files.upload()


#COMBINAR AMBOS DATASETS

true_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')


true_df['label'] = 1
fake_df['label'] = 0
df = pd.concat([true_df, fake_df], ignore_index=True)

df.head(5)


#CONVERTIR TEXTO A VECTORES

vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['title']).toarray()
y = df['label'].values


#DIVISIÓN ALEATORIA DEL DATASET EN TRAIN Y TEST

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, df.index, test_size=0.2, random_state=42)

#6-CREAR EL MODELO DE REDES NEURONALES CON PYTORCH

# Creamos los tensores
X_train_tensor = torch.tensor(X_train).float()
y_train_tensor = torch.tensor(y_train).float().unsqueeze(1)
X_test_tensor = torch.tensor(X_test).float()
y_test_tensor = torch.tensor(y_test).float().unsqueeze(1)


# Cálculo de pesos para la pérdida
weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
weights_tensor = torch.tensor(weights, dtype=torch.float32)


# Dataset y Dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# MODELO A(64 neuronas, 2 capas, con dropout y residual)
class FakeNewsClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FakeNewsClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out

# Crear el modelo
model = FakeNewsClassifier(input_dim=X.shape[1])

# Calcular pesos para clases desbalanceadas
class_counts = np.bincount(y_train)
pos_weight = torch.tensor([class_counts[0] / class_counts[1]], dtype=torch.float32)

# Función de pérdida y optimizador
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

#ENTRENAR EL MODELO

for epoch in range(5):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")


#EVALUACIÓN DEL MODELO

odel.eval()
with torch.no_grad():
    y_pred_logits = model(X_test_tensor)
    y_pred_probs = torch.sigmoid(y_pred_logits)
    y_pred = (y_pred_probs >= 0.5).int().numpy()
    y_true = y_test_tensor.numpy()

print("\nAccuracy MODELO A(El original):", accuracy_score(y_true, y_pred))
print("\nClassification Report MODELO A(El original):\n", classification_report(y_true, y_pred))


# Función para entrenar y evaluar
def train_and_evaluate(model):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.sigmoid(logits)
        y_pred = (probs >= 0.5).int().numpy()
        loss_val = criterion(logits, y_test_tensor).item()
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
    return loss_val, acc, f1

# MODELO B. Modelo de 0 capas (regresión logística)
class ModeloB(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.out = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.out(x)
        
# MODELO C. 64 neuronas, 2 capas, con dropout y residual
class ModeloC(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x1 = self.relu(self.fc1(x))
        x2 = self.relu(self.fc2(x1))
        x2 = self.dropout(x2)
        x2 += x1  # conexión residual
        return self.out(x2)


# MODELO D. 64 neuronas, 2 capas, SIN dropout
class ModeloD(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)


# MODELO E. 256 neuronas, 2 capas, con dropout y residual
class ModeloE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x1 = self.relu(self.fc1(x))
        x2 = self.relu(self.fc2(x1))
        x2 = self.dropout(x2)
        x2 += x1  # residual
        return self.out(x2)


# Entrenar y evaluar
input_dim = X.shape[1]
results = {}

models = {
    "MODELO B. Modelo de 0 capas": ModeloB(input_dim),
    "MODELO C. 64 neuronas, 2 capas, con dropout y residual": ModeloC(input_dim),
    "MODELO D. 64 neuronas, 2 capas, SIN dropout": ModeloD(input_dim),
    "MODELO E. 256 neuronas, 2 capas, con dropout y residual": ModeloE(input_dim),
}

for name, model in models.items():
    bce, acc, f1 = train_and_evaluate(model)
    results[name] = (bce, acc, f1)



#Mostrar resultados
print("\n Resultados comparativos:")
for name, (bce, acc, f1) in results.items():
    print(f"\n🔹{name}")
    print(f"BCE: {bce:.4f}, Accuracy: {acc:.4f}, F1-score: {f1:.4f}")


#Función para predecir y mostrar resultado
def clasificar_titulo(titulo):
    entrada_bow = vectorizer.transform([titulo]).toarray()
    entrada_tensor = torch.tensor(entrada_bow).float()

    with torch.no_grad():
        pred_logit = model(entrada_tensor)
        prob = torch.sigmoid(pred_logit).item()

    print(f"\n Título ingresado de la noticia: {titulo}")
    print(f" Probabilidad de ser *true*: {prob:.2%}")

    if prob >= 0.5:
        print("Resultado: Es una *true* new (noticia verdadera)")
    else:
        print("Resultado: Es una *fake* new (noticia falsa)")

    # Si es verdadera, buscar contenido similar
    if prob >= 0.5:
        resultados = df[df['title'].str.lower().str.contains(titulo.lower()) & (df['label'] == 1)]
        if not resultados.empty:
            print("\nTexto relacionado encontrado:\n")
            print(resultados.iloc[0]['text'][:1000], '...')
        else:
            print("No se encontró el texto completo de esta noticia en el dataset.")

# Bucle interactivo (funciona bien en Google Colab)
while True:
    entrada = input("\n Hola! Introduce el título de la noticia y comprobaremos si es falsa o no (o escribe 'salir'):\n")
    if entrada.strip().lower() in ['salir', 'exit']:
        print("Hasta luego.")
        break
    clasificar_titulo(entrada)





















# Definir el modelo y función de pérdida
class FakeNewsClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FakeNewsClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)  # Sin sigmoid porque usaremos BCEWithLogitsLoss

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out

# Crear el modelo
model = FakeNewsClassifier(input_dim=X.shape[1])

# Calcular pesos para clases desbalanceadas
class_counts = np.bincount(y_train)
pos_weight = torch.tensor([class_counts[0] / class_counts[1]], dtype=torch.float32)

# Función de pérdida y optimizador
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)


#7- ENTRENAR EL MODELO

for epoch in range(5):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

#8-EVALUACIÓN DEL MODELO

model.eval()
with torch.no_grad():
    y_pred_logits = model(X_test_tensor)
    y_pred_probs = torch.sigmoid(y_pred_logits)
    y_pred = (y_pred_probs >= 0.5).int().numpy()
    y_true = y_test_tensor.numpy()

print("\nAccuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred))


#9-INTERACCIÓN CON EL USUARIO
# Función para predecir y mostrar resultado
def clasificar_titulo(titulo):
    entrada_bow = vectorizer.transform([titulo]).toarray()
    entrada_tensor = torch.tensor(entrada_bow).float()

    with torch.no_grad():
        pred_logit = model(entrada_tensor)
        prob = torch.sigmoid(pred_logit).item()

    print(f"\n Título ingresado de la noticia: {titulo}")
    print(f" Probabilidad de ser *true*: {prob:.2%}")

    if prob >= 0.5:
        print("Resultado: Es una *true* new (noticia verdadera)")
    else:
        print("Resultado: Es una *fake* new (noticia falsa)")

    # Si es verdadera, buscar contenido similar
    if prob >= 0.5:
        resultados = df[df['title'].str.lower().str.contains(titulo.lower()) & (df['label'] == 1)]
        if not resultados.empty:
            print("\nTexto relacionado encontrado:\n")
            print(resultados.iloc[0]['text'][:1000], '...')
        else:
            print("⚠️ No se encontró el texto completo de esta noticia en el dataset.")

# Bucle interactivo (funciona bien en Google Colab)
while True:
    entrada = input("\n Hola! Introduce el título de la noticia y comprobaremos si es falsa o no (o escribe 'salir'):\n")
    if entrada.strip().lower() in ['salir', 'exit']:
        print("Hasta luego.")
        break
    clasificar_titulo(entrada)

