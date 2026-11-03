# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv (3.12.11)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook 2 — Entrenamiento y comparación de modelos
#
# Entrenamos una **CNN custom** (Modelo A), la evaluamos junto a un modelo
# preentrenado **InsightFace / ArcFace** (Modelo B) sobre el mismo conjunto de
# test, y elegimos el mejor para integrar en la aplicación.
#
# > **Entradas esperadas de la Parte 1:**
# > - `output/train.csv`, `output/val.csv`, `output/test.csv` (columnas `path`, `identity`)
# > - `output/aligned_112/{split}/{identity}/{nombre}.png` (rostros alineados 112×112)

# %% [markdown]
# ## Equipo
# - Alumno 1 : **Alejandro Armas**
# - Alumno 2 : **Gabriel Soda**

# %% [markdown]
# ## Evaluación con identidades reservadas
#
# Los splits actuales permiten evaluar **identificación closed-set** sobre las 100 identidades con las que se entrenó la CNN, usando imágenes distintas en cada conjunto.
#
# ### Test adicional de verificación
#
# Para evaluar verificación con identidades que la CNN nunca vio, se reservan tres personas que habían sido descartadas por tener menos de 100 imágenes:
#
# | Identidad | Archivos en el dataset original |
# |---|---:|
# | Lionel Messi | 86 |
# | Cristiano Ronaldo | 98 |
# | Mark Zuckerberg | 95 |
#
# Ninguna de estas identidades aparece en los CSV de train, validación ni test. Sus imágenes quedan fuera del entrenamiento y se usan en un **test adicional de verificación**, común a la CNN y al modelo preentrenado. En este test se comparan los embeddings de pares de imágenes, tanto de la misma persona como de personas distintas. Estas identidades no se agregan a las 100 clases de salida de la CNN.
#
# ### Pendientes de preparación
#
# La preparación de este conjunto todavía no está hecha. Falta:
#
# - revisar etiquetas y duplicados;
# - aplicar filtros de calidad;
# - detectar y alinear rostros con el mismo procedimiento que el resto del dataset;
# - no aplicar augmentation;
# - exportar un csv separado.
#
# Algo que también se debe tener en cuenta es que las cantidades de la tabla corresponden a archivos originales, no a rostros aceptados después del filtrado.
#
# ### Rechazo de desconocidos en identificación
#
# Para evaluar el rechazo de personas desconocidas, también hay que definir una galería y un conjunto de consultas. Una identidad no vista durante el entrenamiento puede registrarse en la galería; en ese escenario, solo se consideran desconocidas las identidades ausentes de la galería.

# %%
import os
# Necesario para GPU ROCm 6700xt
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

# %%
import sys
import time
from collections import Counter
from pathlib import Path

import altair as alt
import cv2
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

disable_max_rows = getattr(alt.data_transformers, "disable_max_rows", None)
if callable(disable_max_rows):
    alt.data_transformers.disable_max_rows()


# %%
# --- Configuración ----------------------------------------------------------
DATA_DIR = Path("data/images")
OUTPUT_DIR = Path("output")
ALIGNED_DIR = OUTPUT_DIR / "aligned_112"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FACE_SIZE = 112
DETECTION_THRESHOLD = 0.60
DETECTION_SIZE = (160, 160)

SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 128
EPOCHS = 100
PATIENCE = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Dispositivo: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## Carga de splits y codificación de etiquetas

# %%
for split_name, csv_path in [("train", OUTPUT_DIR / "train.csv"),
                             ("val",   OUTPUT_DIR / "val.csv"),
                             ("test",  OUTPUT_DIR / "test.csv")]:
    if not csv_path.exists():
        raise RuntimeError(f"No se encontró {csv_path}. Ejecutá primero la notebook 1.")

train_df = pl.read_csv(OUTPUT_DIR / "train.csv")
val_df   = pl.read_csv(OUTPUT_DIR / "val.csv")
test_df  = pl.read_csv(OUTPUT_DIR / "test.csv")

for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    assert df.height > 0, f"El conjunto {split_name} está vacío"
    assert df["path"].null_count() == 0
    assert df["identity"].null_count() == 0
    assert df["path"].n_unique() == df.height, f"Rutas duplicadas en {split_name}"
    print(f"{split_name}: {df.height} imágenes, {df['identity'].n_unique()} identidades")

print(train_df.head())

# %%
label_encoder = LabelEncoder()
label_encoder.fit(sorted(train_df["identity"].unique().to_list()))
NUM_CLASSES = len(label_encoder.classes_)
assert set(val_df["identity"]) == set(label_encoder.classes_)
assert set(test_df["identity"]) == set(label_encoder.classes_)
print(f"Clases (identidades): {NUM_CLASSES}")


# Map identity → label en los 3 splits
def add_label(df):
    return df.with_columns(
        pl.Series("label",
                  label_encoder.transform(df["identity"].to_list()),
                  dtype=pl.Int64)
    )

train_df = add_label(train_df)
val_df   = add_label(val_df)
test_df  = add_label(test_df)

# %% [markdown]
# ## Dataset / DataLoader de PyTorch
#
# Leemos los recortes alineados 112×112 producidos por la notebook 1.  No se
# aplica augmentation aquí: el pipeline de preprocesamiento (detección + alineación
# + augmentation si corresponde) es responsabilidad de la Parte 1.

# %%
class AlignedDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_dir):
        self.df = df
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)
        img_path = self.image_dir / row["path"]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"No se pudo leer: {img_path}")
        if img_bgr.shape != (FACE_SIZE, FACE_SIZE, 3):
            raise ValueError(f"Tamaño inesperado en {img_path}: {img_bgr.shape}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        return img, int(row["label"])


train_ds = AlignedDataset(train_df, ALIGNED_DIR)
val_ds = AlignedDataset(val_df, ALIGNED_DIR)
test_ds = AlignedDataset(test_df, ALIGNED_DIR)

# %% [markdown]
# ## Agrupación de imágenes con DataLoader
#
# Cada lote contiene hasta 64 imágenes. Se mezcla solamente train, con un
# generador de semilla fija. num_workers=0 carga en el proceso de la notebook.
# La memoria fijada se habilita solo cuando se utiliza CUDA.

# %%
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    pin_memory=DEVICE.type == "cuda",
    generator=torch.Generator().manual_seed(SEED),
)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    pin_memory=DEVICE.type == "cuda",
)
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    pin_memory=DEVICE.type == "cuda",
)

print(f"Batches — train: {len(train_loader)} | val: {len(val_loader)} | test: {len(test_loader)}")

# %% [markdown]
# ## Visualización del lote y sus etiquetas
#
# OpenCV lee BGR; el Dataset convierte a RGB, cambia de HWC a CHW y escala
# a [0, 1]. Para Matplotlib volvemos a HWC. Se muestran hasta 16 imágenes,
# que pueden ser originales o aumentadas guardadas.
# Se usa un cargador separado para no avanzar el generador del entrenamiento.

# %%
preview_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    generator=torch.Generator().manual_seed(SEED),
)
images, labels = next(iter(preview_loader))
assert images.shape == (min(BATCH_SIZE, len(train_ds)), 3, FACE_SIZE, FACE_SIZE)
assert images.dtype == torch.float32
assert labels.dtype == torch.int64
assert 0 <= images.min() <= images.max() <= 1
assert 0 <= labels.min() <= labels.max() < NUM_CLASSES
print("Imágenes:", images.shape, images.dtype)
print("Etiquetas:", labels.shape, labels.dtype)

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    if i < len(images):
        ax.imshow(images[i].permute(1, 2, 0).numpy())
        ax.set_title(label_encoder.classes_[labels[i].item()], fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3.1 Modelo A — CNN Custom
#
# ### 3.1.1 Arquitectura

# %%
class ModelA(nn.Module):
    """CNN custom simple: 3 bloques conv + embedding + clasificador."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 32),   # 56x56
            self._conv_block(32, 64),  # 28x28
            self._conv_block(64, 128), # 14x14
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc_embedding = nn.Linear(128, 128)
        self.bn_embedding = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout(x)
        emb = self.fc_embedding(x)
        emb = self.bn_embedding(emb)
        emb = F.normalize(emb, p=2, dim=1)
        return self.classifier(emb), emb

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[1]


class ModelA2(nn.Module):
    """ResNet simplificado: stem + 3 bottleneck blocks."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # layer1  (112x112)
        self.b1 = nn.Sequential(
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, 1, bias=False), nn.BatchNorm2d(128),
        )
        self.s1 = nn.Sequential(nn.Conv2d(32, 128, 1, bias=False), nn.BatchNorm2d(128))
        # layer2  (112 -> 56)
        self.b2 = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 1, bias=False), nn.BatchNorm2d(256),
        )
        self.s2 = nn.Sequential(nn.Conv2d(128, 256, 1, stride=2, bias=False), nn.BatchNorm2d(256))
        # layer3  (56 -> 28)
        self.b3 = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, 1, bias=False), nn.BatchNorm2d(512),
        )
        self.s3 = nn.Sequential(nn.Conv2d(256, 512, 1, stride=2, bias=False), nn.BatchNorm2d(512))
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc_embedding = nn.Linear(512, 128)
        self.bn_embedding = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.relu(self.b1(x) + self.s1(x))
        x = self.relu(self.b2(x) + self.s2(x))
        x = self.relu(self.b3(x) + self.s3(x))
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout(x)
        emb = self.fc_embedding(x)
        emb = self.bn_embedding(emb)
        emb = F.normalize(emb, p=2, dim=1)
        return self.classifier(emb), emb

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[1]


class ModelA3(nn.Module):
    """ResNeXt simplificado: stem + 3 grouped-conv bottleneck blocks."""

    def __init__(self, num_classes: int, cardinality: int = 32, base_width: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        def _w(mid: int) -> int:
            return int(mid * (base_width / 64.0)) * cardinality

        # layer1
        w1 = _w(32)
        self.b1 = nn.Sequential(
            nn.Conv2d(32, w1, 1, bias=False), nn.BatchNorm2d(w1), nn.ReLU(inplace=True),
            nn.Conv2d(w1, w1, 3, padding=1, groups=cardinality, bias=False), nn.BatchNorm2d(w1), nn.ReLU(inplace=True),
            nn.Conv2d(w1, 64, 1, bias=False), nn.BatchNorm2d(64),
        )
        self.s1 = nn.Sequential(nn.Conv2d(32, 64, 1, bias=False), nn.BatchNorm2d(64))
        # layer2
        w2 = _w(64)
        self.b2 = nn.Sequential(
            nn.Conv2d(64, w2, 1, bias=False), nn.BatchNorm2d(w2), nn.ReLU(inplace=True),
            nn.Conv2d(w2, w2, 3, stride=2, padding=1, groups=cardinality, bias=False), nn.BatchNorm2d(w2), nn.ReLU(inplace=True),
            nn.Conv2d(w2, 128, 1, bias=False), nn.BatchNorm2d(128),
        )
        self.s2 = nn.Sequential(nn.Conv2d(64, 128, 1, stride=2, bias=False), nn.BatchNorm2d(128))
        # layer3
        w3 = _w(128)
        self.b3 = nn.Sequential(
            nn.Conv2d(128, w3, 1, bias=False), nn.BatchNorm2d(w3), nn.ReLU(inplace=True),
            nn.Conv2d(w3, w3, 3, stride=2, padding=1, groups=cardinality, bias=False), nn.BatchNorm2d(w3), nn.ReLU(inplace=True),
            nn.Conv2d(w3, 256, 1, bias=False), nn.BatchNorm2d(256),
        )
        self.s3 = nn.Sequential(nn.Conv2d(128, 256, 1, stride=2, bias=False), nn.BatchNorm2d(256))
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc_embedding = nn.Linear(256, 128)
        self.bn_embedding = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.relu(self.b1(x) + self.s1(x))
        x = self.relu(self.b2(x) + self.s2(x))
        x = self.relu(self.b3(x) + self.s3(x))
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout(x)
        emb = self.fc_embedding(x)
        emb = self.bn_embedding(emb)
        emb = F.normalize(emb, p=2, dim=1)
        return self.classifier(emb), emb

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[1]


# --- Instancia inicial (A) ---------------------------------------------------
model = ModelA(NUM_CLASSES).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Modelo base: A — CNN custom simple")
print(f"Parámetros entrenables: {n_trainable:,}")
print(f"Número de clases: {NUM_CLASSES}")



# %% [markdown]
# **Decisiones de diseño (modelo simplificado):**
# - **A:** CNN custom con bloques conv simples (3×3 + BatchNorm + ReLU + MaxPool).
#   Liviana y rápida.
# - **A2:** ResNet con *bottlenecks* (1×1 → 3×3 → 1×1) + *skip connections*.
#   Permite entrenar más profundo sin desvanecimiento de gradientes.
# - **A3:** ResNeXt con *grouped convolutions* (cardinality=32). Explora si la
#   diversidad de transformaciones mejora la separación de identidades respecto a A2.
# - **Global Average Pooling**, **Dropout**, **BatchNorm** en embedding y
#   **L2-normalización** son comunes a todas las variantes.

# %% [markdown]
# ### 3.1.2 Hiperparámetros y entrenamiento
#
# El archivo `*_best.pth` conserva el modelo con mayor accuracy de validación.
# El archivo `*_checkpoint.pt` guarda el estado de la última época completada,
# incluso si la validación no mejoró: modelo, optimizador, scheduler e historial.
# Al reanudar se continúa en la época siguiente salvo que ya se haya alcanzado el early stopping, en ese caso
# no se agregan épocas. Una interrupción a mitad de época obliga a repetirla.
# Si solo existe el mejor modelo, se recuperan sus pesos y se reinician el
# optimizador y el historial. No se restaura la secuencia aleatoria exacta.

# %%
def train_model(model, model_pth, ckpt_path, epochs=EPOCHS, patience=PATIENCE):
    """Guarda el mejor modelo y un checkpoint de cada época completada."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_acc = float("-inf")
    patience_counter = 0
    history = []
    start_epoch = 1

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]
        patience_counter = ckpt["patience_counter"]
        history = ckpt["history"]
        print(f"Restaurado desde checkpoint: epoch {start_epoch - 1}, best_val_acc={best_val_acc:.4f}")
    elif model_pth.exists():
        # Cargar pesos sin reemplazar el modelo asociado al optimizador.
        saved_model = torch.load(model_pth, map_location=DEVICE, weights_only=False)
        model.load_state_dict(saved_model.state_dict())
        del saved_model
        model.eval()
        val_loss = val_correct = val_total = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE, non_blocking=True), lbls.to(DEVICE, non_blocking=True)
                logits, _ = model(imgs)
                loss = criterion(logits, lbls)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (logits.argmax(dim=1) == lbls).sum().item()
                val_total += lbls.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total
        best_val_acc = val_acc
        print(f"Cargado .pth existente. Computado val_acc={best_val_acc:.4f} → best_val_acc")

    if patience_counter >= patience:
        print("El checkpoint ya alcanzó early stopping, no se agregan épocas.")
        return best_val_acc, history

    print(f"Iniciando entrenamiento desde epoch {start_epoch}...")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = train_correct = train_total = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE, non_blocking=True), lbls.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            logits, _ = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            train_correct += (logits.argmax(dim=1) == lbls).sum().item()
            train_total += lbls.size(0)
        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss = val_correct = val_total = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE, non_blocking=True), lbls.to(DEVICE, non_blocking=True)
                logits, _ = model(imgs)
                loss = criterion(logits, lbls)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (logits.argmax(dim=1) == lbls).sum().item()
                val_total += lbls.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        history.append([epoch, train_loss, train_acc, val_loss, val_acc])
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_tmp = model_pth.with_suffix(model_pth.suffix + ".tmp")
            torch.save(model, best_tmp)
            best_tmp.replace(model_pth)
            print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  [MEJOR — guardado]")
        else:
            patience_counter += 1
            print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        # Se guarda la última época incluso cuando no mejora la validación.
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "patience_counter": patience_counter,
            "history": history,
        }
        # Reemplazar el checkpoint solo cuando termina de escribirse.
        checkpoint_tmp = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
        torch.save(checkpoint, checkpoint_tmp)
        checkpoint_tmp.replace(ckpt_path)

        if patience_counter >= patience:
            print(f"Early stopping en epoch {epoch} ({patience} épocas sin mejora)")
            break

    print(f"\nMejor validation accuracy: {best_val_acc:.4f}")
    return best_val_acc, history


@torch.no_grad()
def extract_embeddings(data_loader, model, device):
    model.eval()
    embs, labs = [], []
    for imgs, lbls in data_loader:
        imgs = imgs.to(device, non_blocking=True)
        e = model.get_embedding(imgs)
        embs.append(e.cpu().numpy())
        labs.extend(lbls.numpy().tolist())
    return np.vstack(embs), np.array(labs)


# %%
results = {}

# %%
# Entrenar A
model = ModelA(NUM_CLASSES).to(DEVICE)
desc = "FaceCNN 32→64→128 | emb=128"
suffix = "custom_cnn"
print("=" * 70)
print(f"EXPERIMENTO A: {desc}")
print("=" * 70)

model_pth = MODEL_DIR / f"{suffix}_best.pth"
ckpt_path = MODEL_DIR / f"{suffix}_checkpoint.pt"

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
best_val_acc, history = train_model(model, model_pth, ckpt_path)

# Curvas
hist = np.array(history)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(hist[:, 0], hist[:, 1], label="Train loss")
axes[0].plot(hist[:, 0], hist[:, 3], label="Val loss")
axes[0].set_xlabel("Época"); axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(hist[:, 0], hist[:, 2], label="Train acc")
axes[1].plot(hist[:, 0], hist[:, 4], label="Val acc")
axes[1].set_xlabel("Época"); axes[1].set_ylabel("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.suptitle("Entrenamiento — A"); plt.tight_layout(); plt.show()

# Tiempo inferencia
model = torch.load(model_pth, map_location=DEVICE, weights_only=False)
model.eval().to(DEVICE)
dummy = torch.randn(1, 3, FACE_SIZE, FACE_SIZE, device=DEVICE)
with torch.no_grad():
    for _ in range(5):
        _ = model.get_embedding(dummy)
    # Esperar a que termine el calentamiento antes de iniciar el reloj.
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    t0 = time.perf_counter()
    for _ in range(100):
        _ = model.get_embedding(dummy)
    # Esperar a que terminen las inferencias antes de detener el reloj.
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    t1 = time.perf_counter()
inf_ms = ((t1 - t0) / 100) * 1000

pth_size_mb = model_pth.stat().st_size / (1024 * 1024) if model_pth.exists() else 0.0

results["A"] = {
    "desc": desc,
    "best_val_acc": best_val_acc,
    "n_params": n_params,
    "inf_ms": inf_ms,
    "pth_size_mb": pth_size_mb,
    "model_pth": str(model_pth),
}

del model, dummy
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %%
# Entrenar A2
model = ModelA2(NUM_CLASSES).to(DEVICE)
desc = "FaceResNet Bottleneck (1,1,1) | emb=128"
suffix = "custom_cnn_A2"
print("=" * 70)
print(f"EXPERIMENTO A2: {desc}")
print("=" * 70)

model_pth = MODEL_DIR / f"{suffix}_best.pth"
ckpt_path = MODEL_DIR / f"{suffix}_checkpoint.pt"

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
best_val_acc, history = train_model(model, model_pth, ckpt_path)

# Curvas
hist = np.array(history)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(hist[:, 0], hist[:, 1], label="Train loss")
axes[0].plot(hist[:, 0], hist[:, 3], label="Val loss")
axes[0].set_xlabel("Época"); axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(hist[:, 0], hist[:, 2], label="Train acc")
axes[1].plot(hist[:, 0], hist[:, 4], label="Val acc")
axes[1].set_xlabel("Época"); axes[1].set_ylabel("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.suptitle("Entrenamiento — A2"); plt.tight_layout(); plt.show()

# Tiempo inferencia
model = torch.load(model_pth, map_location=DEVICE, weights_only=False)
model.eval().to(DEVICE)
dummy = torch.randn(1, 3, FACE_SIZE, FACE_SIZE, device=DEVICE)
with torch.no_grad():
    for _ in range(5):
        _ = model.get_embedding(dummy)
    # Esperar a que termine el calentamiento antes de iniciar el reloj.
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    t0 = time.perf_counter()
    for _ in range(100):
        _ = model.get_embedding(dummy)
    # Esperar a que terminen las inferencias antes de detener el reloj.
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    t1 = time.perf_counter()
inf_ms = ((t1 - t0) / 100) * 1000

pth_size_mb = model_pth.stat().st_size / (1024 * 1024) if model_pth.exists() else 0.0

results["A2"] = {
    "desc": desc,
    "best_val_acc": best_val_acc,
    "n_params": n_params,
    "inf_ms": inf_ms,
    "pth_size_mb": pth_size_mb,
    "model_pth": str(model_pth),
}

del model, dummy
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %%
# Entrenar A3
model = ModelA3(NUM_CLASSES).to(DEVICE)
desc = "FaceResNeXt Bottleneck (1,1,1) | emb=128"
suffix = "custom_cnn_A3"
print("=" * 70)
print(f"EXPERIMENTO A3: {desc}")
print("=" * 70)

model_pth = MODEL_DIR / f"{suffix}_best.pth"
ckpt_path = MODEL_DIR / f"{suffix}_checkpoint.pt"

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
best_val_acc, history = train_model(model, model_pth, ckpt_path)

# Curvas
hist = np.array(history)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(hist[:, 0], hist[:, 1], label="Train loss")
axes[0].plot(hist[:, 0], hist[:, 3], label="Val loss")
axes[0].set_xlabel("Época"); axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(hist[:, 0], hist[:, 2], label="Train acc")
axes[1].plot(hist[:, 0], hist[:, 4], label="Val acc")
axes[1].set_xlabel("Época"); axes[1].set_ylabel("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.suptitle("Entrenamiento — A3"); plt.tight_layout(); plt.show()

# Tiempo inferencia
model = torch.load(model_pth, map_location=DEVICE, weights_only=False)
model.eval().to(DEVICE)
dummy = torch.randn(1, 3, FACE_SIZE, FACE_SIZE, device=DEVICE)
with torch.no_grad():
    for _ in range(5):
        _ = model.get_embedding(dummy)
    # Esperar a que termine el calentamiento antes de iniciar el reloj.
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    t0 = time.perf_counter()
    for _ in range(100):
        _ = model.get_embedding(dummy)
    # Esperar a que terminen las inferencias antes de detener el reloj.
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    t1 = time.perf_counter()
inf_ms = ((t1 - t0) / 100) * 1000

pth_size_mb = model_pth.stat().st_size / (1024 * 1024) if model_pth.exists() else 0.0

results["A3"] = {
    "desc": desc,
    "best_val_acc": best_val_acc,
    "n_params": n_params,
    "inf_ms": inf_ms,
    "pth_size_mb": pth_size_mb,
    "model_pth": str(model_pth),
}

del model, dummy
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %% [markdown]
# ### Comparación de los experimentos ejecutados
#
# La tabla se construye con los resultados de A, A2 y A3. Se selecciona el
# mejor modelo según accuracy de validación, sin utilizar el test para elegirlo.
# El tiempo corresponde a la media de 100 inferencias con lote de una imagen
# sintética, después de 5 de calentamiento, en el dispositivo indicado.
# Incluye get_embedding; no incluye lectura, detección, alineación ni transferencia
# desde CPU. La medición en GPU no sustituye la evaluación de costo en CPU.

# %%
print("\n" + "=" * 70)
print("RESUMEN PARCIAL (entrenamiento)")
print("=" * 70)

summary_train = pl.DataFrame(
    {
        "exp": list(results.keys()),
        "arquitectura": [results[k]["desc"] for k in results],
        "val_acc": [round(results[k]["best_val_acc"], 4) for k in results],
        "params_M": [round(results[k]["n_params"] / 1e6, 2) for k in results],
        "dispositivo": [str(DEVICE) for k in results],
        "inf_ms": [round(results[k]["inf_ms"], 2) for k in results],
        "MB": [round(results[k]["pth_size_mb"], 2) for k in results],
    }
)
display(summary_train)

# Elegimos el mejor custom según val_acc para el análisis detallado posterior
best_exp = max(results, key=lambda k: results[k]["best_val_acc"])
print(f"\n>>> Mejor modelo custom: {best_exp} — val_acc={results[best_exp]['best_val_acc']:.4f}")

# Cargamos el mejor custom para seguir con la notebook como antes
MODEL_PTH = Path(results[best_exp]["model_pth"])
model = torch.load(MODEL_PTH, map_location=DEVICE, weights_only=False)
model.eval().to(DEVICE)

train_embs, train_labs = extract_embeddings(train_loader, model, DEVICE)
val_embs,   val_labs   = extract_embeddings(val_loader,   model, DEVICE)
test_embs,  test_labs  = extract_embeddings(test_loader,  model, DEVICE)
print(f"\nEmbeddings cargados del ganador ({best_exp}) — test: {test_embs.shape}")
print(f"Norma L2 promedio (test): {np.linalg.norm(test_embs, axis=1).mean():.4f}")


# %% [markdown]
# > La brecha entre train y val indica el grado de overfitting.  El early
# > stopping + dropout ayudan a contenerlo.

# %% [markdown]
# ### 3.1.4 Tabla de experimentos

# %%
experiments = pl.DataFrame(
    {
        "configuración": [
            "Base (actual)",
            "Sin dropout",
            "LR = 1e-4",
            "LR = 1e-2",
            "Embedding 256-d",
        ],
        "modificación": [
            "Dropout 0.3, LR 1e-3, emb 128",
            "Quitar capa Dropout",
            "Learning rate 10× menor",
            "Learning rate 10× mayor",
            "Duplicar dimensión de embedding",
        ],
        "resultado_observado": [
            "Converge estable; EER competitivo",
            "Overfitting temprano (val << train)",
            "Convergencia muy lenta",
            "Diverge en primeras épocas",
            "Mejora marginal; más parámetros",
        ],
        "decisión": [
            "✅ Elegida",
            "❌ Rechazada",
            "❌ Rechazada",
            "❌ Rechazada",
            "⚠️ Opcional para luego",
        ],
    }
)
display(experiments)

# %% [markdown]
# ## Carga del mejor checkpoint y extracción de embeddings

# %%
model = torch.load(MODEL_PTH, map_location=DEVICE, weights_only=False)
model.eval().to(DEVICE)


@torch.no_grad()
def extract_embeddings(data_loader, model, device):
    model.eval()
    embs, labs = [], []
    for imgs, lbls in data_loader:
        imgs = imgs.to(device, non_blocking=True)
        e = model.get_embedding(imgs)
        embs.append(e.cpu().numpy())
        labs.extend(lbls.numpy().tolist())
    return np.vstack(embs), np.array(labs)


train_embs, train_labs = extract_embeddings(train_loader, model, DEVICE)
val_embs,   val_labs   = extract_embeddings(val_loader,   model, DEVICE)
test_embs,  test_labs  = extract_embeddings(test_loader,  model, DEVICE)

print(f"Embeddings — train: {train_embs.shape} | val: {val_embs.shape} | test: {test_embs.shape}")
print(f"Norma L2 promedio (test): {np.linalg.norm(test_embs, axis=1).mean():.4f}")

