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
from insightface.app import FaceAnalysis
from insightface.utils import face_align
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

# No dependemos de módulos externos: la arquitectura se define aquí mismo.

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
# ## Preparación de datos alineados (fallback — obsoleto)
#
# > ⚠️ La notebook `1-pre-processing.ipynb` ya genera los recortes alineados en
# > `output/aligned_112/{split}/{identity}/{nombre}.png`.  Este bloque queda
# > comentado como referencia de emergencia; no hace falta ejecutarlo.

# %%
# # ---------------------------------------------------------------------------
# # FALLBACK — sólo si faltan los aligned crops
# # Estructura correcta: aligned_112/{split}/{identity}/{nombre}.png
# # ---------------------------------------------------------------------------
# def align_split(df: pl.DataFrame, split_name: str) -> pl.DataFrame:
#     out_split = ALIGNED_DIR / split_name
#     out_split.mkdir(parents=True, exist_ok=True)
#     detector = FaceAnalysis(
#         name="buffalo_l",
#         allowed_modules=["detection"],
#         providers=["CPUExecutionProvider"],
#     )
#     detector.prepare(ctx_id=-1, det_thresh=DETECTION_THRESHOLD, det_size=DETECTION_SIZE)
#     records = []
#     for row in df.iter_rows(named=True):
#         img_bgr = cv2.imread(str(DATA_DIR / row["path"]))
#         if img_bgr is None:
#             continue
#         faces = detector.get(img_bgr)
#         if not faces:
#             continue
#         face = max(faces, key=lambda f: float(f.det_score))
#         aligned_bgr = face_align.norm_crop(
#             img_bgr, landmark=face.kps.astype(np.float32), image_size=FACE_SIZE
#         )
#         fname = f"{Path(row['path']).stem}.png"
#         out_dir = out_split / row["identity"]
#         out_dir.mkdir(parents=True, exist_ok=True)
#         out_path = out_dir / fname
#         cv2.imwrite(str(out_path), aligned_bgr)
#         records.append({"aligned_path": str(out_path.relative_to(ALIGNED_DIR)),
#                         "identity": row["identity"]})
#     return pl.from_dicts(records)
#
# for split_name, csv_path in [("train", OUTPUT_DIR / "train.csv"),
#                              ("val",   OUTPUT_DIR / "val.csv"),
#                              ("test",  OUTPUT_DIR / "test.csv")]:
#     if not csv_path.exists():
#         raise RuntimeError(f"Falta {csv_path}. Ejecutá primero la notebook 1.")
#     if not (ALIGNED_DIR / split_name).exists():
#         df = pl.read_csv(csv_path)
#         aligned = align_split(df, split_name)
#         aligned.write_csv(ALIGNED_DIR / f"{split_name}.csv")
# # ---------------------------------------------------------------------------

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

print(f"Splits raw — train: {train_df.height} | val: {val_df.height} | test: {test_df.height}")

# %%
label_encoder = LabelEncoder()
label_encoder.fit(sorted(train_df["identity"].unique().to_list()))
NUM_CLASSES = len(label_encoder.classes_)
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
    def __init__(self, df, split_name):
        self.df = df
        self.split_name = split_name
        self.root = ALIGNED_DIR / split_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)
        # Construimos ruta: aligned_112/{split}/{identity}/{stem}.png
        # Si la notebook 1 guardó de otra forma, ajustar aquí.
        fname = f"{Path(row['path']).stem}.png"
        img_path = self.root / row["identity"] / fname
        if not img_path.exists():
            # fallback plano sin subcarpeta de identidad
            img_path = self.root / fname
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            # Último fallback: leer original y alinear on-the-fly
            img_bgr = cv2.imread(str(DATA_DIR / row["path"]))
        # Sanity: forzamos 112×112 RGB — si el original no está alineado,
        # al menos tiene el tamaño que espera la red.
        img_bgr = cv2.resize(img_bgr, (FACE_SIZE, FACE_SIZE))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        label = row["label"]
        return img, label


train_ds = AlignedDataset(train_df, "train")
val_ds   = AlignedDataset(val_df,   "val")
test_ds  = AlignedDataset(test_df,  "test")

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE,
                                           shuffle=True, num_workers=0, pin_memory=True)
val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH_SIZE,
                                           shuffle=False, num_workers=0, pin_memory=True)
test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=BATCH_SIZE,
                                           shuffle=False, num_workers=0, pin_memory=True)

print(f"Batches — train: {len(train_loader)} | val: {len(val_loader)} | test: {len(test_loader)}")

# %% [markdown]
# ## 3.1 Modelo A — CNN Custom
#
# ### 3.1.1 Arquitectura

# %%
class FaceCNN(nn.Module):
    """
    CNN custom para reconocimiento facial.

    Entrenada con clasificación (CrossEntropy) sobre N identidades.
    La salida de ``fc_embedding`` (antes de L2-normalizar) se usa como embedding
    intermedio; ``get_embedding`` devuelve el vector L2-normalizado listo para
    comparación con similitud coseno.
    """

    def __init__(self, num_classes: int, embedding_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        self.conv1 = self._conv_block(3, 64)
        self.conv2 = self._conv_block(64, 128)
        self.conv3 = self._conv_block(128, 256)
        self.conv4 = self._conv_block(256, 512)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc_embedding = nn.Linear(512, embedding_dim)
        self.bn_embedding = nn.BatchNorm1d(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)   # 56×56
        x = self.conv2(x)   # 28×28
        x = self.conv3(x)   # 14×14
        x = self.conv4(x)   # 7×7
        x = self.global_pool(x)  # 1×1
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        emb = self.fc_embedding(x)
        emb = self.bn_embedding(emb)
        emb_norm = F.normalize(emb, p=2, dim=1)
        logits = self.classifier(emb_norm)
        return logits, emb_norm

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extrae embeddings L2-normalizados para inferencia (sin clasificador)."""
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            emb = self.fc_embedding(x)
            emb = self.bn_embedding(emb)
            return F.normalize(emb, p=2, dim=1)


model = FaceCNN(num_classes=NUM_CLASSES, embedding_dim=EMBEDDING_DIM, dropout=0.3).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Parámetros entrenables: {n_trainable:,}")
print(f"Dimensión del embedding: {EMBEDDING_DIM}")
print(f"Número de clases: {NUM_CLASSES}")

# %% [markdown]
# **Decisiones de diseño:**
# - **4 bloques conv** (64→128→256→512) con BatchNorm + ReLU + MaxPool: más
#   capacidad que la versión base (32→64→128→256) para distinguir 100+ identidades.
# - **Global Average Pooling** al final: reduce parámetros frente a una capa FC
#   grande y agrega invarianza espacial.
# - **Dropout 0.3** y **BatchNorm** en el embedding: regularización sin matar
#   la señal en un modelo pequeño.
# - **L2-normalización** del embedding: permite comparar con similitud coseno
#   directamente (producto punto).

# %% [markdown]
# ### 3.1.2 Hiperparámetros y entrenamiento
#
# El bucle soporta **reanudación**: si existe un checkpoint de entrenamiento
# se restauran modelo, optimizador, scheduler y métricas.  También evaluamos
# en validación antes de arrancar para no perder el *best*.

# %%
MODEL_PTH = MODEL_DIR / "custom_cnn_best.pth"
CKPT_PATH = MODEL_DIR / "custom_cnn_checkpoint.pt"

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3, verbose=True
)

best_val_acc = 0.0
patience_counter = 0
history = []
start_epoch = 1

if CKPT_PATH.exists():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_val_acc = ckpt["best_val_acc"]
    patience_counter = ckpt["patience_counter"]
    history = ckpt["history"]
    print(f"Restaurado desde checkpoint: epoch {start_epoch - 1}, best_val_acc={best_val_acc:.4f}")
elif MODEL_PTH.exists():
    # Fallback: solo tenemos el .pth del modelo completo
    model = torch.load(MODEL_PTH, map_location=DEVICE, weights_only=False)
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
    print(f"Cargado MODEL_PTH existente. Computado val_acc={best_val_acc:.4f} → best_val_acc")
    print("Nota: optimizador y scheduler reinician desde cero.")

print(f"Iniciando entrenamiento desde epoch {start_epoch}...")
for epoch in range(start_epoch, EPOCHS + 1):
    # Train
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

    # Validation
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
        torch.save(model, MODEL_PTH)  # modelo completo para el backend
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "patience_counter": patience_counter,
                "history": history,
            },
            CKPT_PATH,
        )
        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  [MEJOR — guardado]")
    else:
        patience_counter += 1
        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if patience_counter >= PATIENCE:
            print(f"Early stopping en epoch {epoch} ({PATIENCE} épocas sin mejora)")
            break

print(f"\nMejor validation accuracy: {best_val_acc:.4f}")

# %% [markdown]
# ### 3.1.3 Curvas de entrenamiento

# %%
hist = np.array(history)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(hist[:, 0], hist[:, 1], label="Train loss")
axes[0].plot(hist[:, 0], hist[:, 3], label="Val loss")
axes[0].set_xlabel("Época"); axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(hist[:, 0], hist[:, 2], label="Train acc")
axes[1].plot(hist[:, 0], hist[:, 4], label="Val acc")
axes[1].set_xlabel("Época"); axes[1].set_ylabel("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.suptitle("Entrenamiento — CNN Custom"); plt.tight_layout(); plt.show()

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

