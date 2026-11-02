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
#     display_name: .venv (Python 3.12)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Entrenamiento y comparación de modelos
#
# ## Equipo
# - Gabriel Soda
# - Alejandro Armas
#
# Primera etapa:
# Cargar los recortes exportados, preparar sus etiquetas y visualizar un lote

# %% [markdown]
# ## 1. Lectura de las listas de imágenes
#
# Los CSV son los del procesamiento de la notebook 1, indican qué imagen pertenece a cada conjunto y quién aparece.
#
# Cada `path` es relativo a `output/aligned_112/`. Train ya incluye una variante aumentada por imagen original

# %%
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

OUTPUT_DIR = Path("output")
IMAGE_DIR = OUTPUT_DIR / "aligned_112"

train_df = pd.read_csv(OUTPUT_DIR / "train.csv")
val_df = pd.read_csv(OUTPUT_DIR / "val.csv")
test_df = pd.read_csv(OUTPUT_DIR / "test.csv")

for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    assert not df[["path", "identity"]].isna().any().any()
    assert df["path"].is_unique
    print(f"{name}: {len(df)} imágenes, {df['identity'].nunique()} identidades")

print(train_df.head().to_string(index=False))

# %% [markdown]
# ## 2. Asignación de un número a cada identidad
#
# La CNN recibirá etiquetas enteras, de 0 a 99. Construimos el mapa una sola vez con las identidades de train, ordenadas alfabéticamente, y lo reutilizamos en los tres conjuntos. Así, un número siempre representa a la misma persona.
#
# Estos splits son **closed-set**: contienen las mismas personas con fotos distintas. Antes de entrenar debemos definir también la evaluación con identidades no vistas y la calibración del umbral fuera del test.

# %%
class_names = sorted(train_df["identity"].unique())
class_to_idx = {name: idx for idx, name in enumerate(class_names)}

assert set(val_df["identity"]) == set(class_names)
assert set(test_df["identity"]) == set(class_names)

for df in [train_df, val_df, test_df]:
    df["label"] = df["identity"].map(class_to_idx)

print(f"Cantidad de clases: {len(class_names)}")
print(train_df[["identity", "label"]].drop_duplicates().head().to_string(index=False))


# %% [markdown]
# ## 3. Carga de imágenes desde dataset
#
# OpenCV lee **BGR**; convertimos a **RGB** para mantener los colores correctos. Luego cambiamos de `(alto, ancho, canales)` a `(canales, alto, ancho)`, el orden usado por la CNN en PyTorch. Convertimos a `float32` y dividimos por 255 para pasar de `[0, 255]` a `[0, 1]`.
#
# Los rostros ya están alineados a 112 × 112. Esta escala es nuestra preparación inicial para la CNN.


# %%
class FaceDataset(Dataset):
    def __init__(self, dataframe, image_dir):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = self.image_dir / row["path"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"No se pudo leer: {image_path}")
        if image.shape != (112, 112, 3):
            raise ValueError(f"Tamaño inesperado en {image_path}: {image.shape}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label = int(row["label"])
        return image, label


train_dataset = FaceDataset(train_df, IMAGE_DIR)
val_dataset = FaceDataset(val_df, IMAGE_DIR)
test_dataset = FaceDataset(test_df, IMAGE_DIR)

image, label = train_dataset[0]
print("Imagen:", image.shape, image.dtype)
print("Rango:", image.min().item(), image.max().item())
print("Etiqueta:", label, class_names[label])

# %% [markdown]
# ## 4. Agrupación de imágenes con DataLoader
#
# `DataLoader` pide muestras al Dataset y las apila en lotes. Usamos 32 imágenes por lote y mezclamos solo train. Validación y test conservan su orden.
#
# `num_workers=0` carga en el proceso de la notebook. La semilla permite reproducir el orden al volver a ejecutar esta celda.

# %%
BATCH_SIZE = 32
SEED = 42

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    generator=torch.Generator().manual_seed(SEED),
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

images, labels = next(iter(train_loader))
print("Imágenes:", images.shape, images.dtype)
print("Etiquetas:", labels.shape, labels.dtype)
assert images.shape == (BATCH_SIZE, 3, 112, 112)
assert labels.dtype == torch.int64
assert 0 <= images.min() <= images.max() <= 1
assert 0 <= labels.min() <= labels.max() < len(class_names)

# %% [markdown]
# ## 5. Visualización del lote y sus etiquetas
#
# Mostramos las primeras 16 imágenes del lote. Para Matplotlib volvemos de `(canales, alto, ancho)` a `(alto, ancho, canales)` y como los valores ya están en `[0, 1]` se pueden mostrar directamente.
#
# El lote puede incluir originales y aumentadas guardadas.

# %%
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].permute(1, 2, 0).numpy())
    ax.set_title(class_names[labels[i].item()], fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# Hasta este punto preparamos los datos. Un lote contiene **32 imágenes × 3 canales × 112 alto × 112 ancho**, junto con 32 etiquetas.
#
