# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook de documentacion, tratamiento datos y entrenamiento
#

# %% [markdown]
# ## Equipo
# - Alumno 1 : **Alejandro Armas**
# - Alumno 2 : **Gabriel Soda**

# %%
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from IPython.display import display
import warnings

import altair as alt
import cv2
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from insightface.app import FaceAnalysis
from insightface.utils import face_align

alt.data_transformers.disable_max_rows()

DATA_DIR = Path("data/images")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FACE_SIZE = 112
MIN_FACE_SIDE = FACE_SIZE // 2
DETECTION_THRESHOLD = 0.5
DETECTION_SIZE = (640, 640)

# %%
# Escaneamos metadatos: no cargamos todos los píxeles ni imágenes completas en memoria.
# Cada subcarpeta representa una identidad en el dataset Pins.
records = []
candidates = sorted(
    path for path in DATA_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
)

for path in candidates:
    identity = path.parent.name.removeprefix("pins_").strip()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with Image.open(path) as image:
                image.load()  # valida que el archivo pueda decodificarse
                width, height = image.size
                image_format, mode = image.format, image.mode
                grayscale = np.asarray(image.convert("L"), dtype=np.float32)
                brightness = grayscale.mean()
                contrast = grayscale.std()

        records.append(
            {
                "path": str(path.relative_to(DATA_DIR)),
                "identity": identity,
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 3),
                "pixels": width * height,
                "format": image_format,
                "mode": mode,
                "brightness": round(float(brightness), 2),
                "contrast": round(float(contrast), 2),
                "readable": True,
                "error": None,
            }
        )
    except (UnidentifiedImageError, OSError, ValueError, Warning) as error:
        records.append(
            {
                "path": str(path.relative_to(DATA_DIR)),
                "identity": identity,
                "width": None, "height": None, "aspect_ratio": None, "pixels": None,
                "format": None, "mode": None, "brightness": None, "contrast": None,
                "readable": False, "error": type(error).__name__,
            }
        )

images = pl.from_dicts(records)
images.head()

# %% [markdown]
# ## EDA — Análisis exploratorio del dataset
#
# Usamos `polars` para agregar metadatos sin cargar todos los píxeles, y `altair` para visualizar. El objetivo es entender el **desbalance de clases**, la **distribución de resoluciones**, la **calidad fotométrica** y la **variabilidad visual** (pose, expresión, oclusiones) **antes** de aplicar ningún filtro de calidad.

# %%
# Resumen general y formatos (sobre TODAS las imágenes legibles)
readable = images.filter(pl.col("readable"))
unreadable_count = images.filter(~pl.col("readable")).height
identity_counts = (
    readable.group_by("identity")
    .len()
    .rename({"len": "count"})
    .sort("count", descending=True)
)

summary = pl.DataFrame({
    "métrica": [
        "Total imágenes", "Imágenes no legibles", "Identidades",
        "Imgs/identidad (media)", "Imgs/identidad (mediana)", "Mín / Máx por identidad",
    ],
    "valor": [
        str(readable.height),
        str(unreadable_count),
        str(identity_counts.height),
        f"{identity_counts['count'].mean():.1f}",
        str(identity_counts["count"].median()),
        f"{identity_counts['count'].min()} / {identity_counts['count'].max()}",
    ],
})
display(summary)

format_counts = readable.group_by(["format", "mode"]).len().sort("len", descending=True)
display(format_counts)

# %%
# Distribución de imágenes por identidad — desbalance de clases
class_stats = identity_counts.select(
    pl.len().alias("identidades"),
    pl.col("count").min().alias("min"),
    pl.col("count").quantile(0.25).alias("q1"),
    pl.col("count").mean().round(2).alias("media"),
    pl.col("count").median().alias("mediana"),
    pl.col("count").quantile(0.75).alias("q3"),
    pl.col("count").max().alias("max"),
    pl.col("count").std().round(2).alias("std"),
    (pl.col("count").std() / pl.col("count").mean()).round(3).alias("cv"),
)
display(class_stats)

# Histograma + KDE (Altair)
BIN = 20
N = identity_counts.height
data = alt.InlineData(values=identity_counts.to_dicts())

hist = alt.Chart(data).mark_bar(opacity=0.6, color="#4c78a8").encode(
    x=alt.X("count:Q", bin=alt.Bin(step=BIN), title="Imágenes por identidad"),
    y=alt.Y("count():Q", title="Identidades"),
)

kde = alt.Chart(data).transform_density(
    "count",
    as_=["count", "density"],
    extent=[identity_counts["count"].min(), identity_counts["count"].max()],
    steps=100,
).transform_calculate(scaled_density=f"datum.density * {N * BIN}").mark_line(
    color="#e45756", strokeWidth=2
).encode(x="count:Q", y="scaled_density:Q")

display((hist + kde).properties(width=600, height=300, title="Desbalance de clases"))

# Top 10 y Bottom 10
top10 = identity_counts.head(10)
bottom10 = identity_counts.tail(10).sort("count")

top_chart = alt.Chart(alt.InlineData(values=top10.to_dicts())).mark_bar(color="#4c78a8").encode(
    x=alt.X("count:Q", title="Imágenes"),
    y=alt.Y("identity:N", sort="-x", title=None),
).properties(title="Top 10 identidades", width=300, height=250)

bottom_chart = alt.Chart(alt.InlineData(values=bottom10.to_dicts())).mark_bar(color="#e45756").encode(
    x=alt.X("count:Q", title="Imágenes"),
    y=alt.Y("identity:N", sort="x", title=None),
).properties(title="Bottom 10 identidades", width=300, height=250)

display(top_chart | bottom_chart)

# %%
# Resoluciones y relación de aspecto (EDA — sin filtrar)
res_stats = readable.select(
    pl.col("width").min().alias("w_min"),
    pl.col("width").median().alias("w_med"),
    pl.col("width").max().alias("w_max"),
    pl.col("height").min().alias("h_min"),
    pl.col("height").median().alias("h_med"),
    pl.col("height").max().alias("h_max"),
    pl.col("aspect_ratio").mean().round(3).alias("aspect_ratio_mean"),
)
display(res_stats)

# Heatmap de resoluciones (bins de 32 px)
res_bins = (
    readable.with_columns(
        (pl.col("width") // 32 * 32).alias("w_bin"),
        (pl.col("height") // 32 * 32).alias("h_bin"),
    )
    .group_by(["w_bin", "h_bin"])
    .len()
    .rename({"len": "n"})
)

res_chart = alt.Chart(alt.InlineData(values=res_bins.to_dicts())).mark_rect().encode(
    x=alt.X("w_bin:O", title="Ancho (px)"),
    y=alt.Y("h_bin:O", title="Alto (px)"),
    color=alt.Color("n:Q", title="Imágenes", scale=alt.Scale(scheme="blues")),
    tooltip=[
        alt.Tooltip("w_bin:Q", title="Ancho (px)"),
        alt.Tooltip("h_bin:Q", title="Alto (px)"),
        alt.Tooltip("n:Q", title="Imágenes"),
    ],
).properties(width=400, height=400, title="Distribución de resoluciones (bins 32 px)")
display(res_chart)

# %%
# Muestra visual por identidad (pose, expresión, oclusiones)
N_IDENTITIES = 5
N_SAMPLES = 5

sample_idents = identity_counts.head(30).sample(n=N_IDENTITIES, seed=42)["identity"].to_list()
fig, axes = plt.subplots(N_IDENTITIES, N_SAMPLES, figsize=(10, 2 * N_IDENTITIES))

for i, ident in enumerate(sample_idents):
    n_available = readable.filter(pl.col("identity") == ident).height
    paths = (
        readable.filter(pl.col("identity") == ident)
        .sample(n=min(N_SAMPLES, n_available), seed=42)
        ["path"]
        .to_list()
    )
    for j in range(N_SAMPLES):
        ax = axes[i, j]
        if j < len(paths):
            with Image.open(DATA_DIR / paths[j]) as img:
                ax.imshow(img.convert("RGB"))
        ax.axis("off")
        if j == 0:
            ax.set_ylabel(ident, fontsize=9, rotation=0, ha="right", va="center")

fig.suptitle("Muestra visual por identidad — pose, expresión, oclusiones", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Variaciones fotométricas
#
# Analizamos brillo y contraste como proxy de condiciones de iluminación. Los extremos se inspeccionan visualmente para detectar posibles problemas de calidad.

# %%
# Estadísticas y distribución de brillo / contraste
photo_stats = readable.select(
    pl.col("brightness").min().alias("brillo_min"),
    pl.col("brightness").median().alias("brillo_med"),
    pl.col("brightness").max().alias("brillo_max"),
    pl.col("contrast").min().alias("contraste_min"),
    pl.col("contrast").median().alias("contraste_med"),
    pl.col("contrast").max().alias("contraste_max"),
)
display(photo_stats)

photo_long = readable.unpivot(
    index=["path", "identity"],
    on=["brightness", "contrast"],
    variable_name="metric",
    value_name="value",
)

photo_chart = alt.Chart(alt.InlineData(values=photo_long.to_dicts())).mark_bar(opacity=0.7).encode(
    x=alt.X("value:Q", bin=alt.Bin(maxbins=30), title="Valor"),
    y=alt.Y("count():Q", title="Imágenes"),
    color=alt.Color("metric:N", title=None),
    column=alt.Column("metric:N", title=None),
).properties(width=280, height=240, title="Distribución de brillo y contraste")
display(photo_chart)

# %%
# Casos extremos para inspección visual
extremes = pl.concat([
    readable.sort("brightness").head(3).with_columns(pl.lit("brillo bajo").alias("case")),
    readable.sort("brightness", descending=True).head(3).with_columns(pl.lit("brillo alto").alias("case")),
    readable.sort("contrast").head(3).with_columns(pl.lit("contraste bajo").alias("case")),
    readable.sort("contrast", descending=True).head(3).with_columns(pl.lit("contraste alto").alias("case")),
])

fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for ax, row in zip(axes.flat, extremes.iter_rows(named=True)):
    with Image.open(DATA_DIR / row["path"]) as img:
        ax.imshow(img.convert("RGB"))
    ax.set_title(
        f"{row['case']}\n{row['identity']}\nb={row['brightness']}, c={row['contrast']}",
        fontsize=8,
    )
    ax.axis("off")

for ax in axes.flat[len(extremes):]:
    ax.axis("off")

fig.suptitle("Casos extremos de iluminación y contraste", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Control de calidad y limpieza
#
# Aplicamos reglas de filtrado **en orden**, reportando cuántas imágenes descarta cada una. El orden prioriza eficiencia: descartamos identidades con muy pocas muestras **antes** de correr detección facial.
#
# Reglas:
# 1. Identidades con $< 100$ imágenes (descartamos la identidad completa).
# 2. Imagen con algún lado $< 56$ px (no puede contener una cara de $56 \times 56$).
# 3. Detector: 0 rostros detectados.
# 4. Detector: más de 1 rostro detectado.
# 5. Bbox del rostro con lado $< 56$ px.

# %%
# Control de calidad: pipeline de filtrado con conteo por regla
MIN_IMAGES_PER_IDENTITY = 100

# --- Regla 1: identidades con pocas muestras ---
identity_counts_all = (
    readable.group_by("identity")
    .len()
    .rename({"len": "count"})
    .sort("count", descending=True)
)

eligible_identities = identity_counts_all.filter(pl.col("count") >= MIN_IMAGES_PER_IDENTITY)
low_count_identities = identity_counts_all.filter(pl.col("count") < MIN_IMAGES_PER_IDENTITY)

images_after_rule1 = readable.filter(
    pl.col("identity").is_in(eligible_identities["identity"].implode())
)

# --- Regla 2: resolución mínima de imagen ---
images_after_rule2 = images_after_rule1.filter(
    (pl.col("width") >= MIN_FACE_SIDE) & (pl.col("height") >= MIN_FACE_SIDE)
)

# Tabla resumen pre-detección
pre_detection_summary = pl.DataFrame({
    "regla": [
        "0. Imágenes legibles (base)",
        "1. Identidades < 100 imgs",
        "2. Imagen con lado < 56 px",
    ],
    "identidades_descartadas": [
        None,
        low_count_identities.height,
        None,
    ],
    "imagenes_descartadas": [
        0,
        readable.height - images_after_rule1.height,
        images_after_rule1.height - images_after_rule2.height,
    ],
    "imagenes_restantes": [
        readable.height,
        images_after_rule1.height,
        images_after_rule2.height,
    ],
})

print("=== Resumen pre-detección ===")
display(pre_detection_summary)
print(f"Identidades descartadas por regla 1 ({low_count_identities.height}):")
display(low_count_identities.sort("count"))

# %%
# Detección facial con SCRFD (buffalo_l) sobre imágenes que pasaron reglas 1+2
detector = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection"],
    providers=["CPUExecutionProvider"],
)
detector.prepare(ctx_id=-1, det_thresh=DETECTION_THRESHOLD, det_size=DETECTION_SIZE)

detection_records = []
for index, row in enumerate(images_after_rule2.iter_rows(named=True), start=1):
    image_bgr = cv2.imread(str(DATA_DIR / row["path"]))
    faces = detector.get(image_bgr) if image_bgr is not None else []
    face_count = len(faces)

    if face_count == 1:
        face = faces[0]
        x1, y1, x2, y2 = face.bbox.astype(int).tolist()
        bbox_width, bbox_height = x2 - x1, y2 - y1
        min_bbox_side = min(bbox_width, bbox_height)
        keypoints = face.kps.astype(np.float32).tolist()
        requires_upscale = min_bbox_side < FACE_SIZE
        quality_rule = (
            "5. Bbox < 56 px"
            if min_bbox_side < MIN_FACE_SIDE
            else "Aceptada: requiere escalado"
            if requires_upscale
            else "Aceptada"
        )
    elif face_count == 0:
        x1 = y1 = x2 = y2 = bbox_width = bbox_height = min_bbox_side = None
        keypoints = None
        requires_upscale = None
        quality_rule = "3. Sin rostro detectado"
    else:
        x1 = y1 = x2 = y2 = bbox_width = bbox_height = min_bbox_side = None
        keypoints = None
        requires_upscale = None
        quality_rule = "4. Más de un rostro"

    detection_records.append({
        "path": row["path"],
        "identity": row["identity"],
        "face_count": face_count,
        "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
        "bbox_width": bbox_width, "bbox_height": bbox_height,
        "min_bbox_side": min_bbox_side,
        "keypoints": keypoints,
        "requires_upscale_after_detection": requires_upscale,
        "quality_rule": quality_rule,
    })

    if index % 500 == 0 or index == images_after_rule2.height:
        print(f"Procesadas {index:,}/{images_after_rule2.height:,} imágenes")

detections = pl.from_dicts(detection_records)

# Conteo post-detección
post_detection_counts = (
    detections.group_by("quality_rule")
    .len()
    .rename({"len": "imagenes_descartadas"})
    .with_columns(pl.col("imagenes_descartadas").cast(pl.Int64))
    .sort("quality_rule")
)

accepted = detections.filter(pl.col("quality_rule").str.starts_with("Aceptada"))
rejected_rules = (
    post_detection_counts
    .filter(~pl.col("quality_rule").str.starts_with("Aceptada"))
    .with_columns(
        pl.lit(None).cast(pl.Int64).alias("identidades_descartadas"),
        (
            pl.lit(images_after_rule2.height, dtype=pl.Int64)
            - pl.col("imagenes_descartadas").cum_sum()
        ).alias("imagenes_restantes"),
    )
    .rename({"quality_rule": "regla"})
    .select([
        "regla",
        "identidades_descartadas",
        "imagenes_descartadas",
        "imagenes_restantes",
    ])
)

# Pipeline completo: conteo por regla
pipeline_summary = pl.concat([
    pre_detection_summary,
    rejected_rules,
])

# Total final aceptado
final_accepted = accepted.height
final_identities = accepted["identity"].n_unique()

pipeline_summary = pl.concat([
    pipeline_summary,
    pl.DataFrame({
        "regla": ["Aceptadas (todas las reglas)"],
        "identidades_descartadas": [None],
        "imagenes_descartadas": [0],
        "imagenes_restantes": [final_accepted],
    }),
])

print("\n=== Pipeline completo de calidad ===")
display(pipeline_summary)

print(f"\nAceptadas: {final_accepted} imágenes de {final_identities} identidades")
print(f"Requieren escalado: {accepted.filter(pl.col('requires_upscale_after_detection')).height}")

# %%
# Ejemplos descartados por cada regla (inspección visual de umbrales)
rejected_examples = (
    detections.filter(~pl.col("quality_rule").str.starts_with("Aceptada"))
    .group_by("quality_rule")
    .agg(pl.col("path").head(3).alias("paths"))
)

if rejected_examples.is_empty():
    print("No hay imágenes descartadas por las reglas de detección.")
else:
    for row in rejected_examples.iter_rows(named=True):
        rule = row["quality_rule"]
        paths = row["paths"]
        n = len(paths)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), squeeze=False)
        for ax, p in zip(axes.flat, paths):
            with Image.open(DATA_DIR / p) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_title(f"{rule}\n{p[:40]}...", fontsize=8)
            ax.axis("off")
        fig.suptitle(f"Ejemplos: {rule}", fontsize=10)
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## Detección, landmarks y alineación con SCRFD
#
# A continuación visualizamos el pipeline `detectar → validar → alinear` sobre las imágenes que pasaron todas las reglas de calidad.

# %%
# Antes/después: usamos los 5 landmarks almacenados por SCRFD para alinear a FACE_SIZE.
alignment_examples = accepted.sample(n=min(4, accepted.height), seed=42)
fig, axes = plt.subplots(alignment_examples.height, 2, figsize=(8, 3 * alignment_examples.height))
axes = np.atleast_2d(axes)

for axis_pair, row in zip(axes, alignment_examples.iter_rows(named=True)):
    image_bgr = cv2.imread(str(DATA_DIR / row["path"]))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    keypoints = np.asarray(row["keypoints"], dtype=np.float32)

    preview = image_rgb.copy()
    cv2.rectangle(
        preview,
        (row["bbox_x1"], row["bbox_y1"]),
        (row["bbox_x2"], row["bbox_y2"]),
        (0, 255, 0),
        2,
    )
    for x, y in keypoints.astype(int):
        cv2.circle(preview, (x, y), 3, (255, 0, 0), -1)

    aligned_bgr = face_align.norm_crop(
        image_bgr, landmark=keypoints, image_size=FACE_SIZE
    )
    axis_pair[0].imshow(preview)
    axis_pair[0].set_title(f"Antes: {row['identity']}")
    axis_pair[1].imshow(cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB))
    axis_pair[1].set_title(f"Alineada: {FACE_SIZE} × {FACE_SIZE}")
    for axis in axis_pair:
        axis.axis("off")

fig.suptitle("SCRFD: bbox y 5 landmarks → alineación", fontsize=14)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Subset balanceado
#
# Partimos de las imágenes que pasaron todas las reglas de calidad. Para entrenar la CNN custom necesitamos clases balanceadas: muestreamos **exactamente 100 imágenes por identidad** (`MIN = MAX`). El muestreo aleatorio simple preserva proporcionalmente las condiciones difíciles porque nunca filtramos por calidad fotométrica.

# %%
MIN_PER_ID = 100
MAX_PER_ID = 100

accepted_counts = (
    accepted.group_by("identity")
    .len()
    .rename({"len": "count"})
    .sort("count", descending=True)
)
kept = accepted_counts.filter(pl.col("count") >= MIN_PER_ID)["identity"].to_list()

if not kept:
    display(accepted_counts)
    raise ValueError(
        f"Ninguna identidad tiene al menos {MIN_PER_ID} imágenes aceptadas. "
        "Reducí MIN_PER_ID o revisá las reglas de calidad."
    )

balanced = (
    accepted.filter(pl.col("identity").is_in(kept))
    .group_by("identity", maintain_order=True)
    .agg(pl.all().sample(n=MAX_PER_ID, seed=42))
    .explode(pl.all().exclude("identity"))
    .sort(["identity", "path"])
)

print(
    f"Subset balanceado: {balanced['identity'].n_unique()} identidades "
    f"× {MAX_PER_ID} = {balanced.height} imágenes"
)

# %% [markdown]
# ## Clustering exploratorio
#
# Extraemos embeddings con **ArcFace** (modelo preentrenado de `buffalo_l`) sobre las imágenes que pasaron el control de calidad. Proyectamos a 2D con **t-SNE** y comparamos clusters K-Means contra las etiquetas reales (ARI, NMI, silhouette). El objetivo no es decorar, sino detectar identidades mal etiquetadas, outliers o sub-clusters por condiciones de captura.

# %%
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# Cargamos ArcFace (reconocimiento + detección ya está en buffalo_l)
face_analysis = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
)
face_analysis.prepare(ctx_id=-1, det_thresh=DETECTION_THRESHOLD, det_size=DETECTION_SIZE)

# Extraemos embeddings sobre `accepted` (ya tienen exactamente 1 rostro)
embedding_records = []
for row in accepted.iter_rows(named=True):
    image_bgr = cv2.imread(str(DATA_DIR / row["path"]))
    if image_bgr is None:
        continue
    faces = face_analysis.get(image_bgr)
    if len(faces) == 1:
        emb = faces[0].embedding
        embedding_records.append({
            "path": row["path"],
            "identity": row["identity"],
            "embedding": emb,
        })

embeddings_df = pl.from_dicts(embedding_records)
X = np.vstack(embeddings_df["embedding"].to_list())
y = embeddings_df["identity"].to_list()
print(f"Embeddings extraídos: {len(y)} imágenes de {embeddings_df['identity'].n_unique()} identidades")

# t-SNE
n_samples = min(2000, X.shape[0])
tsne_idx = np.random.RandomState(42).choice(X.shape[0], n_samples, replace=False)
X_tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca", learning_rate="auto").fit_transform(X[tsne_idx])

# K-Means con k = n identidades
n_clusters = embeddings_df["identity"].n_unique()
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X)
labels_km = kmeans.labels_

ari = adjusted_rand_score(y, labels_km)
nmi = normalized_mutual_info_score(y, labels_km)
sil = silhouette_score(X, labels_km)

metrics = pl.DataFrame({
    "métrica": ["ARI (K-Means vs identidad)", "NMI (K-Means vs identidad)", "Silhouette (K-Means)"],
    "valor": [round(ari, 3), round(nmi, 3), round(sil, 3)],
})
display(metrics)

# Plot t-SNE con Altair
tsne_data = pl.DataFrame({
    "x": X_tsne[:, 0],
    "y": X_tsne[:, 1],
    "identity": [y[i] for i in tsne_idx],
    "kmeans_label": [int(labels_km[i]) for i in tsne_idx],
})

scatter = alt.Chart(tsne_data.to_dicts()).mark_circle(size=30, opacity=0.6).encode(
    x=alt.X("x:Q", title="t-SNE 1"),
    y=alt.Y("y:Q", title="t-SNE 2"),
    color=alt.Color("identity:N", title="Identidad", scale=alt.Scale(scheme="category20")),
    tooltip=["identity:N"],
).properties(width=550, height=450, title="t-SNE de embeddings ArcFace (coloreado por identidad)")

display(scatter)

# %% [markdown]
# ### Interpretación del clustering
#
# - **ARI y NMI cercanos a 1** indican que los embeddings ya separan bien las identidades.
# - **Silhouette** cuantifica qué tan cohesionados están los clusters.
# - Identidades que se mezclan en el t-SNE suelen ser personas con parecido facial real o condiciones de captura similares (iluminación, pose).
# - Outliers (puntos lejos de su grupo) pueden ser errores de etiquetado o imágenes de muy baja calidad.

# %% [markdown]
# ## Data augmentation
#
# Definimos un pipeline con `albumentations` acorde al dominio facial. Incluimos transformaciones que ocurren en el mundo real y **excluimos** las que destruyen la identidad (flip vertical, rotaciones de 90°).

# %%
import albumentations as A

augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),                         # simétrico en rostros (común en datasets)
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),  # pequeñas variaciones de pose
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),   # iluminación variable
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MotionBlur(blur_limit=3, p=1.0),
    ], p=0.3),                                        # desenfoque leve (cámara, movimiento)
    A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),      # ruido de sensor
    A.ImageCompression(quality_lower=70, quality_upper=95, p=0.2),  # compresión JPEG
    A.CoarseDropout(max_holes=3, max_height=16, max_width=16, min_holes=1, p=0.2),  # oclusiones parciales
])

# Grilla: una imagen + 8 aumentaciones
def show_augmentations(image_path, n=8):
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 3))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")
    for i in range(n):
        aug = augmentation_pipeline(image=img)["image"]
        axes[i + 1].imshow(cv2.cvtColor(aug, cv2.COLOR_BGR2RGB) if aug.shape[-1] == 3 else aug)
        axes[i + 1].set_title(f"Aug {i+1}")
        axes[i + 1].axis("off")
    fig.suptitle("Pipeline de augmentation", fontsize=14)
    plt.tight_layout()
    plt.show()

sample_img = accepted.sample(n=1, seed=42)["path"].to_list()[0]
show_augmentations(DATA_DIR / sample_img)

# %% [markdown]
# ### Justificación de transformaciones
#
# | Transformación | ¿Incluida? | Justificación |
# |---|---|---|
# | Horizontal flip | ✅ | Rostros son aproximadamente simétricos; duplica datos sin destruir identidad. |
# | Rotación ±10° | ✅ | Variación natural de pose (inclinación de cabeza). |
# | Flip vertical | ❌ | No ocurre en la realidad; destruye identidad. |
# | Rotación 90° | ❌ | No ocurre en uso real de la app. |
# | Brillo/contraste | ✅ | Cambios de iluminación entre ambientes. |
# | Blur / ruido | ✅ | Desenfoque de cámara o baja luz. |
# | Coarse dropout | ✅ | Simula oclusiones (gafas, pelo, mano). |
# | JPEG compression | ✅ | Artefactos de compresión comunes en imágenes web. |
#
# La augmentation se aplica **solo en entrenamiento**. Validación y test usan la imagen original alineada.

# %% [markdown]
# ## Partición de datos (splits)
#
# Dividimos en **train / validation / test** estratificados por identidad. Como el objetivo final es **open-set** (verificación con personas no vistas en entrenamiento), esta partición inicial es **closed-set** para entrenar la CNN custom: cada identidad aparece en los tres splits, con imágenes distintas. El escenario open-set se evaluará en la notebook `2-train.ipynb`.
#
# Proporciones por identidad: **70% train / 15% validation / 15% test**.
# Semilla fija (`SEED = 42`) y guardado en CSV para reproducibilidad.

# %%
import json

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Split closed-set: cada identidad queda representada en train, val y test.
# El hash fijo evita depender del orden original de las filas.
split_source = (
    balanced.with_columns(
        pl.struct(["identity", "path"]).hash(seed=SEED).alias("_split_order")
    )
    .sort(["identity", "_split_order"])
    .with_columns([
        (pl.col("path").cum_count().over("identity") - 1).alias("_position"),
        pl.len().over("identity").alias("_identity_size"),
    ])
    .with_columns([
        (pl.col("_identity_size") * TRAIN_RATIO).floor().cast(pl.Int64).alias("_train_end"),
        (pl.col("_identity_size") * (TRAIN_RATIO + VAL_RATIO)).floor().cast(pl.Int64).alias("_val_end"),
    ])
    .with_columns(
        pl.when(pl.col("_position") < pl.col("_train_end"))
        .then(pl.lit("train"))
        .when(pl.col("_position") < pl.col("_val_end"))
        .then(pl.lit("val"))
        .otherwise(pl.lit("test"))
        .alias("split")
    )
)

train_df = split_source.filter(pl.col("split") == "train")
val_df = split_source.filter(pl.col("split") == "val")
test_df = split_source.filter(pl.col("split") == "test")

# Verificación: imágenes únicas y todas las identidades presentes en cada split.
train_paths = set(train_df["path"].to_list())
val_paths = set(val_df["path"].to_list())
test_paths = set(test_df["path"].to_list())

leak_train_val = len(train_paths & val_paths)
leak_train_test = len(train_paths & test_paths)
leak_val_test = len(val_paths & test_paths)
total_identities = balanced["identity"].n_unique()

split_summary = pl.DataFrame({
    "split": ["train", "val", "test", "total"],
    "identidades": [
        train_df["identity"].n_unique(),
        val_df["identity"].n_unique(),
        test_df["identity"].n_unique(),
        total_identities,
    ],
    "imagenes": [train_df.height, val_df.height, test_df.height, balanced.height],
    "imgs_por_identidad": [
        round(train_df.height / train_df["identity"].n_unique(), 1),
        round(val_df.height / val_df["identity"].n_unique(), 1),
        round(test_df.height / test_df["identity"].n_unique(), 1),
        round(balanced.height / total_identities, 1),
    ],
})

print("=== Splits ===")
display(split_summary)
print(f"Fuga train↔val: {leak_train_val}, train↔test: {leak_train_test}, val↔test: {leak_val_test}")
assert leak_train_val == 0 and leak_train_test == 0 and leak_val_test == 0, "Fuga de datos detectada"
assert all(df["identity"].n_unique() == total_identities for df in [train_df, val_df, test_df]), "Faltan identidades en algún split"

# Guardar splits
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

train_df.select(["path", "identity"]).write_csv(OUTPUT_DIR / "train.csv")
val_df.select(["path", "identity"]).write_csv(OUTPUT_DIR / "val.csv")
test_df.select(["path", "identity"]).write_csv(OUTPUT_DIR / "test.csv")

with open(OUTPUT_DIR / "split_info.json", "w", encoding="utf-8") as f:
    json.dump({
        "seed": SEED,
        "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "strategy": "closed-set: split estratificado dentro de cada identidad",
        "identities": balanced["identity"].sort().to_list(),
    }, f, indent=2)

print(f"Splits guardados en {OUTPUT_DIR}/")

# %% [markdown]
# ## Ética, privacidad y licencias
#
# ### Dataset
# Usamos **Pins Face Recognition** (Kaggle), un dataset de imágenes públicas de celebridades. Las imágenes provienen de fuentes web públicas; verificamos que el uso es para fines de investigación/académicos, consistente con el propósito de este trabajo.
#
# ### Sesgos
# El dataset está compuesto principalmente por celebridades de cine y TV occidental. Esto introduce sesgos demográficos notables:
# - **Género**: predominio femenino (el dataset está diseñado como "beauty dataset").
# - **Edad**: adultos jóvenes predominantes.
# - **Tono de piel**: infra-representación de tonos oscuros.
#
# Estos sesgos impactan directamente en el desempeño: un modelo entrenado solo con este dataset tendrá **menor accuracy al reconocer personas fuera de estas distribuciones demográficas**.
#
# ### Marco legal
# Los rostros son **datos biométricos sensibles** según la **Ley 25.326 de Protección de Datos Personales** (Argentina). El reconocimiento facial implica riesgos de:
# - **Vigilancia masiva** sin consentimiento.
# - **Sesgos algorítmicos** que amplían discriminación existente.
# - **Falsos positivos** que pueden llevar a identificaciones erróneas.
#
# Para un despliegue real de esta aplicación sería obligatorio:
# 1. Obtener **consentimiento informado explícito** de cada persona registrada.
# 2. Documentar el propósito específico del tratamiento de los datos biométricos.
# 3. Garantizar el derecho al olvido (eliminación de embeddings y fotos).
# 4. Auditar el desempeño por subgrupos demográficos.
#
# ### Licencias de modelos preentrenados
# Los modelos de InsightFace (ArcFace, buffalo_l) se distribuyen bajo licencia de **uso no comercial / investigación**. Su uso en una aplicación comercial requeriría acuerdos de licenciamiento adicionales.
