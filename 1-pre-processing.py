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
# # Notebook de documentacion, tratamiento datos y entrenamiento
#

# %% [markdown]
# ## Equipo
# - Alumno 1 : **Alejandro Armas**
# - Alumno 2 : **Gabriel Soda**

# %%
import warnings
from pathlib import Path

import altair as alt
import cv2
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from IPython.display import display
from matplotlib.patches import Rectangle
from PIL import Image, UnidentifiedImageError

alt.data_transformers.disable_max_rows()

DATA_DIR = Path("data/images")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FACE_SIZE = 112
MIN_FACE_SIDE = FACE_SIZE // 2
DETECTION_THRESHOLD = 0.5
DETECTION_SIZE = (160, 160)

# %%
# Escaneamos metadatos: no cargamos todos los píxeles ni imágenes completas en memoria.
# Cada subcarpeta representa una identidad en el dataset Pins.
records = []
candidates = sorted(
    path
    for path in DATA_DIR.rglob("*")
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
                "width": None,
                "height": None,
                "aspect_ratio": None,
                "pixels": None,
                "format": None,
                "mode": None,
                "brightness": None,
                "contrast": None,
                "readable": False,
                "error": type(error).__name__,
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

summary = pl.DataFrame(
    {
        "métrica": [
            "Total imágenes",
            "Imágenes no legibles",
            "Identidades",
            "Imgs/identidad (media)",
            "Imgs/identidad (mediana)",
            "Mín / Máx por identidad",
        ],
        "valor": [
            str(readable.height),
            str(unreadable_count),
            str(identity_counts.height),
            f"{identity_counts['count'].mean():.1f}",
            str(identity_counts["count"].median()),
            f"{identity_counts['count'].min()} / {identity_counts['count'].max()}",
        ],
    }
)
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

hist = (
    alt.Chart(data)
    .mark_bar(opacity=0.6, color="#4c78a8")
    .encode(
        x=alt.X("count:Q", bin=alt.Bin(step=BIN), title="Imágenes por identidad"),
        y=alt.Y("count():Q", title="Identidades"),
    )
)

kde = (
    alt.Chart(data)
    .transform_density(
        "count",
        as_=["count", "density"],
        extent=[identity_counts["count"].min(), identity_counts["count"].max()],
        steps=100,
    )
    .transform_calculate(scaled_density=f"datum.density * {N * BIN}")
    .mark_line(color="#e45756", strokeWidth=2)
    .encode(x="count:Q", y="scaled_density:Q")
)

display((hist + kde).properties(width=600, height=300, title="Desbalance de clases"))

# Top 10 y Bottom 10
top10 = identity_counts.head(10)
bottom10 = identity_counts.tail(10).sort("count")

top_chart = (
    alt.Chart(alt.InlineData(values=top10.to_dicts()))
    .mark_bar(color="#4c78a8")
    .encode(
        x=alt.X("count:Q", title="Imágenes"),
        y=alt.Y("identity:N", sort="-x", title=None),
    )
    .properties(title="Top 10 identidades", width=300, height=250)
)

bottom_chart = (
    alt.Chart(alt.InlineData(values=bottom10.to_dicts()))
    .mark_bar(color="#e45756")
    .encode(
        x=alt.X("count:Q", title="Imágenes"),
        y=alt.Y("identity:N", sort="x", title=None),
    )
    .properties(title="Bottom 10 identidades", width=300, height=250)
)

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

res_chart = (
    alt.Chart(alt.InlineData(values=res_bins.to_dicts()))
    .mark_rect()
    .encode(
        x=alt.X("w_bin:O", title="Ancho (px)"),
        y=alt.Y("h_bin:O", title="Alto (px)"),
        color=alt.Color("n:Q", title="Imágenes", scale=alt.Scale(scheme="blues")),
        tooltip=[
            alt.Tooltip("w_bin:Q", title="Ancho (px)"),
            alt.Tooltip("h_bin:Q", title="Alto (px)"),
            alt.Tooltip("n:Q", title="Imágenes"),
        ],
    )
    .properties(
        width=400, height=400, title="Distribución de resoluciones (bins 32 px)"
    )
)
display(res_chart)

# %%
# Muestra visual por identidad (pose, expresión, oclusiones)
N_IDENTITIES = 5
N_SAMPLES = 5

sample_idents = (
    identity_counts.head(30).sample(n=N_IDENTITIES, seed=42)["identity"].to_list()
)
fig, axes = plt.subplots(N_IDENTITIES, N_SAMPLES, figsize=(10, 2 * N_IDENTITIES))

for i, ident in enumerate(sample_idents):
    n_available = readable.filter(pl.col("identity") == ident).height
    paths = (
        readable.filter(pl.col("identity") == ident)
        .sample(n=min(N_SAMPLES, n_available), seed=42)["path"]
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

photo_chart = (
    alt.Chart(alt.InlineData(values=photo_long.to_dicts()))
    .mark_bar(opacity=0.7)
    .encode(
        x=alt.X("value:Q", bin=alt.Bin(maxbins=30), title="Valor"),
        y=alt.Y("count():Q", title="Imágenes"),
        color=alt.Color("metric:N", title=None),
        column=alt.Column("metric:N", title=None),
    )
    .properties(width=280, height=240, title="Distribución de brillo y contraste")
)
display(photo_chart)

# %%
# Casos extremos para inspección visual
extremes = pl.concat(
    [
        readable.sort("brightness")
        .head(3)
        .with_columns(pl.lit("brillo bajo").alias("case")),
        readable.sort("brightness", descending=True)
        .head(3)
        .with_columns(pl.lit("brillo alto").alias("case")),
        readable.sort("contrast")
        .head(3)
        .with_columns(pl.lit("contraste bajo").alias("case")),
        readable.sort("contrast", descending=True)
        .head(3)
        .with_columns(pl.lit("contraste alto").alias("case")),
    ]
)

fig, axes = plt.subplots(4, 3, figsize=(8, 10), layout="constrained")
fig.get_layout_engine().set(w_pad=0.02, h_pad=0.04, wspace=0.02, hspace=0.02)

for ax, row in zip(axes.flat, extremes.iter_rows(named=True)):
    with Image.open(DATA_DIR / row["path"]) as img:
        ax.imshow(img.convert("RGB"))
    ax.set_title(
        f"{row['case']} · {row['identity']}\n"
        f"b={row['brightness']:.0f}, c={row['contrast']:.0f}",
        fontsize=8,
    )
    ax.axis("off")

for ax in axes.flat[len(extremes) :]:
    ax.axis("off")

fig.suptitle("Casos extremos de iluminación y contraste", fontsize=12)
plt.show()

# %% [markdown]
# ## Control de calidad y limpieza
#
# Aplicamos reglas de filtrado **en orden**, reportando cuántas imágenes descarta cada una.
# El orden prioriza eficiencia: descartamos identidades con muy pocas muestras **antes**
# de correr detección facial. La detección se usa luego para verificar que exista al menos
# un rostro y para seleccionar el rostro principal que se alineará.
#
# Reglas aplicadas:
# 1. Identidades con menos de 100 imágenes (descartamos la identidad completa).
# 2. Rostro principal con lado mínimo de la bbox <56px.
# 3. Detector: 0 rostros detectados (imagen OK pero sin cara visible).
#
# Si el detector encuentra uno o más rostros, se conserva el de mayor confianza como rostro
# principal. La cantidad de detecciones se registra como metadato, pero las imágenes con
# múltiples rostros no se descartan. Las caras aceptadas se normalizan a 112x112 px:
# las menores se amplían y las mayores se reducen.

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

eligible_identities = identity_counts_all.filter(
    pl.col("count") >= MIN_IMAGES_PER_IDENTITY
)
low_count_identities = identity_counts_all.filter(
    pl.col("count") < MIN_IMAGES_PER_IDENTITY
)

images_after_rule1 = readable.filter(
    pl.col("identity").is_in(eligible_identities["identity"].implode())
)

# Pre-detección: Regla 1
pre_detection_summary = pl.DataFrame(
    {
        "regla": [
            "0. Imágenes legibles (base)",
            "1. Identidades < 100 imgs",
        ],
        "identidades_descartadas": [
            None,
            low_count_identities.height,
        ],
        "imagenes_descartadas": [
            0,
            readable.height - images_after_rule1.height,
        ],
        "imagenes_restantes": [
            readable.height,
            images_after_rule1.height,
        ],
    }
)

print("=== Resumen pre-detección ===")
display(pre_detection_summary)
print(f"Identidades descartadas por regla 1 ({low_count_identities.height}):")
display(low_count_identities.sort("count"))

# %%
smallest = (
    readable.with_columns(pl.min_horizontal("width", "height").alias("min_side"))
    .sort(["min_side", "pixels", "path"])
    .head(12)
)

fig, axes = plt.subplots(3, 4, figsize=(12, 10))

for ax, row in zip(axes.flat, smallest.iter_rows(named=True)):
    with Image.open(DATA_DIR / row["path"]) as image:
        ax.imshow(image.convert("RGB"))

    ax.set_title(
        f"{row['identity']}\n"
        f"{row['width']} × {row['height']} px | "
        f"lado menor: {row['min_side']} px",
        fontsize=9,
    )
    ax.axis("off")

for ax in axes.flat[smallest.height :]:
    ax.axis("off")

fig.suptitle("Imágenes con menor lado mínimo")
plt.tight_layout()
plt.show()

# %% [markdown]
# Preguntas abiertas luego de esta exploración de imágenes pequeñas:
#
# - El detector encuentra estas caras? comprobar graficando boxes, scores y landmarks.
# - ayudan a entrenar el reconocedor? implica comparar entrenamientos -> ablación, quitar un componente y observar qué cambia.

# %% [markdown]
# ## Calibración del umbral de detección
#
# Prueba sobre un subsample del ~25% de las imágenes (4225 imágenes) para determinar el umbral de detección que mejor clasifica las imágenes en 0, 1 o 2+ rostros detectados. Se inspeccionan visualmente los falsos positivos y negativos para ajustar el umbral.

# %%
SUBSAMPLE_SIZE = 4225
test_sample = images_after_rule1.sample(
    n=min(SUBSAMPLE_SIZE, images_after_rule1.height), seed=42
)

thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
results = []
examples = {thresh: {"0": [], "1": [], "2+": []} for thresh in thresholds}

# Detectamos UNA VEZ por imagen con umbral 0.1 (suficientemente bajo
# para capturar caras reales, pero evita la avalancha de basura de 0.0).
tune_detector = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection"],
    providers=["CPUExecutionProvider"],
)
tune_detector.prepare(ctx_id=-1, det_thresh=0.1, det_size=DETECTION_SIZE)

images_with_small_bbox = 0
detection_cache = []
for row in test_sample.iter_rows(named=True):
    img = cv2.imread(str(DATA_DIR / row["path"]))
    if img is None:
        continue
    faces = tune_detector.get(img)
    face_data = []
    for f in faces:
        face_data.append(
            {
                "score": float(f.det_score),
                "bbox": np.asarray(f.bbox, dtype=float).tolist(),
                "keypoints": (
                    np.asarray(f.kps, dtype=float).tolist()
                    if f.kps is not None
                    else None
                ),
            }
        )
    detection_cache.append(
        {
            "path": row["path"],
            "image_size": [int(img.shape[1]), int(img.shape[0])],
            "faces": face_data,
        }
    )

for thresh in thresholds:
    counts = {"0": 0, "1": 0, "2+": 0}
    for item in detection_cache:
        n = sum(1 for f in item["faces"] if f["score"] >= thresh)
        cat = "0" if n == 0 else "1" if n == 1 else "2+"
        counts[cat] += 1
        if len(examples[thresh][cat]) < 2:
            examples[thresh][cat].append(item["path"])
    results.append(
        {
            "threshold": thresh,
            "0_faces": counts["0"],
            "1_face": counts["1"],
            "2+_faces": counts["2+"],
        }
    )

tune_df = pl.from_dicts(results)
display(tune_df)


# %% [markdown]
# Inspección sin repetir inferencia, sobre coordenadas de la imagen original:
# **verde**: mayor score entre las detecciones que pasan el umbral;
# **naranja**: otras aceptadas; **gris discontinuo**: candidatas bajo el umbral.
# Los puntos indican landmarks. La identidad es la etiqueta
# esperada, no es una predicción. El score es de detección.
# Ejecutamos el detector con umbral 0,1 y guardamos las detecciones que devuelve.
# Sobreeso, probamos umbrales más exigentes como 0,3 o 0,6, sin ejecutar nuevamente la red.


# %%
def show_threshold_samples(
    threshold: float,
    category: str,
    detection_cache,
    n: int = 6,
    seed: int = 42,
    show_below_threshold: bool = True,
):
    """Dibuja cajas, scores y landmarks cacheados; devuelve las rutas mostradas."""
    if category not in {"0", "1", "2+"}:
        raise ValueError("category debe ser '0', '1' o '2+'.")
    if n < 1:
        raise ValueError("n debe ser mayor que cero.")

    all_matched = []
    for item in detection_cache:
        n_faces = sum(face["score"] >= threshold for face in item["faces"])
        cat = "0" if n_faces == 0 else "1" if n_faces == 1 else "2+"
        if cat == category:
            all_matched.append(item["path"])

    n = min(n, len(all_matched))
    if n == 0:
        print(f"No hay imágenes para t={threshold}, categoría={category}.")
        return []

    cache_by_path = {item["path"]: item for item in detection_cache}
    rng = np.random.default_rng(seed)
    matched = rng.choice(all_matched, size=n, replace=False).tolist()
    cols = min(n, 3 if n <= 6 else 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()

    for ax, path in zip(axes, matched):
        img = cv2.imread(str(DATA_DIR / path))
        if img is None:
            ax.set_title(f"No se pudo leer: {path}", fontsize=9)
            ax.axis("off")
            continue

        faces = cache_by_path[path]["faces"]
        if any("bbox" not in face for face in faces):
            plt.close(fig)
            raise ValueError(
                "La caché no tiene cajas. Volvé a ejecutar la celda de detección."
            )
        passing = [i for i, face in enumerate(faces) if face["score"] >= threshold]
        selected = max(passing, key=lambda i: faces[i]["score"], default=None)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        for i, face in enumerate(faces):
            passes = face["score"] >= threshold
            if not passes and not show_below_threshold:
                continue
            color = "limegreen" if i == selected else "darkorange" if passes else "gray"
            x1, y1, x2, y2 = face["bbox"]
            ax.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor=color,
                    linewidth=2 if passes else 1,
                    linestyle="-" if passes else "--",
                    alpha=1 if passes else 0.6,
                )
            )
            label = f"#{i + 1}: {face['score']:.3f}"
            if i == selected:
                label += " principal"
            elif not passes:
                label += " < umbral"
            ax.text(
                max(0, x1),
                max(0, y1),
                label,
                fontsize=8,
                color="white",
                va="top",
                bbox={"facecolor": color, "alpha": 0.8, "pad": 1},
            )
            if face.get("keypoints") is not None:
                points = np.asarray(face["keypoints"])
                ax.scatter(points[:, 0], points[:, 1], s=12, c=color)

        # Mantener la vista original aunque una caja predicha exceda sus bordes.
        ax.set_xlim(-0.5, img.shape[1] - 0.5)
        ax.set_ylim(img.shape[0] - 0.5, -0.5)
        identity = Path(path).parent.name.removeprefix("pins_")
        ax.set_title(
            f"{identity}\n{len(passing)} detecciones >= {threshold}", fontsize=9
        )
        ax.axis("off")

    for ax in axes[len(matched) :]:
        ax.axis("off")
    fig.suptitle(
        f"t={threshold} | categoría={category} | {len(matched)} imágenes\n"
        "Verde: mayor score · Naranja: otras aceptadas · Gris: bajo umbral",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()
    return matched


# %%
show_threshold_samples(0.60, "2+", detection_cache, n=25)

# %% [markdown]
# Decidimos quedarnos con el umbral 0.6 para la detección de rostros, buen balance

# %%
DETECTION_THRESHOLD = 0.6

# %%
# Detección facial con SCRFD: se conserva el rostro principal si hay varias detecciones.
detector = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection"],
    providers=["CPUExecutionProvider"],
)
detector.prepare(ctx_id=-1, det_thresh=DETECTION_THRESHOLD, det_size=DETECTION_SIZE)

detection_records = []
for index, row in enumerate(images_after_rule1.iter_rows(named=True), start=1):
    image_bgr = cv2.imread(str(DATA_DIR / row["path"]))
    if image_bgr is None:
        faces = []
    else:
        faces = detector.get(image_bgr)

    face_count = len(faces)

    if face_count == 0:
        x1 = y1 = x2 = y2 = bbox_width = bbox_height = min_bbox_side = None
        keypoints = None
        requires_upscale = requires_downscale = None
        quality_rule = "3. Sin rostro detectado"

    else:
        # Elegimos explícitamente la detección de mayor confianza como rostro principal.
        face = max(faces, key=lambda detected_face: float(detected_face.det_score))
        x1, y1, x2, y2 = face.bbox.astype(int).tolist()
        bbox_width, bbox_height = x2 - x1, y2 - y1
        min_bbox_side = min(bbox_width, bbox_height)
        keypoints = face.kps.astype(np.float32).tolist()
        requires_upscale = min_bbox_side < FACE_SIZE
        requires_downscale = min_bbox_side > FACE_SIZE
        quality_rule = (
            "2. Rostro principal menor a 56 px"
            if min_bbox_side < MIN_FACE_SIDE
            else "Aceptada"
        )

    detection_records.append(
        {
            "path": row["path"],
            "identity": row["identity"],
            "face_count": face_count,
            "detection_score": (float(face.det_score) if face_count > 0 else None),
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "bbox_width": bbox_width,
            "bbox_height": bbox_height,
            "min_bbox_side": min_bbox_side,
            "keypoints": keypoints,
            "requires_upscale_after_detection": requires_upscale,
            "requires_downscale_after_detection": requires_downscale,
            "quality_rule": quality_rule,
        }
    )

    if index % 500 == 0 or index == images_after_rule1.height:
        print(f"Procesadas {index:,}/{images_after_rule1.height:,} imágenes")

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
    post_detection_counts.filter(~pl.col("quality_rule").str.starts_with("Aceptada"))
    .with_columns(
        pl.lit(None).cast(pl.Int64).alias("identidades_descartadas"),
        (
            pl.lit(images_after_rule1.height, dtype=pl.Int64)
            - pl.col("imagenes_descartadas").cum_sum()
        ).alias("imagenes_restantes"),
    )
    .rename({"quality_rule": "regla"})
    .select(
        [
            "regla",
            "identidades_descartadas",
            "imagenes_descartadas",
            "imagenes_restantes",
        ]
    )
)

# Pipeline completo: conteo por regla
pipeline_summary = pl.concat(
    [
        pre_detection_summary,
        rejected_rules,
    ]
)

# Total final aceptado
final_accepted = accepted.height
final_identities = accepted["identity"].n_unique()

pipeline_summary = pl.concat(
    [
        pipeline_summary,
        pl.DataFrame(
            {
                "regla": ["Aceptadas (todas las reglas)"],
                "identidades_descartadas": [None],
                "imagenes_descartadas": [0],
                "imagenes_restantes": [final_accepted],
            }
        ),
    ]
)

print("\n=== Pipeline completo de calidad ===")
display(pipeline_summary)

print(f"\nAceptadas: {final_accepted} imágenes de {final_identities} identidades")
print(
    f"Requieren ampliación: {accepted.filter(pl.col('requires_upscale_after_detection')).height}"
)
print(
    f"Requieren reducción: {accepted.filter(pl.col('requires_downscale_after_detection')).height}"
)
print(
    f"Con múltiples rostros (se conserva el principal): {accepted.filter(pl.col('face_count') > 1).height}"
)

# %% [markdown]
# Fijamos que ambos lados de la caja facial tengan al menos 56 píxeles para evitar ampliar rostros con muy poca resolución original antes de prepararlos a 112×112.

# %%
# Ejemplos descartados: imagen original, caja, landmarks y medidas.

rejected = detections.filter(~pl.col("quality_rule").str.starts_with("Aceptada"))

if rejected.is_empty():
    print("No hay imágenes descartadas por las reglas de detección.")
else:
    rules = rejected["quality_rule"].unique().sort().to_list()

    for rule in rules:
        # Conservamos los primeros tres ejemplos de cada regla.
        examples = rejected.filter(pl.col("quality_rule") == rule).head(3)

        fig, axes = plt.subplots(
            1,
            examples.height,
            figsize=(5 * examples.height, 5),
            squeeze=False,
        )

        for ax, row in zip(axes.flat, examples.iter_rows(named=True)):
            with Image.open(DATA_DIR / row["path"]) as image:
                image_rgb = image.convert("RGB")
                width, height = image_rgb.size
                ax.imshow(image_rgb)

            title = f"{row['identity']}\nImagen: {width} × {height} px"

            if row["bbox_x1"] is not None:
                x1, y1 = row["bbox_x1"], row["bbox_y1"]
                x2, y2 = row["bbox_x2"], row["bbox_y2"]

                ax.add_patch(
                    Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                    )
                )

                if row["keypoints"] is not None:
                    points = np.asarray(row["keypoints"])
                    ax.scatter(
                        points[:, 0],
                        points[:, 1],
                        s=18,
                        c="limegreen",
                    )

                title += (
                    f"\nCaja: {row['bbox_width']} × "
                    f"{row['bbox_height']} px"
                    f"\nLado menor: {row['min_bbox_side']} px"
                    f" | Score: {row['detection_score']:.3f}"
                )
            else:
                title += "\nSin detección aceptada"

            ax.set_title(title, fontsize=9)
            ax.set_xlim(-0.5, width - 0.5)
            ax.set_ylim(height - 0.5, -0.5)
            ax.axis("off")

        fig.suptitle(f"Rechazo: {rule}", fontsize=12)
        plt.tight_layout()
        plt.show()

# %% [markdown]
# Los ejemplos descartados por tamaño presentan rostros localizados con scores superiores al umbral de detección, pero con lados mínimos de 53–54 px, próximos al corte de 56 px.

# %% [markdown]
# ## Detección, landmarks y alineación con SCRFD
#
# A continuación visualizamos el pipeline `detectar → validar → alinear` sobre las imágenes que pasaron todas las reglas de calidad.

# %%
# Antes/después: usamos los 5 landmarks almacenados por SCRFD para alinear a FACE_SIZE.
alignment_examples = accepted.sample(n=min(4, accepted.height), seed=42)
fig, axes = plt.subplots(
    alignment_examples.height, 2, figsize=(8, 3 * alignment_examples.height)
)
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
# ## Revisión manual de imágenes con varios rostros
#
# Tras los filtros automáticos, inspeccionamos todas las imágenes con varias detecciones, antes de construir el conjunto balanceado. El score mide confianza de deteccion, no identidad: la cara de mayor score puede no corresponder a la persona de la carpeta.
#
# Primero mostramos los casos y sus recortes. Luego documentamos las selecciones incorrectas y excluimos esas rutas del conjunto aceptado. No modificamos los archivos originales. Verde: rostro elegido por score. Naranja: otras detecciones. Los recortes mostrados son cajas, no la alineacion final de ArcFace.

# %%
# 1. Casos con varias detecciones, antes del balanceo.
audit_candidates = detections.filter(pl.col("face_count") > 1).sort(
    ["identity", "path"]
)
print("Imágenes con varias caras:", audit_candidates.height)
print(
    "Aceptadas por filtros automaticos:",
    audit_candidates.filter(pl.col("quality_rule").str.starts_with("Aceptada")).height,
)
display(audit_candidates.select("identity", "path", "face_count", "quality_rule"))

# %%
# 2. Recuperar todas las cajas SOLO para los pocos casos anteriores.
# Reutilizamos el detector del pipeline completo (mismo umbral y resolucion).
# No usamos detection_cache: esa lista solo cubre la muestra de calibracion.
multi_face_audit = []
audit_failures = []
for row in audit_candidates.iter_rows(named=True):
    image = cv2.imread(str(DATA_DIR / row["path"]))
    if image is None:
        audit_failures.append({"path": row["path"], "motivo": "No se pudo leer"})
        continue
    faces = sorted(detector.get(image), key=lambda f: float(f.det_score), reverse=True)
    if len(faces) < 2:
        audit_failures.append(
            {"path": row["path"], "motivo": "Ahora hay menos de 2 detecciones"}
        )
        continue
    boxes = [np.asarray(face.bbox, dtype=float) for face in faces]
    areas = [max(0, b[2] - b[0]) * max(0, b[3] - b[1]) for b in boxes]
    multi_face_audit.append(
        {
            **row,
            "faces": faces,
            "score_1": float(faces[0].det_score),
            "score_2": float(faces[1].det_score),
            "diferencia_score": float(faces[0].det_score - faces[1].det_score),
            "elegida_es_mayor": bool(areas[0] == max(areas)),
        }
    )

if multi_face_audit:
    audit_table = pl.from_dicts(
        [
            {key: value for key, value in item.items() if key != "faces"}
            for item in multi_face_audit
        ]
    ).with_row_index("caso")
    display(
        audit_table.select(
            "caso",
            "identity",
            "path",
            "score_1",
            "score_2",
            "diferencia_score",
            "elegida_es_mayor",
        )
    )
    print("Efecto hipotetico del margen 0.1; NO se aplica ningun descarte:")
    display(
        audit_table.select(
            pl.len().alias("casos"),
            (pl.col("diferencia_score") <= 0.1).sum().alias("descartaria_margen_01"),
            (~pl.col("elegida_es_mayor")).sum().alias("elegida_no_es_la_mayor"),
        )
    )
else:
    print("No hay casos multiples para inspeccionar.")
if audit_failures:
    display(pl.from_dicts(audit_failures))

# %%
# 3. Recorrer TODOS los casos en paginas estables.
# Para buscar el ejemplo puntual: AUDIT_IDENTITY = "Robert De Niro".
AUDIT_IDENTITY = ""
AUDIT_PAGE = 0
AUDIT_PAGE_SIZE = 4

audit_view = [
    (index, item)
    for index, item in enumerate(multi_face_audit)
    if AUDIT_IDENTITY.lower() in item["identity"].lower()
]
audit_pages = (len(audit_view) + AUDIT_PAGE_SIZE - 1) // AUDIT_PAGE_SIZE
print(f"Casos: {len(audit_view)}. Paginas: {audit_pages}; numeradas desde 0.")
audit_page = audit_view[
    AUDIT_PAGE * AUDIT_PAGE_SIZE : (AUDIT_PAGE + 1) * AUDIT_PAGE_SIZE
]

if audit_page:
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(
        len(audit_page), 3, figsize=(13, 4 * len(audit_page)), squeeze=False
    )
    for row_axes, (index, item) in zip(axes, audit_page):
        image = cv2.imread(str(DATA_DIR / item["path"]))
        if image is None:
            for ax in row_axes:
                ax.axis("off")
            print("No se pudo leer:", item["path"])
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        row_axes[0].imshow(rgb)
        for rank, face in enumerate(item["faces"]):
            x1, y1, x2, y2 = face.bbox
            color = "limegreen" if rank == 0 else "darkorange"
            row_axes[0].add_patch(
                Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2
                )
            )
            row_axes[0].text(
                max(0, x1),
                max(0, y1),
                f"#{rank + 1}: {face.det_score:.3f}",
                color="white",
                fontsize=8,
                bbox=dict(facecolor=color, alpha=0.8),
            )
        row_axes[0].set_xlim(-0.5, rgb.shape[1] - 0.5)
        row_axes[0].set_ylim(rgb.shape[0] - 0.5, -0.5)
        row_axes[0].set_title(
            f"Caso {index}: {item['identity']}\n"
            f"Diferencia: {item['diferencia_score']:.3f}",
            fontsize=9,
        )
        for ax, face, title in zip(
            row_axes[1:], item["faces"][:2], ["Elegida por score", "Segunda por score"]
        ):
            x1, y1, x2, y2 = face.bbox
            x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
            x2, y2 = (
                min(rgb.shape[1], int(np.ceil(x2))),
                min(rgb.shape[0], int(np.ceil(y2))),
            )
            if x2 > x1 and y2 > y1:
                ax.imshow(rgb[y1:y2, x1:x2])
            ax.set_title(f"{title}: {face.det_score:.3f}", fontsize=10)
        for ax in row_axes:
            ax.axis("off")
        print(f"Caso {index}: {item['path']}")
    plt.tight_layout()
    plt.show()
else:
    print("No hay casos en esta pagina. Revisar filtros y numero de pagina.")

# %% [markdown]
# ### Resultado de la revision visual
#
# Se revisaron los 62 casos con varios rostros y se identificaron cuatro imágenes en las que la cara seleccionada no corresponde a la identidad etiquetada de la carpeta (la identidad principal). Las registramos por ruta por si hay cambios posteriores que modifiquen el orden del dataset.
#
# La siguiente visualización muestra esos casos con errores. Después los excluimos del conjunto aceptado antes de balancear. Si una imagen ya fue descartada por un filtro automático, la exclusion manual no vuelve a restarla.

# %% [markdown]
# En un paso posterior del desarrollo se detectó una imagen mal etiquetada del dataset. La identidad principal etiquetada era Johnny Depp, pero en la imagen se puede ver claramente que es Leonardo Di Caprio

# %%
image_path = Path("pins_Johnny Depp/Johnny Depp116_1780.jpg")
identity = image_path.parent.name.removeprefix("pins_")
image = cv2.imread(str(DATA_DIR / image_path))

plt.figure(figsize=(3, 3))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(f"Etiqueta original: {identity}\nExcluida por etiquetado incorrecto")
plt.axis("off")
plt.show()

# %%
# Rutas confirmadas durante la revision visual. Se conservan los originales.
EXCLUDED_IMAGES = [
    r"pins_Morgan Freeman\Morgan Freeman80_3131.jpg",
    r"pins_Robert De Niro\Robert De Niro163_700.jpg",
    r"pins_Chris Evans\Chris Evans94_1272.jpg",
    r"pins_Jimmy Fallon\Jimmy Fallon143_2779.jpg",
    r"pins_Johnny Depp\Johnny Depp116_1780.jpg",
]

# Derivar los indices actuales de las rutas revisadas, no al reves.
audit_excluded_paths = {path.replace("\\", "/") for path in EXCLUDED_IMAGES}
AUDIT_WRONG_CASES = [
    index
    for index, item in enumerate(multi_face_audit)
    if item["path"].replace("\\", "/") in audit_excluded_paths
]
wrong_items = [
    {
        "path": multi_face_audit[index]["path"],
        "faces": [
            {
                "score": float(face.det_score),
                "bbox": np.asarray(face.bbox).tolist(),
                "keypoints": np.asarray(face.kps).tolist()
                if face.kps is not None
                else None,
            }
            for face in multi_face_audit[index]["faces"]
        ],
    }
    for index in AUDIT_WRONG_CASES
]
print("Selecciones incorrectas encontradas en la auditoria:", len(wrong_items))
for index in AUDIT_WRONG_CASES:
    print(f"Caso {index}: {multi_face_audit[index]['path']}")


# %%
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

for ax, case, item in zip(axes.flat, AUDIT_WRONG_CASES, wrong_items):
    img = cv2.imread(str(DATA_DIR / item["path"]))
    ax.axis("off")

    if img is None:
        ax.set_title(f"Caso {case}: no se pudo leer")
        continue

    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for rank, face in enumerate(item["faces"]):
        color = "limegreen" if rank == 0 else "darkorange"
        x1, y1, x2, y2 = face["bbox"]

        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linewidth=2,
            )
        )
        ax.text(
            max(0, x1),
            max(0, y1),
            f"#{rank + 1}: {face['score']:.3f}",
            color="white",
            fontsize=9,
            bbox=dict(facecolor=color, alpha=0.8),
        )

        if face["keypoints"] is not None:
            points = np.asarray(face["keypoints"])
            ax.scatter(points[:, 0], points[:, 1], c=color, s=12)

    ax.set_xlim(-0.5, img.shape[1] - 0.5)
    ax.set_ylim(img.shape[0] - 0.5, -0.5)
    identity = multi_face_audit[case]["identity"]
    ax.set_title(f"Caso {case}: {identity}")

fig.suptitle(
    "Selecciones incorrectas\nVerde: rostro elegido · Naranja: otros rostros",
    fontsize=14,
)
plt.tight_layout()
plt.show()

# %%
# Excluir antes de balancear; partir de los filtros automaticos en cada ejecucion.
accepted_before_manual_review = detections.filter(
    pl.col("quality_rule").str.starts_with("Aceptada")
)
accepted = accepted_before_manual_review.filter(
    ~pl.col("path")
    .str.replace_all("\\", "/", literal=True)
    .is_in(list(audit_excluded_paths))
)
print("Aceptadas por filtros automaticos:", accepted_before_manual_review.height)
print(
    "Excluidas adicionalmente por revision manual:",
    accepted_before_manual_review.height - accepted.height,
)
print("Aceptadas despues de revision manual:", accepted.height)


# %% [markdown]
# ## Subset balanceado
#
# Partimos de las imágenes que pasaron los filtros automaticos y la revision manual. Como referencia inicial, conservamos identidades con al menos 100 imágenes aceptadas y muestreamos **exactamente 100 imágenes por identidad** (`MIN = MAX`). El muestreo es aleatorio con semilla fija preserva proporcionalmente las condiciones difíciles porque nunca filtramos por calidad fotométrica.

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

print(
    f"Se descartan {len(accepted_counts) - len(kept)} identidades con menos de {MIN_PER_ID} imágenes aceptadas"
)

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

# %%
print("Identidades excluidas después de la limpieza:")
display(accepted_counts.filter(pl.col("count") < MIN_PER_ID))

eligible_images = accepted.filter(pl.col("identity").is_in(kept))

print("Imágenes aceptadas antes del balanceo:", accepted.height)
print("Imágenes de las identidades conservadas:", eligible_images.height)
print("Imágenes seleccionadas:", balanced.height)
print(
    "Imágenes fuera de la selección por el máximo definido:",
    eligible_images.height - balanced.height,
)

# %% [markdown]
# Los filtros automáticos dejaron 16.820 imágenes aceptadas. Después de revisar visualmente las imágenes con mútiples rostros, se registraron cuatro selecciones incorrectas. dos ya estaban descartadas por los filtros automáticos y otras dos se excluyeron manualmente, dejando 16.818 imágenes aceptadas.
# El balanceo reduce esas 16.818 imágenes a 10.000. Se excluyen 190 imágenes al retirar dos identidades con menos de 100 muestras y otras 6.628 por el máximo de 100 imágenes por identidad. Esta última reducción afecta imágenes que superaron tanto los filtros automáticos como la revisión manual.

# %% [markdown]
# ## Clustering exploratorio
#
# Extraemos embeddings con **ArcFace** (modelo preentrenado de `buffalo_l`) sobre el **subset balanceado**: 100 imágenes aceptadas por identidad. Proyectamos a 2D con **t-SNE** y comparamos clusters K-Means contra las etiquetas reales (ARI, NMI, silhouette). El objetivo no es decorar, sino detectar identidades mal etiquetadas, outliers o sub-clusters por condiciones de captura.

# %%
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

# True: detección y embeddings en GPU. False: ejecución original en CPU.
# t-SNE, K-Means y las métricas siguen en CPU en ambos casos.
USE_GPU = False

# True: recalcular aunque exista cache. Usarlo si cambian fotos o pesos del modelo.
FORCE_RECOMPUTE_EMBEDDINGS = False

# Un archivo por dispositivo, resolucion y umbral. No se sube a Git.
cache_dir = Path("output/cache_embeddings")
cache_dir.mkdir(parents=True, exist_ok=True)
device = "gpu" if USE_GPU else "cpu"
width, height = DETECTION_SIZE
cache_path = (
    cache_dir / f"arcface_{device}_{width}x{height}_t{DETECTION_THRESHOLD}.parquet"
)

# Solo reutilizamos embeddings si coinciden las rutas, identidades y su orden.
cached_embeddings = None
if cache_path.exists() and not FORCE_RECOMPUTE_EMBEDDINGS:
    cached_embeddings = pl.read_parquet(cache_path)
    if not cached_embeddings.select("path", "identity").equals(
        balanced.select("path", "identity")
    ):
        print("Cambió la selección de imágenes: se recalcularan los embeddings.")
        cached_embeddings = None

if cached_embeddings is not None:
    embeddings_df = cached_embeddings
    print("Cache reutilizada, sin ejecutar modelos:", cache_path)
else:
    providers = ["CPUExecutionProvider"]
    if USE_GPU:
        import os
        import sysconfig

        import onnxruntime as ort

        # En Windows, CUDA/cuDNN necesitan encontrar las DLL instaladas en la venv.
        nvidia_dir = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
        dll_dirs = sorted({str(path.parent) for path in nvidia_dir.rglob("*.dll")})
        if os.name == "nt" and dll_dirs:
            os.environ["PATH"] = (
                os.pathsep.join(dll_dirs) + os.pathsep + os.environ.get("PATH", "")
            )
        ort.preload_dlls(directory="")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                "CUDA no disponible: revisar onnxruntime-gpu o usar USE_GPU = False."
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Cargamos los mismos modelos de buffalo_l que en la version original.
    face_analysis = FaceAnalysis(
        name="buffalo_l",
        providers=providers,
        allowed_modules=["detection", "recognition"],
    )
    face_analysis.prepare(
        ctx_id=0 if USE_GPU else -1,
        det_thresh=DETECTION_THRESHOLD,
        det_size=DETECTION_SIZE,
    )
    if USE_GPU:
        for model in face_analysis.models.values():
            if "CUDAExecutionProvider" not in model.session.get_providers():
                raise RuntimeError(
                    "Un modelo no pudo habilitar CUDA. Revisar los mensajes anteriores."
                )
            model.session.disable_fallback()
    print(
        "Dispositivo solicitado para detección y embeddings:",
        "GPU" if USE_GPU else "CPU",
    )

    # Extraemos embeddings del subset balanceado: 100 imágenes por identidad.
    # Si hay varias detecciones usamos el rostro principal.
    embedding_records = []
    for row in balanced.iter_rows(named=True):
        image_bgr = cv2.imread(str(DATA_DIR / row["path"]))
        if image_bgr is None:
            continue
        faces = face_analysis.get(image_bgr)
        if faces:
            face = max(faces, key=lambda detected_face: float(detected_face.det_score))
            embedding_records.append(
                {
                    "path": row["path"],
                    "identity": row["identity"],
                    "embedding": face.embedding,
                }
            )

    if not embedding_records:
        raise ValueError("No se pudieron extraer embeddings.")
    embeddings_df = pl.from_dicts(embedding_records)

    # Guardar al terminar. El reemplazo evita reutilizar una escritura interrumpida.
    temporary_cache = cache_path.with_suffix(".tmp")
    embeddings_df.write_parquet(temporary_cache)
    temporary_cache.replace(cache_path)
    print("Cache guardada:", cache_path)
    print("Imagenes omitidas:", balanced.height - embeddings_df.height)

X = np.vstack(embeddings_df["embedding"].to_list())
y = embeddings_df["identity"].to_list()
print(
    f"Embeddings extraídos: {len(y)} imágenes de {embeddings_df['identity'].n_unique()} identidades"
)

# t-SNE
n_samples = min(2000, X.shape[0])
tsne_idx = np.random.RandomState(42).choice(X.shape[0], n_samples, replace=False)
X_tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30,
    init="pca",
    learning_rate="auto",
).fit_transform(X[tsne_idx])

# K-Means con k = n identidades
n_clusters = embeddings_df["identity"].n_unique()
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X)
labels_km = kmeans.labels_

ari = adjusted_rand_score(y, labels_km)
nmi = normalized_mutual_info_score(y, labels_km)
sil = silhouette_score(X, labels_km)

metrics = pl.DataFrame(
    {
        "métrica": [
            "ARI (K-Means vs identidad)",
            "NMI (K-Means vs identidad)",
            "Silhouette (K-Means)",
        ],
        "valor": [round(ari, 3), round(nmi, 3), round(sil, 3)],
    }
)
display(metrics)

# Plot t-SNE con Altair
tsne_data = pl.DataFrame(
    {
        "x": X_tsne[:, 0],
        "y": X_tsne[:, 1],
        "identity": [y[i] for i in tsne_idx],
        "kmeans_label": [int(labels_km[i]) for i in tsne_idx],
    }
)

scatter = (
    alt.Chart(alt.InlineData(values=tsne_data.to_dicts()))
    .mark_circle(size=30, opacity=0.6)
    .encode(
        x=alt.X("x:Q", title="t-SNE 1"),
        y=alt.Y("y:Q", title="t-SNE 2"),
        color=alt.Color(
            "identity:N",
            title="Identidad",
            scale=alt.Scale(scheme="category20"),
        ),
        tooltip=["identity:N", "kmeans_label:Q"],
    )
    .properties(
        width=550,
        height=450,
        title="t-SNE de embeddings ArcFace (coloreado por identidad)",
    )
)

display(scatter)

# %% [markdown]
# ### Interpretación del clustering
#
# - **ARI y NMI cercanos a 1** indican que los embeddings ya separan bien las identidades.
# - **Silhouette** cuantifica qué tan cohesionados están los clusters.
# - Identidades que se mezclan en el t-SNE suelen ser personas con parecido facial real o condiciones de captura similares (iluminación, pose).
# - Outliers (puntos lejos de su grupo) pueden ser errores de etiquetado o imágenes de muy baja calidad.

# %% [markdown]
# ## Data augmentation (offline)
#
# Generamos **una sola imagen aumentada por cada original del split de entrenamiento**.
# Las aumentadas se almacenan en disco junto a las originales, de modo que la
# notebook 2 recibe el set de entrenamiento ya expandido, sin necesidad de
# transformaciones on-the-fly.
#
# **Diseño del pipeline:**
# - Eliminamos transformaciones que empeoran la identidad o introducen artefactos
#   innecesarios en un dataset de baja calidad (Pins ya es web).
# - Movimos las probabilidades hacia arriba (p = 0.5–0.7) para garantizar que cada
#   muestra aumentada sea *consistentemente distinta* sin volverse irreconocible.
#
# | Transformación | Prob. | ¿Qué evitamos? |
# |---|---|---|
# | HorizontalFlip | 0.5 | Simetría facial; no destruye identidad. |
# | ShiftScaleRotate (±5% shift, ±10% scale, ±10°) | 0.6 | Variación natural de pose. |
# | RandomBrightnessContrast (±0.2) | 0.7 | Cambios de iluminación. |
# | GaussianBlur (3–5 px) | 0.3 | Desenfoque de cámara. |
# | GaussNoise (5–15 σ) | 0.3 | Ruido de sensor. |
# | ~~CoarseDropout~~ | — | Oculta landmarks; no responde a la realidad de este dataset. |
# | ~~JPEG compression~~ | — | Pins ya es web-JPEG; re-comprimir agrava artefactos. |
# | ~~MotionBlur~~ | — | Redundante con GaussianBlur. |

# %%
import hashlib

import albumentations as A

AUGMENTATION_SEED = 42

augmentation_pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(std_range=(5 / 255, 15 / 255), p=0.3),
    ],
    seed=AUGMENTATION_SEED,
)


# Visualización rápida antes de generar
def show_augmentations(image_path, n=8):
    # Cada llamada muestra la misma secuencia de variantes.
    augmentation_pipeline.set_random_seed(AUGMENTATION_SEED)
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 3))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")
    for i in range(n):
        aug = augmentation_pipeline(image=img_rgb)["image"]
        axes[i + 1].imshow(aug)
        axes[i + 1].set_title(f"Aug {i + 1}")
        axes[i + 1].axis("off")
    fig.suptitle("Pipeline de augmentation (offline)", fontsize=14)
    plt.tight_layout()
    plt.show()


sample_img = accepted.sample(n=1, seed=42)["path"].to_list()[0]
show_augmentations(DATA_DIR / sample_img)

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
    .with_columns(
        [
            (pl.col("path").cum_count().over("identity") - 1).alias("_position"),
            pl.len().over("identity").alias("_identity_size"),
        ]
    )
    .with_columns(
        [
            (pl.col("_identity_size") * TRAIN_RATIO)
            .floor()
            .cast(pl.Int64)
            .alias("_train_end"),
            (pl.col("_identity_size") * (TRAIN_RATIO + VAL_RATIO))
            .floor()
            .cast(pl.Int64)
            .alias("_val_end"),
        ]
    )
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

split_summary = pl.DataFrame(
    {
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
    }
)

print("=== Splits ===")
display(split_summary)
print(
    f"Fuga train↔val: {leak_train_val}, train↔test: {leak_train_test}, val↔test: {leak_val_test}"
)
assert leak_train_val == 0 and leak_train_test == 0 and leak_val_test == 0, (
    "Fuga de datos detectada"
)
assert all(
    df["identity"].n_unique() == total_identities for df in [train_df, val_df, test_df]
), "Faltan identidades en algún split"

# Guardar splits
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

train_df.select(["path", "identity"]).write_csv(OUTPUT_DIR / "train.csv")
val_df.select(["path", "identity"]).write_csv(OUTPUT_DIR / "val.csv")
test_df.select(["path", "identity"]).write_csv(OUTPUT_DIR / "test.csv")

with open(OUTPUT_DIR / "split_info.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "seed": SEED,
            "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
            "strategy": "closed-set: split estratificado dentro de cada identidad",
            "identities": balanced["identity"].sort().to_list(),
        },
        f,
        indent=2,
    )

print(f"Splits guardados en {OUTPUT_DIR}/")

# %% [markdown]
# ## Guardado de imágenes alineadas
#
# Generamos los recortes de rostro alineados a 112×112 para cada split,
# respetando la estructura esperada por la notebook `2-train.ipynb`:
# `output/aligned_112/{split}/{identity}/{nombre}.png`.
#
# > **Nota:** usamos el mismo detector SCRFD y la misma función
# > `face_align.norm_crop` que en el paso de calidad, para garantizar
# > consistencia entre el preprocesamiento y el entrenamiento.

# %%
ALIGNED_DIR = OUTPUT_DIR / "aligned_112"
ALIGNED_DIR.mkdir(exist_ok=True)

# Reutilizamos el detector cargado previamente; si no existe, lo instanciamos.
if "detector" not in globals():
    detector_aligned = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"],
    )
    detector_aligned.prepare(
        ctx_id=-1, det_thresh=DETECTION_THRESHOLD, det_size=DETECTION_SIZE
    )
else:
    detector_aligned = detector  # type: ignore[name-defined]

aligned_stats = []
aug_records = []  # filas a agregar en train.csv

for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    out_split = ALIGNED_DIR / split_name
    out_split.mkdir(exist_ok=True)
    saved_orig = saved_aug = skipped = 0

    for idx, row in enumerate(split_df.iter_rows(named=True), start=1):
        img_bgr = cv2.imread(str(DATA_DIR / row["path"]))
        if img_bgr is None:
            skipped += 1
            continue

        faces = detector_aligned.get(img_bgr)
        if not faces:
            skipped += 1
            continue

        face = max(faces, key=lambda f: float(f.det_score))
        aligned_bgr = face_align.norm_crop(
            img_bgr, landmark=face.kps.astype(np.float32), image_size=FACE_SIZE
        )

        out_identity = out_split / row["identity"]
        out_identity.mkdir(parents=True, exist_ok=True)
        stem = Path(row["path"]).stem
        fname = f"{stem}.png"
        cv2.imwrite(str(out_identity / fname), aligned_bgr)
        saved_orig += 1

        # Offline augmentation: solo para train, 1 aumentada por original
        if split_name == "train":
            aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
            aug_rgb = augmentation_pipeline(image=aligned_rgb)["image"]
            aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)
            aug_fname = f"{stem}_aug.png"
            cv2.imwrite(str(out_identity / aug_fname), aug_bgr)
            saved_aug += 1
            aug_records.append(
                {
                    "path": str(Path(row["path"]).parent / aug_fname),
                    "identity": row["identity"],
                }
            )

        if idx % 500 == 0 or idx == split_df.height:
            if split_name == "train":
                print(
                    f"  [{split_name:5s}] {idx:4d}/{split_df.height} — "
                    f"{saved_orig} orig + {saved_aug} aug guardadas"
                )
            else:
                print(f"  [{split_name:5s}] {idx:4d}/{split_df.height} guardadas")

    if split_name == "train":
        aligned_stats.append(
            {
                "split": split_name,
                "originales": saved_orig,
                "augmentadas": saved_aug,
                "omitidas": skipped,
            }
        )
        print(
            f"→ {split_name}: {saved_orig} orig + {saved_aug} aug guardadas, "
            f"{skipped} omitidas en {ALIGNED_DIR / split_name}"
        )
    else:
        aligned_stats.append(
            {
                "split": split_name,
                "originales": saved_orig,
                "augmentadas": 0,
                "omitidas": skipped,
            }
        )
        print(
            f"→ {split_name}: {saved_orig} imágenes guardadas, {skipped} omitidas "
            f"en {ALIGNED_DIR / split_name}"
        )

# Actualizamos train.csv para incluir las imágenes aumentadas
if aug_records:
    train_existing = pl.read_csv(OUTPUT_DIR / "train.csv")
    train_aug = pl.from_dicts(aug_records)
    train_combined = pl.concat([train_existing, train_aug])
    train_combined.write_csv(OUTPUT_DIR / "train.csv")
    print(
        f"\nTrain CSV actualizado: {train_existing.height} orig + {train_aug.height} aug "
        f"= {train_combined.height} filas totales → {OUTPUT_DIR / 'train.csv'}"
    )

display(pl.from_dicts(aligned_stats))

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
