# TP1 - Sistema de Reconocimiento Facial

Sistema de detección, alineación, extracción de embeddings e identificación facial.

**Integrantes:** Alejandro Armas y Gabriel Soda.

> 📄 El enunciado completo del trabajo práctico está en [ENUNCIADO.md](ENUNCIADO.md).

## Objetivo del backend

La API asincrónica en Python permite:

- Registrar identidades (`/insert`)
- Ejecutar inferencia sobre una imagen (`/predict`)
- Consultar el estado del procesamiento asincrónico (`/status/{job_id}`)

La API responde `HTTP 202` con un `job_id` y luego permite consultar el resultado con estado:

```json
{
  "status": "done | inProgress | failed",
  "link": "url | none"
}
```

## Estructura

```text
tuia-face-recognition-app/
├── 1-pre-processing.ipynb       # Parte 1: dataset y preprocesamiento
├── 2-train.ipynb                # Parte 2: entrenamiento y comparación de modelos
├── src/
│   ├── app/
│   │   └── main.py              # App FastAPI + /health
│   ├── frontend/                # UI Gradio provista por la cátedra
│   │   ├── app.py
│   │   └── gradio_ui.py
│   └── lib/
│       ├── api.py               # Endpoints /upload, /insert, /predict, /status, /files
│       ├── config.py            # Configuración vía .env
│       ├── schemas.py
│       ├── services/
│       │   ├── face_service.py  # Detección, alineación y embeddings
│       │   └── task_manager.py
│       └── storage/
│           ├── base.py
│           ├── embedding_store.py   # Persistencia en JSON
│           └── pgvector_store.py    # Persistencia en PostgreSQL + pgvector
├── models/                      # Modelo entrenado (.pth / .onnx), no se versiona
├── output/                      # Imágenes subidas y resultados, no se versiona
├── requirements.txt
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.docker.example
└── .env.local.example
```

# Preparando el ambiente local

## Requisitos para trabajar de forma local

- Python 3.12
- Docker

## Configurá tu modelo

El servicio inicializa InsightFace con `buffalo_l`. El paquete incluye `det_10g.onnx` para detección con SCRFD y `w600k_r50.onnx` para reconocimiento con ArcFace.

`FaceAnalysis` busca el paquete en la caché de InsightFace e intenta descargarlo al arrancar si falta. El endpoint `/health` verifica por separado que exista el reconocedor en la ruta indicada por `MODEL_PATH` y `MODEL_NAME`. Descargá el [paquete oficial buffalo_l](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip) y usá las rutas de la opción de arranque elegida.

## Opción 1 - Corriendo dentro de Docker

### 1. Construí y corré la aplicación

Extraé `det_10g.onnx` y `w600k_r50.onnx` en `models/buffalo_l/` desde la raíz del repositorio. Compose monta esa carpeta y usa `.env.docker.example`. El primer arranque puede requerir acceso a Internet para llenar la caché de InsightFace dentro del contenedor. Desde la raíz, corré:

```bash
docker compose build
docker compose up -d
```

Comprobá que [`/health`](http://localhost:8000/health) devuelva `status: ok` y abrí [Gradio](http://localhost:8080).

## Opción 2 - Configurando el ambiente local

### 1. Instalá uv

Para usuarios de Linux o Mac:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para usuarios de Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

[Documentación de uv](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Configurá un ambiente virtual con Python 3.12

```bash
uv venv --python 3.12 .venv
```

### 3. Activá el ambiente virtual

```bash
source .venv/bin/activate
```

En PowerShell, ejecutá `.\.venv\Scripts\Activate.ps1`.

### 4. Instalá las dependencias

```bash
uv pip install -r requirements.txt
```

### 5. Iniciá la base de datos

```bash
docker compose up postgres -d
```

### 6. Iniciá el frontend

```bash
cd src
uvicorn frontend.app:app --port 8080
```

### 7. Iniciá el backend

Copiá `.env.local.example` a `src/.env`. Reemplazá las entradas `MODEL_NAME` del ejemplo por una sola y usá los mismos filtros que en Docker:

```dotenv
MODEL_NAME=buffalo_l/w600k_r50.onnx
DETECTOR_MODEL_NAME=buffalo_l/det_10g.onnx
DET_SCORE_THRESHOLD=0.6
DET_INPUT_SIZE=160
MIN_FACE_SIZE=56
```

Dejá los dos ONNX en `src/models/buffalo_l/`. Desde `src/`, en otra terminal con el entorno virtual activo:

```bash
uvicorn app.main:app --reload --port 8000
```

## Configuración

- `SIMILARITY_METRIC=cosine` y `SIMILARITY_THRESHOLD=0.2231` configuran la comparación y el rechazo como `unknown`.
- `FACE_SIZE=112` y `EMBEDDING_DIM=512` corresponden al recorte alineado y al embedding de ArcFace.
- `DET_SCORE_THRESHOLD=0.6`, `DET_INPUT_SIZE=160` y `MIN_FACE_SIZE=56` son los valores usados en Docker para la detección.
- Con `USE_PGVECTOR=true` los embeddings se guardan en PostgreSQL. Para usar almacenamiento JSON, configurá `USE_PGVECTOR=false` y `EMBEDDINGS_PATH`.

## Endpoints

- Backend: `http://localhost:8000` (documentación Swagger en `http://localhost:8000/docs`)
- PostgreSQL/pgvector: `localhost:5432`
- Frontend (provisto por la cátedra): `http://localhost:8080`

## Pipeline

### Provisto por la cátedra

- API asincrónica (`/upload`, `/insert`, `/predict`, `/status`) y manejo de jobs.
- Búsqueda por similitud configurable (`cosine` o `l2`) e identificación (`identify`).
- Manejo de desconocidos con `SIMILARITY_THRESHOLD`.
- Persistencia configurable en JSON o en PostgreSQL + pgvector (`USE_PGVECTOR`).
- Frontend para registrar identidades, predecir y visualizar los resultados (cajas, etiquetas y keypoints).

### Implementado por el grupo (`src/lib/services/face_service.py`)

El servicio inicializa `FaceAnalysis` una vez y usa sus detecciones, landmarks y embeddings en las tres funciones:

1. `detect_faces`: detección de rostros. Devuelve las cajas `(x1, y1, x2, y2)`.
2. `align_face`: alineación con landmarks y normalización a `FACE_SIZE`. Devuelve un `AlignedFace`.
3. `extract_embedding_from_face`: extracción del embedding de dimensión `EMBEDDING_DIM`, normalizado L2.

Ver detalles y recomendaciones en [ENUNCIADO.md](ENUNCIADO.md#4-parte-3--integración-en-la-aplicación).

## Modelo y fine-tuning

La [notebook de entrenamiento](2-train.ipynb) compara las CNN A y A2 y selecciona la CNN A por accuracy de validación. En test compara sus embeddings con los de ArcFace preentrenado de `buffalo_l`. ArcFace acertó el 99,82 % de las consultas de personas registradas y rechazó el 82,33 % de las consultas fuera de galería. La CNN A fue más rápida en CPU y ocupa menos espacio. Para la app se eligió ArcFace por su desempeño en identificación.

El umbral de ArcFace (`0.2231`) se obtuvo con 600 pares positivos y 600 negativos de validación, en el punto donde FAR y FRR quedan más cerca. La notebook incluye el gráfico de calibración y los resultados en test.

Los pesos de A y A2 de v2 están en la [carpeta de modelos del TP](https://drive.google.com/drive/folders/13bwx2SEOC7FJ4i0PY31Lfij1II7i3aYw?usp=sharing). Para cargarlos desde la notebook, colocá los dos archivos `*_best.pth` en `models/experimento_v2/` y ejecutá primero las definiciones de las clases. Los pesos de `buffalo_l` se ofrecen para investigación no comercial. [Términos de InsightFace](https://github.com/deepinsight/insightface/tree/master/python-package#license)

## Dataset

Usamos [Pins Face Recognition](https://www.kaggle.com/hereisburak/pins-face-recognition), una colección de fotos de celebridades cuya ficha declara CC0. La [notebook de preprocesamiento](1-pre-processing.ipynb) lee las imágenes de `data/images/`, explora el desbalance y la variación de pose e iluminación, aplica filtros de detección y tamaño, y revisa selecciones incorrectas del rostro principal.

Tras balancear, quedaron 100 fotos originales por cada una de 100 identidades. Se exportaron 14.000 recortes para entrenamiento, incluidos 7.000 aumentos, y 1.500 para cada uno de los conjuntos de validación y test. Los recortes alineados y manifiestos en `output/` son la entrada de la notebook de entrenamiento.

## Notas importantes

- Para usar `pgvector`, levanten `postgres` y definan `USE_PGVECTOR=true` en `.env`.
- Si cambian `EMBEDDING_DIM`, la tabla `embeddings` se recrea y se borran los registros previos.
- Los pesos (`*.pth`, `*.onnx`) y los datos se mantienen fuera de Git. Las descargas están indicadas en las secciones del modelo y del dataset.
