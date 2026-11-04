from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch
import onnxruntime
from lib.schemas import EmbeddingRecord, FaceDetection, PredictResult, AlignedFace
from lib.storage.base import EmbeddingStoreProtocol
import os 
import logging
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


class FaceService:
    def __init__(
        self,
        store: EmbeddingStoreProtocol,
        similarity_metric: str,
        similarity_threshold: float,
        face_size: int,
        model_path: Path,
        output_path: Path = Path("output"),
        detector_path: Path | None = None,
        det_score_threshold: float = 0.5,
        det_input_size: int = 640,
        min_face_size: int = 20,
        align_margin: float = 0.5,
    ) -> None:
        self.store = store
        self.similarity_metric = similarity_metric
        self.similarity_threshold = similarity_threshold
        self.face_size = face_size
        self.output_path = output_path
        self.det_score_threshold = det_score_threshold
        self.det_input_size = det_input_size
        self.min_face_size = min_face_size
        self.align_margin = align_margin
        self._last_face_objs: dict = {}

        # ------------------------------------------------------------------
        # Usar modelos de InsightFace (buffalo_l) directamente
        # ------------------------------------------------------------------
        self._face_analysis = FaceAnalysis(
                        name="buffalo_l",
                        allowed_modules=["detection", "recognition"],
                    )
        ctx = 0 if torch.cuda.is_available() else -1
        det_size = (det_input_size, det_input_size)
        self._face_analysis.prepare(ctx_id=ctx, det_size=det_size)
        logger.info(
            "InsightFace buffalo_l loaded, det_size=%s", det_size
        )

        self.model = self._face_analysis.models.get("recognition")
        self.detector = self._face_analysis.models.get("detection")

    @staticmethod
    def _clip_xyxy(
        x1: int, y1: int, x2: int, y2: int, height: int, width: int
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1:
            x2 = min(x1 + 1, width)
        if y2 <= y1:
            y2 = min(y1 + 1, height)
        return x1, y1, x2, y2

    @staticmethod
    def _kps_to_keypoints_dict(kps: np.ndarray | None) -> dict[str, list[int]]:
        if kps is None or len(kps) == 0:
            return {}
        return {
            f"k{i}": [int(round(float(kps[i, 0]))), int(round(float(kps[i, 1])))]
            for i in range(len(kps))
        }


    def _load_model(self, model_path: Path) -> any:
        mp = Path(model_path)
        if not mp.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        suf = mp.suffix.lower()
        if suf == ".pth":
            return torch.load(mp, map_location="cpu", weights_only=False)
        if suf == ".onnx":
            return onnxruntime.InferenceSession(str(mp))
        raise ValueError(f"Unsupported model format (expected .pth or .onnx): {model_path}")

    def _load_detector(self, detector_path: Path | None) -> any:
        """
        Load the face detector once (called from __init__).
        By default loads a .pth / .onnx file from DETECTOR_MODEL_NAME, or returns None
        when it is not set. Override it if your detector comes from a library
        (InsightFace, OpenCV YuNet, MTCNN, etc.) and configure it with
        self.det_score_threshold and self.det_input_size.
        """
        if detector_path is None:
            return None
        return self._load_model(detector_path)

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(source_path)
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (InsightFace / OpenCV convention)
        return image
    
    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        """Intersection over union between two boxes (x1, y1, x2, y2)."""
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return float(inter / union) if union > 0 else 0.0
    
    def detect_faces(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detecta rostros con InsightFace FaceAnalysis.
        Cada caja es (x1, y1, x2, y2) en píxeles enteros.
        Filtra por score de detección y tamaño mínimo.
        """
        if self._face_analysis is None:
            raise NotImplementedError("No hay detector configurado")

        faces = self._face_analysis.get(image)
        self._last_face_objs: dict[tuple[int, int, int, int], any] = {}
        result: list[tuple[int, int, int, int]] = []
        for face in faces:
            if face.det_score < self.det_score_threshold:
                continue
            x1, y1, x2, y2 = map(int, face.bbox)
            h = y2 - y1
            w = x2 - x1
            if min(h, w) < self.min_face_size:
                continue
            x1, y1, x2, y2 = self._clip_xyxy(
                x1, y1, x2, y2, image.shape[0], image.shape[1]
            )
            clipped_box = (x1, y1, x2, y2)
            self._last_face_objs[clipped_box] = face
            result.append(clipped_box)
        return result

    def align_face(
        self, image: np.ndarray, box: tuple[int, int, int, int]
    ) -> AlignedFace:
        """
        Alinea el rostro usando los 5 landmarks de InsightFace y norm_crop.
        Guarda keypoints relativos a la esquina superior izquierda de la caja.
        El embedding se precalcula durante get() y se reutiliza.
        """
        if self._face_analysis is None:
            raise NotImplementedError("No hay detector/alineador configurado")

        # Buscar la detección cacheada que coincida con la caja
        face_obj = self._last_face_objs.get(box)

        # Fallback: detectar de nuevo si no está en cache
        if face_obj is None:
            faces = self._face_analysis.get(image)
            for face in faces:
                fx1, fy1, fx2, fy2 = map(int, face.bbox)
                fx1, fy1, fx2, fy2 = self._clip_xyxy(
                    fx1, fy1, fx2, fy2, image.shape[0], image.shape[1]
                )
                if (fx1, fy1, fx2, fy2) == box:
                    face_obj = face
                    break
            if face_obj is None:
                raise ValueError("No se encontró la detección correspondiente a la caja")

        from insightface.utils.face_align import norm_crop

        aligned_img = norm_crop(
            image, face_obj.kps, image_size=self.face_size, mode="arcface"
        )

        # Keypoints relativos a la esquina superior izquierda de la caja
        kps_rel = face_obj.kps - np.array([box[0], box[1]], dtype=np.float32)

        # Embedding ya calculado por FaceAnalysis; normalizamos L2 por seguridad
        emb = face_obj.embedding.flatten().astype(np.float32)
        emb_norm = np.linalg.norm(emb)
        if emb_norm > 0:
            emb = emb / emb_norm

        return AlignedFace(
            bbox=box,
            keypoints=kps_rel,
            image=aligned_img,
            embedding=emb.astype(float).tolist(),
        )

    def extract_embedding_from_face(self, face: AlignedFace) -> list[float]:
        """
        Devuelve el embedding precalculado durante la alineación.
        Re-normaliza L2 antes de devolverlo.
        """
        if face.embedding is None:
            raise ValueError("AlignedFace no tiene embedding precalculado")
        emb = np.asarray(face.embedding, dtype=np.float32)
        emb_norm = np.linalg.norm(emb)
        if emb_norm > 0:
            emb = emb / emb_norm
        return emb.astype(float).tolist()
        
    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _l2_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)

    def similarity(self, query: list[float], ref: list[float]) -> float:
        a = np.asarray(query, dtype=np.float32)
        b = np.asarray(ref, dtype=np.float32)
        if self.similarity_metric.lower() == "l2":
            return self._l2_similarity(a, b)
        return self._cosine(a, b)

    def identify(self, query_embedding: list[float]) -> tuple[str, float]:
        records = self.store.all()
        if not records:
            return "unknown", 0.0

        best_label = "unknown"
        best_score = -1.0
        for record in records:
            score = self.similarity(query_embedding, record.embedding)
            if score > best_score:
                best_score = score
                best_label = record.etiqueta

        if best_score < self.similarity_threshold:
            return "unknown", max(best_score, 0.0)
        return best_label, best_score

    def register_identity(
        self, identity: str, image_path: str, metadata: dict[str, object]
    ) -> EmbeddingRecord:
        image = self._load_image(image_path)
        faces = self.detect_faces(image)

        if len(faces) != 1:
            raise ValueError("Exactly one face must be detected for identity registration.")
        
        logger.info(f"Face detected: {faces[0]}")

        box = faces[0]
        aligned = self.align_face(image, box)
        embedding = self.extract_embedding_from_face(aligned)

        img_id = str(uuid4())
        img_output_path = self.output_path / f"img_{img_id}.jpg"
        
        record = EmbeddingRecord(
            id_imagen=str(uuid4()),
            embedding=embedding,
            path=str(img_output_path),
            etiqueta=identity,
            metadata=metadata,
        )
        self.store.append(record)

        cv2.imwrite(str(img_output_path), aligned.image)
        logger.info(f"Identity registered: {identity} with image: {image_path}")
        return record

    def predict(self, source_path: str, output_path: Path) -> str:
        image = self._load_image(source_path)
        faces = self.detect_faces(image)
        detections: list[FaceDetection] = []
        for (x1, y1, x2, y2) in faces:
            aligned = self.align_face(image, (x1, y1, x2, y2))
            embedding = self.extract_embedding_from_face(aligned)
            label, score = self.identify(embedding)
            kps = getattr(aligned, "keypoints", None)
            kps_arr = np.asarray(kps) if kps is not None else None
            detections.append(
                FaceDetection(
                    bbox=[x1, y1, x2, y2],
                    keypoints=self._kps_to_keypoints_dict(kps_arr),
                    label=label,
                    score=round(float(score), 4),
                )
            )

        detected_people = sorted({item.label for item in detections if item.label != "unknown"})
        result_payload = PredictResult(
            source_path=source_path,
            detections=detections,
            detected_people=detected_people,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(result_payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)
