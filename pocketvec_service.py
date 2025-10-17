"""PocketVec FastAPI service exposing PocketVecCore operations."""

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from pocketvec_core import PocketVecCore


class VectorPayload(BaseModel):
    item_id: str = Field(..., alias="id")
    vector: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("vector")
    def validate_vector(cls, value: List[float]) -> List[float]:
        if not value:
            raise ValueError("Vector must not be empty")
        return value


class SearchPayload(BaseModel):
    vector: List[float]
    k: int = Field(10, gt=0, le=1024)
    mode: str = Field("exact", pattern="^(exact|clustered|adaptive)$")
    clusters_to_probe: int = Field(2, gt=0, le=64)

    @field_validator("vector")
    def validate_vector(cls, value: List[float]) -> List[float]:
        if not value:
            raise ValueError("Vector must not be empty")
        return value


class RebuildClustersPayload(BaseModel):
    num_clusters: int = Field(..., gt=0, le=1024)


class RebuildWindowsPayload(BaseModel):
    num_neighbors: int = Field(..., gt=0, le=1024)


class ExplainPayload(BaseModel):
    vector: List[float]
    top_n_dims: int = Field(5, gt=0, le=1024)

    @field_validator("vector")
    def validate_vector(cls, value: List[float]) -> List[float]:
        if not value:
            raise ValueError("Vector must not be empty")
        return value


def _ensure_directory(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def _load_config(db_path: Optional[str], dimension: Optional[int]) -> Dict[str, Any]:
    env_path = os.getenv("POCKETVEC_DB_PATH", db_path or "./pocketvec_db")
    env_dim = os.getenv("POCKETVEC_DIMENSION")
    resolved_dim: Optional[int] = dimension
    if resolved_dim is None and env_dim:
        resolved_dim = int(env_dim)
    if resolved_dim is None:
        resolved_dim = 128
    return {"db_path": Path(env_path), "dimension": resolved_dim}


def create_app(db_path: Optional[str] = None, dimension: Optional[int] = None) -> FastAPI:
    config = _load_config(db_path, dimension)
    data_dir: Path = config["db_path"]
    dimension = config["dimension"]
    _ensure_directory(data_dir)

    db = PocketVecCore(str(data_dir), dimension=dimension)
    lock = threading.Lock()

    app = FastAPI(title="PocketVec", version="1.0.0")
    app.state.db = db
    app.state.lock = lock
    app.state.data_dir = data_dir

    def get_db() -> PocketVecCore:
        return app.state.db

    def get_lock() -> threading.Lock:
        return app.state.lock

    @app.get("/health", tags=["system"])
    def healthcheck() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/stats", tags=["system"])
    def stats(db: PocketVecCore = Depends(get_db)) -> Dict[str, Any]:
        return {
            "dimension": db.dimension,
            "count": db.count,
            "capacity": db.capacity,
            "has_centroids": db.centroids is not None,
            "has_windows": db._windows_ready(),
            "device": str(db.device),
        }

    @app.post(
        "/vectors",
        status_code=status.HTTP_201_CREATED,
        tags=["vectors"],
    )
    def add_vector(payload: VectorPayload, db: PocketVecCore = Depends(get_db), lock: threading.Lock = Depends(get_lock)) -> Dict[str, Any]:
        vector_tensor = torch.tensor(payload.vector, dtype=torch.float32, device=db.device)
        with lock:
            db.add(payload.item_id, vector_tensor, payload.metadata)
        return {"id": payload.item_id}

    @app.delete("/vectors/{item_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["vectors"])
    def delete_vector(item_id: str, db: PocketVecCore = Depends(get_db), lock: threading.Lock = Depends(get_lock)) -> None:
        with lock:
            if item_id not in db.id_to_index:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")
            db.delete(item_id)

    @app.post("/search", tags=["search"])
    def search(payload: SearchPayload, db: PocketVecCore = Depends(get_db)) -> Dict[str, Any]:
        vector_tensor = torch.tensor(payload.vector, dtype=torch.float32, device=db.device)
        results = db.search(
            vector_tensor,
            k=payload.k,
            mode=payload.mode,
            clusters_to_probe=payload.clusters_to_probe,
        )
        return {"results": results}

    @app.post("/vectors/{item_id}/explain", tags=["search"])
    def explain(item_id: str, payload: ExplainPayload, db: PocketVecCore = Depends(get_db)) -> Dict[str, Any]:
        vector_tensor = torch.tensor(payload.vector, dtype=torch.float32, device=db.device)
        explanation = db.explain_match(vector_tensor, item_id, top_n_dims=payload.top_n_dims)
        return {"id": item_id, "explanation": explanation}

    @app.post("/rebuild/clusters", tags=["index"])
    def rebuild_clusters(payload: RebuildClustersPayload, db: PocketVecCore = Depends(get_db), lock: threading.Lock = Depends(get_lock)) -> Dict[str, Any]:
        with lock:
            if db.count == 0:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Database is empty")
            db.rebuild_clusters(payload.num_clusters)
        return {"status": "clusters_rebuilt", "num_clusters": payload.num_clusters}

    @app.post("/rebuild/windows", tags=["index"])
    def rebuild_windows(payload: RebuildWindowsPayload, db: PocketVecCore = Depends(get_db), lock: threading.Lock = Depends(get_lock)) -> Dict[str, Any]:
        with lock:
            if db.count == 0:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Database is empty")
            db.rebuild_windows(payload.num_neighbors)
        return {"status": "windows_rebuilt", "num_neighbors": payload.num_neighbors}

    @app.post("/save", tags=["system"])
    def save(db: PocketVecCore = Depends(get_db), lock: threading.Lock = Depends(get_lock)) -> Dict[str, Any]:
        with lock:
            db.save()
        return {"status": "saved", "path": str(app.state.data_dir)}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
