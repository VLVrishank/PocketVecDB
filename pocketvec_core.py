import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


class PocketVecCore:
    def __init__(self, db_path: str, dimension: int | None = None) -> None:
        self.db_path = Path(db_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dimension: int = 0
        self.capacity: int = 0
        self.count: int = 0

        self.vectors = torch.empty((0, 0), dtype=torch.float32, device=self.device)
        self.energy = torch.empty((0,), dtype=torch.float32, device=self.device)

        self.metadata: List[Optional[dict]] = []
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        self.free_indices: List[int] = []
        self.windows: List[List[int]] = []
        self.centroids: Optional[torch.Tensor] = None
        self.labels: Optional[List[int]] = None
        self._windows_valid = False
        self._energy_floor = 1e-6

        if self.db_path.exists():
            try:
                self._load_existing()
                if dimension is not None and dimension != self.dimension:
                    raise ValueError(
                        f"Stored dimension {self.dimension} does not match requested {dimension}"
                    )
            except FileNotFoundError:
                if dimension is None:
                    raise ValueError("Dimension must be set when creating a new database")
                self._start_new(dimension)
        else:
            if dimension is None:
                raise ValueError("Dimension must be set when creating a new database")
            self._start_new(dimension)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        self.db_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.vectors.detach().cpu(), self.db_path / "vectors.pt")
        torch.save(self.energy.detach().cpu(), self.db_path / "energy.pt")
        self._write_sql_state()

    def _load_existing(self) -> None:
        vectors_path = self.db_path / "vectors.pt"
        energy_path = self.db_path / "energy.pt"
        sqlite_path = self.db_path / "database.sqlite"

        if not vectors_path.exists() or not energy_path.exists() or not sqlite_path.exists():
            raise FileNotFoundError("Database files are missing. Cannot load existing store.")

        self.vectors = torch.load(vectors_path, map_location=self.device).to(
            self.device, dtype=torch.float32
        )
        self.energy = torch.load(energy_path, map_location=self.device).to(
            self.device, dtype=torch.float32
        )
        if self.vectors.ndim != 2:
            raise ValueError("Stored vectors must be 2D")
        if self.energy.ndim != 1:
            raise ValueError("Stored energy must be 1D")

        self._read_sql_state(sqlite_path)
        self._refresh_after_load()

    def _write_sql_state(self) -> None:
        db_file = self.db_path / "database.sqlite"
        conn = sqlite3.connect(db_file)
        try:
            with conn:
                conn.executescript(
                    """
                    DROP TABLE IF EXISTS metadata;
                    DROP TABLE IF EXISTS free_indices;
                    DROP TABLE IF EXISTS windows;
                    DROP TABLE IF EXISTS labels;
                    DROP TABLE IF EXISTS centroids;
                    DROP TABLE IF EXISTS state;
                    """
                )
                conn.executescript(
                    """
                    CREATE TABLE metadata (
                        idx INTEGER PRIMARY KEY,
                        item_id TEXT UNIQUE,
                        metadata_json TEXT
                    );
                    CREATE TABLE free_indices (
                        idx INTEGER PRIMARY KEY
                    );
                    CREATE TABLE windows (
                        source_idx INTEGER NOT NULL,
                        neighbor_idx INTEGER NOT NULL,
                        PRIMARY KEY (source_idx, neighbor_idx)
                    );
                    CREATE TABLE labels (
                        idx INTEGER PRIMARY KEY,
                        label INTEGER NOT NULL
                    );
                    CREATE TABLE centroids (
                        cluster_idx INTEGER PRIMARY KEY,
                        vector_json TEXT NOT NULL
                    );
                    CREATE TABLE state (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    """
                )

                if self.count:
                    meta_rows = []
                    for idx in range(self.count):
                        item_id = self.index_to_id.get(idx)
                        meta = self.metadata[idx] if idx < len(self.metadata) else None
                        meta_rows.append((idx, item_id, json.dumps(meta) if meta is not None else None))
                    conn.executemany(
                        "INSERT INTO metadata (idx, item_id, metadata_json) VALUES (?, ?, ?)",
                        meta_rows,
                    )

                if self.free_indices:
                    conn.executemany(
                        "INSERT INTO free_indices (idx) VALUES (?)",
                        [(int(idx),) for idx in self.free_indices],
                    )

                if self.windows:
                    window_rows = []
                    for src_idx, neighbors in enumerate(self.windows[: self.count]):
                        for nbr in neighbors:
                            window_rows.append((int(src_idx), int(nbr)))
                    if window_rows:
                        conn.executemany(
                            "INSERT INTO windows (source_idx, neighbor_idx) VALUES (?, ?)",
                            window_rows,
                        )

                if self.labels:
                    label_rows = [
                        (int(idx), int(label))
                        for idx, label in enumerate(self.labels[: self.count])
                        if label >= 0
                    ]
                    if label_rows:
                        conn.executemany(
                            "INSERT INTO labels (idx, label) VALUES (?, ?)", label_rows
                        )

                if self.centroids is not None:
                    centroid_rows = [
                        (int(idx), json.dumps(vec))
                        for idx, vec in enumerate(self.centroids.detach().cpu().tolist())
                    ]
                    if centroid_rows:
                        conn.executemany(
                            "INSERT INTO centroids (cluster_idx, vector_json) VALUES (?, ?)",
                            centroid_rows,
                        )

                state_rows = [
                    ("dimension", str(self.dimension)),
                    ("capacity", str(self.capacity)),
                    ("count", str(self.count)),
                    ("windows_valid", "1" if self._windows_valid else "0"),
                ]
                conn.executemany(
                    "INSERT INTO state (key, value) VALUES (?, ?)",
                    state_rows,
                )
        finally:
            conn.close()

    def _read_sql_state(self, sqlite_path: Path) -> None:
        conn = sqlite3.connect(sqlite_path)
        try:
            state = dict(conn.execute("SELECT key, value FROM state"))
            if "dimension" not in state:
                raise ValueError("SQLite state is missing the required dimension field")

            self.dimension = int(state["dimension"])
            self.capacity = int(state.get("capacity", self.vectors.size(0)))
            self.count = int(state.get("count", self.capacity))
            self._windows_valid = state.get("windows_valid", "0") == "1"

            if self.capacity != self.vectors.size(0):
                raise ValueError("Stored capacity does not match vectors tensor size")
            if self.energy.size(0) != self.capacity:
                raise ValueError("Energy tensor size mismatch")
            if self.count > self.capacity:
                raise ValueError("Active item count cannot exceed capacity")

            self.metadata = [None] * self.count
            self.id_to_index = {}
            self.index_to_id = {}

            for idx, item_id, payload in conn.execute(
                "SELECT idx, item_id, metadata_json FROM metadata"
            ):
                idx = int(idx)
                if idx >= self.count:
                    continue
                if payload is not None:
                    self.metadata[idx] = json.loads(payload)
                if item_id is not None:
                    item = str(item_id)
                    self.id_to_index[item] = idx
                    self.index_to_id[idx] = item

            self.free_indices = [
                int(row[0])
                for row in conn.execute("SELECT idx FROM free_indices ORDER BY idx")
                if int(row[0]) < self.capacity
            ]

            self.windows = [[] for _ in range(self.count)]
            for src_idx, nbr_idx in conn.execute(
                "SELECT source_idx, neighbor_idx FROM windows ORDER BY source_idx, neighbor_idx"
            ):
                src_idx = int(src_idx)
                nbr_idx = int(nbr_idx)
                if 0 <= src_idx < self.count:
                    self.windows[src_idx].append(nbr_idx)

            label_rows = list(conn.execute("SELECT idx, label FROM labels"))
            if label_rows:
                labels = [-1] * self.count
                has_valid_label = False
                for idx, label in label_rows:
                    idx = int(idx)
                    if 0 <= idx < self.count:
                        labels[idx] = int(label)
                        if int(label) >= 0:
                            has_valid_label = True
                self.labels = labels if has_valid_label else None
            else:
                self.labels = None

            centroid_rows = list(
                conn.execute(
                    "SELECT cluster_idx, vector_json FROM centroids ORDER BY cluster_idx"
                )
            )
            if centroid_rows:
                vectors = [json.loads(vec_json) for _, vec_json in centroid_rows]
                self.centroids = torch.tensor(
                    vectors, dtype=torch.float32, device=self.device
                )
            else:
                self.centroids = None
        finally:
            conn.close()

    def _refresh_after_load(self) -> None:
        if self.vectors.size(1) != self.dimension:
            raise ValueError("Stored vectors do not match saved dimension")
        if len(self.metadata) < self.count:
            self.metadata.extend([None] * (self.count - len(self.metadata)))
        elif len(self.metadata) > self.count:
            self.metadata = self.metadata[: self.count]

        if len(self.windows) < self.count:
            self.windows.extend([[] for _ in range(self.count - len(self.windows))])
        elif len(self.windows) > self.count:
            self.windows = self.windows[: self.count]

        if self.labels is not None:
            if len(self.labels) < self.count:
                self.labels.extend([-1] * (self.count - len(self.labels)))
            elif len(self.labels) > self.count:
                self.labels = self.labels[: self.count]

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _start_new(self, dimension: int) -> None:
        self.dimension = int(dimension)
        self.capacity = 0
        self.count = 0
        self.vectors = torch.empty((0, self.dimension), dtype=torch.float32, device=self.device)
        self.energy = torch.empty((0,), dtype=torch.float32, device=self.device)
        self.metadata = []
        self.id_to_index = {}
        self.index_to_id = {}
        self.free_indices = []
        self.windows = []
        self.centroids = None
        self.labels = None
        self._windows_valid = False

    def _grow_capacity(self) -> None:
        old_capacity = self.capacity
        new_capacity = max(16, old_capacity * 2 if old_capacity else 16)

        new_vectors = torch.zeros(
            (new_capacity, self.dimension), dtype=torch.float32, device=self.device
        )
        new_energy = torch.full(
            (new_capacity,), self._initial_energy_value(), dtype=torch.float32, device=self.device
        )

        if old_capacity:
            new_vectors[:old_capacity] = self.vectors[:old_capacity]
            new_energy[:old_capacity] = self.energy[:old_capacity]

        self.vectors = new_vectors
        self.energy = new_energy
        self.capacity = new_capacity

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add(self, item_id: str, vector, metadata: dict) -> None:
        tensor = self._to_vector(vector)

        if item_id in self.id_to_index:
            slot = self.id_to_index[item_id]
            self.vectors[slot] = tensor
            self.metadata[slot] = metadata
            self._invalidate_indexes()
            return

        if self.free_indices:
            slot = self.free_indices.pop()
        else:
            if self.count >= self.capacity:
                self._grow_capacity()
            slot = self.count
            self.count += 1

        if slot >= len(self.metadata):
            self.metadata.extend([None] * (slot - len(self.metadata) + 1))
        if slot >= len(self.windows):
            self.windows.extend([[] for _ in range(slot - len(self.windows) + 1)])

        self.vectors[slot] = tensor
        self.energy[slot] = self._initial_energy_value()
        self.metadata[slot] = metadata
        self.windows[slot] = []
        self.id_to_index[item_id] = slot
        self.index_to_id[slot] = item_id
        self._invalidate_indexes()

    def update(self, item_id: str, vector=None, metadata: Optional[dict] = None) -> None:
        """Update an existing item without changing its slot."""
        if item_id not in self.id_to_index:
            raise KeyError(f"Item '{item_id}' does not exist")

        slot = self.id_to_index[item_id]
        if vector is not None:
            self.vectors[slot] = self._to_vector(vector)
        if metadata is not None:
            self.metadata[slot] = metadata
        self._invalidate_indexes()

    def delete(self, item_id: str) -> None:
        if item_id not in self.id_to_index:
            raise KeyError(f"Item '{item_id}' does not exist")

        slot = self.id_to_index.pop(item_id)
        self.index_to_id.pop(slot, None)
        self.metadata[slot] = None
        self.windows[slot] = []
        self.vectors[slot] = 0.0
        self.energy[slot] = self._initial_energy_value()
        self.free_indices.append(slot)
        self._invalidate_indexes()

    # ------------------------------------------------------------------
    # Index builders
    # ------------------------------------------------------------------
    def rebuild_clusters(self, num_clusters: int) -> None:
        active = self._active_indices()
        if not active:
            raise ValueError("No data to cluster")
        if num_clusters <= 0 or num_clusters > len(active):
            raise ValueError("Invalid number of clusters")

        data = self.vectors[active].detach().cpu().numpy()
        model = KMeans(n_clusters=num_clusters, n_init=10)
        model.fit(data)

        labels = [-1] * self.count
        for idx, label in zip(active, model.labels_):
            labels[idx] = int(label)

        self.centroids = torch.tensor(model.cluster_centers_, dtype=torch.float32, device=self.device)
        self.labels = labels

    def rebuild_windows(self, num_neighbors: int) -> None:
        active = self._active_indices()
        if not active:
            raise ValueError("No data to index")
        if num_neighbors <= 0:
            raise ValueError("Number of neighbors must be positive")
        if len(active) == 1:
            self.windows = [[] for _ in range(self.count)]
            self._windows_valid = True
            return

        data = self.vectors[active].detach().cpu().numpy()
        k = min(num_neighbors + 1, len(active))
        if k >= len(active):
            k = len(active) - 1
        k = max(1, k)

        model = NearestNeighbors(n_neighbors=k, metric="euclidean")
        model.fit(data)
        neighbor_idx = model.kneighbors(return_distance=False)

        windows = [[] for _ in range(self.count)]
        for row, neighbors in enumerate(neighbor_idx):
            source = active[row]
            neighbor_list = []
            for nbr in neighbors:
                candidate = active[nbr]
                if candidate != source:
                    neighbor_list.append(candidate)
            windows[source] = neighbor_list[:num_neighbors]

        self.windows = windows
        self._windows_valid = True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, vector, k: int = 10, mode: str = "exact", **extras) -> List[Dict[str, object]]:
        if k <= 0:
            raise ValueError("k must be positive")
        mode = mode.lower()
        vector_tensor = self._to_vector(vector)

        if mode == "exact":
            return self._run_exact(vector_tensor, k)
        if mode == "clustered":
            return self._run_clustered(vector_tensor, k, extras.get("clusters_to_probe", 2))
        if mode == "adaptive":
            return self._run_adaptive(vector_tensor, k)
        raise ValueError(f"Unknown search mode '{mode}'")

    def _run_exact(self, query: torch.Tensor, k: int, candidates: Optional[Sequence[int]] = None) -> List[Dict[str, object]]:
        indices = list(candidates) if candidates is not None else self._active_indices()
        if not indices:
            return []

        scores = self._cosine_scores(self.vectors[indices], query)
        if scores.numel() == 0:
            return []

        top_k = min(k, scores.size(0))
        top_scores, top_pos = torch.topk(scores, k=top_k)
        chosen = [indices[int(pos)] for pos in top_pos.tolist()]

        self._bump_energy(chosen)
        return [self._format_result(idx, float(score)) for idx, score in zip(chosen, top_scores.tolist())]

    def _run_clustered(self, query: torch.Tensor, k: int, clusters_to_probe: int) -> List[Dict[str, object]]:
        if self.centroids is None or self.labels is None:
            return self._run_exact(query, k)
        if clusters_to_probe <= 0:
            raise ValueError("clusters_to_probe must be positive")

        distances = torch.cdist(query.unsqueeze(0), self.centroids).squeeze(0)
        top_clusters = torch.topk(distances, k=min(clusters_to_probe, distances.numel()), largest=False).indices.tolist()
        target = {int(idx) for idx in top_clusters}

        candidates = [idx for idx in self._active_indices() if self.labels[idx] in target]
        if not candidates:
            return self._run_exact(query, k)
        return self._run_exact(query, k, candidates=candidates)

    def _run_adaptive(self, query: torch.Tensor, k: int) -> List[Dict[str, object]]:
        active = self._active_indices()
        if not active:
            return []
        if not self._windows_ready():
            return self._run_exact(query, k)

        active_tensor = torch.tensor(active, dtype=torch.long, device=self.device)
        seed_count = min(20, active_tensor.numel())

        energy_vals = self.energy[active_tensor]
        total = float(energy_vals.sum().item())
        if total <= 0:
            weights = torch.full((active_tensor.numel(),), 1.0 / active_tensor.numel(), device=self.device)
        else:
            weights = (energy_vals / total).detach()
        weights = (weights / weights.sum()).cpu()

        seeds = active_tensor[
            torch.multinomial(weights, num_samples=seed_count, replacement=False)
        ].tolist()

        candidates = set(seeds)
        for seed in list(candidates):
            for neighbor in self.windows[seed]:
                if neighbor in self.index_to_id:
                    candidates.add(neighbor)

        return self._run_exact(query, k, candidates=sorted(candidates))

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------
    def explain_match(self, vector, item_id: str, top_n_dims: int = 5) -> Dict[int, float]:
        if item_id not in self.id_to_index:
            raise KeyError(f"Item '{item_id}' does not exist")
        if top_n_dims <= 0:
            raise ValueError("top_n_dims must be positive")

        query = self._to_vector(vector)
        slot = self.id_to_index[item_id]
        contribution = query * self.vectors[slot]
        size = min(top_n_dims, contribution.numel())
        values, indices = torch.topk(contribution, k=size)
        return {int(idx): float(val) for idx, val in zip(indices.tolist(), values.tolist())}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_vector(self, vector) -> torch.Tensor:
        if isinstance(vector, torch.Tensor):
            tensor = vector.detach().to(self.device, dtype=torch.float32)
        else:
            tensor = torch.tensor(vector, dtype=torch.float32, device=self.device)
        tensor = tensor.view(-1)
        if tensor.numel() != self.dimension:
            raise ValueError(f"Vector must have length {self.dimension}")
        return tensor

    def _active_indices(self) -> List[int]:
        return sorted(idx for idx in self.index_to_id.keys())

    def _cosine_scores(self, matrix: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        if matrix.numel() == 0:
            return torch.empty((0,), dtype=torch.float32, device=self.device)
        matrix_norm = torch.nn.functional.normalize(matrix, dim=1, eps=1e-12)
        query_norm = torch.nn.functional.normalize(query.unsqueeze(0), dim=1, eps=1e-12).squeeze(0)
        return torch.matmul(matrix_norm, query_norm)

    def _bump_energy(self, indices: Sequence[int]) -> None:
        if not indices:
            return
        self.energy.mul_(0.99)
        add = torch.ones(len(indices), dtype=torch.float32, device=self.device)
        positions = torch.tensor(list(indices), dtype=torch.long, device=self.device)
        self.energy.index_add_(0, positions, add)
        self.energy.clamp_(min=self._energy_floor)

    def _initial_energy_value(self) -> float:
        return 1.0

    def _invalidate_indexes(self) -> None:
        self.centroids = None
        self.labels = None
        self._windows_valid = False
        if len(self.windows) < self.count:
            self.windows.extend([[] for _ in range(self.count - len(self.windows))])
        else:
            for idx in range(self.count):
                self.windows[idx] = []

    def _windows_ready(self) -> bool:
        return self._windows_valid and len(self.windows) == self.count and self.count > 0

    def _format_result(self, index: int, score: float) -> Dict[str, object]:
        item_id = self.index_to_id.get(index)
        meta = self.metadata[index] if index < len(self.metadata) else None
        return {"id": item_id, "metadata": meta, "score": score}
