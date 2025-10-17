# PocketVec
<p align="center">
  <img width="532" height="297" alt="image" src="https://github.com/user-attachments/assets/14411e59-2dff-402c-ae36-6130760cb2e5" />
</p>

PocketVec is a clean, drop-in vector database engine for local projects. The core (`PocketVecCore`) runs entirely on your machine, keeps data in simple `.pt` + SQLite files, and ships with a FastAPI wrapper (`pocketvec_service.py`), smoke tests, and benchmarking utilities. Drop it into scripts, notebooks, or small services when you need vector search without managing heavy infrastructure.

## Features
- PyTorch backed vectors with automatic capacity growth.
- Energy-based adaptive search, clustering (KMeans) and neighbor windows (k-NN).
- Durable storage: tensors saved as `.pt`, metadata in SQLite.
- FastAPI service for CRUD, search, explain, and rebuild operations.
- Benchmarks covering speed vs accuracy, adaptive learning, and scalability.

## Repository Layout
- `pocketvec_core.py` - main core engine class.
- `pocketvec_service.py` - FastAPI application (`uvicorn pocketvec_service:app`).
- `howtouse.py` - minimal example script.

## Adaptive Search (Energy Driven)
PocketVec's adaptive search mode keeps track of how often each vector appears in top results. Every exact or clustered lookup slightly increases the "energy" score of the winning vectors. When you run an adaptive search, PocketVec:
1. Chooses a handful of starting points based on those energy scores.
2. Expands to their pre-built neighbor windows.
3. Runs an exact search over that smaller candidate set.
## Adaptive Learning in Action 
<p align="center">
<img width="1200" height="800" alt="adaptive_learning" src="https://github.com/user-attachments/assets/65ffe557-4c79-4c1b-87e7-4442cbf33d3a" /> </p>
The line starts near 3 ms and settles closer to 2 ms as the system repeatedly sees the same queries. That drop is the energy scores steering adaptive search toward the right neighborhood without scanning everything.


Hot items become faster to retrieve, while rarely used items keep their baseline probability. This makes adaptive search a good default for workloads with repeating or trending queries - latency drops automatically as the system "learns".

## Quick Start
1. Install dependencies (PyTorch, scikit-learn, FastAPI, uvicorn, matplotlib, optional psutil).
4. Explore the example script: `python howtouse.py`.
3. Start the API: `uvicorn pocketvec_service:app --reload` if needed.


