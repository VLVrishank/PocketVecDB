"""
Quick walkthrough of PocketVecCore basics.

Run:
    python howtouse.py
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from pocketvec_core import PocketVecCore


def main() -> None:
    with TemporaryDirectory() as tmp:
        db_path = Path(tmp)
        print(f"Storing data in: {db_path}")

        core = PocketVecCore(str(db_path), dimension=4)

        # Add a couple of vectors
        core.add("apple", torch.tensor([0.1, 0.2, 0.3, 0.4]), {"tag": "fruit"})
        core.add("banana", torch.tensor([0.2, 0.1, 0.4, 0.8]), {"tag": "fruit"})
        core.add("car", torch.tensor([0.9, 0.1, 0.2, 0.1]), {"tag": "vehicle"})

        # Exact search
        query = torch.tensor([0.15, 0.15, 0.35, 0.5])
        results = core.search(query, k=2, mode="exact")
        print("Exact search:", results)

        # Update a record
        core.update("banana", metadata={"tag": "fruit", "status": "ripe"})
        print("Updated banana metadata:", core.metadata[core.id_to_index["banana"]])

        # Build indexes and try adaptive search
        core.rebuild_clusters(num_clusters=2)
        core.rebuild_windows(num_neighbors=1)
        adaptive_results = core.search(query, k=2, mode="adaptive")
        print("Adaptive search:", adaptive_results)

        # Explain a match
        explanation = core.explain_match(query, "banana", top_n_dims=3)
        print("Explain banana:", explanation)

        # Delete an item
        core.delete("car")
        print("After deleting car, remaining ids:", list(core.id_to_index.keys()))

        # Persist and reload
        core.save()
        print("Saved database to disk.")

        reloaded = PocketVecCore(str(db_path))
        reload_results = reloaded.search(query, k=2, mode="exact")
        print("Reloaded search:", reload_results)


if __name__ == "__main__":
    main()
