# scripts/compute_kmeans_clusters.py
import argparse, json, csv
from pathlib import Path
import numpy as np

def load_paper_ids(metadata_csv: Path):
    ids = []
    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            # metadata.csv에 paper_id 컬럼이 있다고 가정
            ids.append(row["paper_id"])
    return ids

def l2_normalize(x: np.ndarray, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

def kmeans_cosine(X: np.ndarray, k: int, seed: int = 42, max_iter: int = 50):
    """
    Cosine k-means (단위벡터로 normalize 후, center와의 dot(sim) 최대화로 할당)
    """
    rng = np.random.default_rng(seed)
    X = X.astype(np.float32)
    X = l2_normalize(X)

    n = X.shape[0]
    if k <= 1 or k > n:
        raise ValueError(f"k must be in [2, {n}]")

    # init centers: random points
    idx = rng.choice(n, size=k, replace=False)
    C = X[idx].copy()

    labels = np.full(n, -1, dtype=np.int32)

    for _ in range(max_iter):
        # assign: maximize cosine similarity => argmax(X @ C.T)
        S = X @ C.T  # (n, k)
        new_labels = S.argmax(axis=1).astype(np.int32)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # update centers
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                # empty cluster -> reinit
                C[j] = X[rng.integers(0, n)]
            else:
                C[j] = X[mask].mean(axis=0)
        C = l2_normalize(C)

    return labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True, type=Path)     # data/embeddings_Neu/embeddings.npy
    ap.add_argument("--metadata", required=True, type=Path)       # data/embeddings_Neu/metadata.csv
    ap.add_argument("--out_dir", required=True, type=Path)        # output/graph
    ap.add_argument("--k_min", type=int, default=2)
    ap.add_argument("--k_max", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.embeddings)
    paper_ids = load_paper_ids(args.metadata)

    if len(paper_ids) != X.shape[0]:
        raise RuntimeError(f"metadata rows({len(paper_ids)}) != embeddings rows({X.shape[0]})")

    for k in range(args.k_min, args.k_max + 1):
        labels = kmeans_cosine(X, k=k, seed=args.seed)
        mapping = {pid: int(lbl) for pid, lbl in zip(paper_ids, labels)}
        sizes = {}
        for lbl in labels:
            sizes[int(lbl)] = sizes.get(int(lbl), 0) + 1

        out = {
            "k": k,
            "seed": args.seed,
            "paper_id_to_cluster": mapping,
            "cluster_sizes": sizes,
        }
        out_path = args.out_dir / f"neurips_clusters_k{k}.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        print("wrote", out_path)

if __name__ == "__main__":
    main()
