#!/usr/bin/env python3
"""
Compute pairwise cosine similarities from NeurIPS embeddings.
Creates a JSON file with edges for the graph visualization.

Usage:
    python scripts/compute_similarities.py --threshold 0.7 --top-k 10
"""

import argparse
import json
import numpy as np
from pathlib import Path


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix."""
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embeddings / norms

    # Compute similarity matrix
    return np.dot(normalized, normalized.T)


def main():
    parser = argparse.ArgumentParser(description='Compute embedding similarities')
    parser.add_argument('--threshold', type=float, default=0.75,
                        help='Minimum similarity threshold for edges (default: 0.75)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Maximum neighbors per node (default: 5)')
    parser.add_argument('--embeddings', type=str,
                        default='data/embeddings_Neu/embeddings.npy',
                        help='Path to embeddings.npy')
    parser.add_argument('--metadata', type=str,
                        default='data/embeddings_Neu/metadata.csv',
                        help='Path to metadata.csv')
    parser.add_argument('--output', type=str,
                        default='data/embeddings_Neu/similarities.json',
                        help='Output JSON path')
    args = parser.parse_args()

    print(f"Loading embeddings from {args.embeddings}...")
    embeddings = np.load(args.embeddings)
    print(f"Loaded embeddings shape: {embeddings.shape}")

    # Load paper IDs from metadata
    print(f"Loading metadata from {args.metadata}...")
    paper_ids = []
    with open(args.metadata, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            if line.strip():
                # Parse CSV - first column is paper_id
                parts = line.split(',', 1)
                if parts:
                    paper_ids.append(parts[0].strip())

    print(f"Loaded {len(paper_ids)} paper IDs")

    if len(paper_ids) != embeddings.shape[0]:
        print(f"Warning: paper_ids ({len(paper_ids)}) != embeddings ({embeddings.shape[0]})")
        # Truncate to minimum
        n = min(len(paper_ids), embeddings.shape[0])
        paper_ids = paper_ids[:n]
        embeddings = embeddings[:n]

    print("Computing similarity matrix...")
    sim_matrix = cosine_similarity_matrix(embeddings)

    print(f"Creating edges (threshold={args.threshold}, top-k={args.top_k})...")
    edges = []
    edge_set = set()  # To avoid duplicate edges

    for i in range(len(paper_ids)):
        # Get similarities for this paper
        sims = sim_matrix[i]

        # Get top-k indices (excluding self)
        indices = np.argsort(sims)[::-1]

        count = 0
        for j in indices:
            if i == j:
                continue

            similarity = float(sims[j])
            if similarity < args.threshold:
                break

            # Create unique edge key
            edge_key = tuple(sorted([i, j]))
            if edge_key in edge_set:
                continue

            edge_set.add(edge_key)
            edges.append({
                'source': paper_ids[i],
                'target': paper_ids[j],
                'similarity': round(similarity, 4)
            })

            count += 1
            if count >= args.top_k:
                break

    print(f"Created {len(edges)} edges")

    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        'edges': edges,
        'threshold': args.threshold,
        'top_k': args.top_k,
        'total_papers': len(paper_ids)
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
