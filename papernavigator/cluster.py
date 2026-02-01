"""Clustering engine for paper semantic similarity analysis.

This module provides embedding and clustering functionality for academic papers,
enabling discovery of research themes and related work through semantic analysis.

WHY: Clustering papers by semantic similarity helps identify research themes,
     find related work, and understand the structure of a research domain.
HOW: 1. Embed title + abstract using OpenAI text-embedding-3-small
     2. Reduce dimensions with UMAP/t-SNE/PCA for visualization
     3. Cluster using HDBSCAN/DBSCAN (auto k) or KMeans (manual k)

Note: UMAP and HDBSCAN are optional dependencies that require Python < 3.13.
      When unavailable, PCA and DBSCAN are used as fallbacks.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from openai import OpenAI

from papernavigator.openai_usage import record_openai_response, raise_if_openai_insufficient_funds

# Feature availability flags
_HAS_UMAP = False
_HAS_HDBSCAN = False


def _check_optional_deps() -> tuple[bool, bool]:
    """Check availability of optional dependencies."""
    global _HAS_UMAP, _HAS_HDBSCAN

    try:
        import umap  # noqa: F401
        _HAS_UMAP = True
    except ImportError:
        _HAS_UMAP = False

    try:
        import hdbscan  # noqa: F401
        _HAS_HDBSCAN = True
    except ImportError:
        _HAS_HDBSCAN = False

    return _HAS_UMAP, _HAS_HDBSCAN


# Check on module load
_check_optional_deps()


@dataclass
class ClusterSummary:
    """Summary statistics for a single cluster."""
    cluster_id: int
    label: str
    count: int
    paper_indices: list[int]
    top_papers: list[dict[str, Any]]  # Top 3 papers by citation count


@dataclass
class ClusteringResult:
    """Complete clustering result with all metadata."""
    method: str
    dim_reduction: str
    n_clusters: int
    labels: np.ndarray
    coords_2d: np.ndarray
    cluster_summaries: list[ClusterSummary]
    papers: list[dict[str, Any]]


class EmbeddingService:
    """Service for generating text embeddings using OpenAI API.
    
    Uses text-embedding-3-small model which is cost-effective (1536 dims).
    Handles batching and text truncation automatically.
    """

    MODEL = "text-embedding-3-small"
    MAX_TOKENS = 8191  # Model limit
    MAX_CHARS = 8000 * 4  # Approximate character limit (4 chars per token avg)
    BATCH_SIZE = 100  # OpenAI recommends batching

    def __init__(self, client: OpenAI | None = None):
        """Initialize embedding service.
        
        Args:
            client: Optional OpenAI client. If None, creates a new one.
        """
        self.client = client or OpenAI()

    def _prepare_text(self, title: str, abstract: str | None) -> str:
        """Combine and truncate title + abstract for embedding.
        
        Args:
            title: Paper title
            abstract: Paper abstract (may be None)
            
        Returns:
            Combined text, truncated if necessary
        """
        text = title
        if abstract:
            text = f"{title}\n\n{abstract}"

        # Truncate if too long
        if len(text) > self.MAX_CHARS:
            text = text[:self.MAX_CHARS]

        return text

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape (n_texts, 1536)
        """
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i:i + self.BATCH_SIZE]
            try:
                response = self.client.embeddings.create(
                    model=self.MODEL,
                    input=batch,
                )
            except Exception as exc:
                raise_if_openai_insufficient_funds(exc)
                raise
            record_openai_response(response, model=self.MODEL)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    def embed_papers(self, papers: list[dict[str, Any]]) -> np.ndarray:
        """Generate embeddings for a list of paper dictionaries.
        
        Args:
            papers: List of paper dicts with 'title' and optional 'abstract' keys
            
        Returns:
            numpy array of shape (n_papers, 1536)
        """
        texts = [
            self._prepare_text(
                p.get("title", ""),
                p.get("abstract")
            )
            for p in papers
        ]
        return self.embed_texts(texts)


class ClusteringEngine:
    """Engine for clustering papers by semantic similarity.
    
    Supports HDBSCAN (automatic cluster detection) and KMeans (manual k).
    Uses UMAP or t-SNE for dimensionality reduction before visualization.
    """

    # HDBSCAN defaults
    HDBSCAN_MIN_CLUSTER_SIZE = 3
    HDBSCAN_MIN_SAMPLES = 2

    # KMeans default
    DEFAULT_N_CLUSTERS = 5

    # Dimensionality reduction defaults
    UMAP_N_NEIGHBORS = 15
    UMAP_MIN_DIST = 0.1
    TSNE_PERPLEXITY = 30

    def __init__(self, embedding_service: EmbeddingService | None = None):
        """Initialize clustering engine.
        
        Args:
            embedding_service: Optional embedding service. Creates one if None.
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self._last_eps: float | None = None

    def embed_papers(self, papers: list[dict[str, Any]]) -> np.ndarray:
        """Generate embeddings for papers.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            numpy array of embeddings
        """
        return self.embedding_service.embed_papers(papers)

    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: Literal["umap", "tsne", "pca"] = "umap",
    ) -> np.ndarray:
        """Reduce embedding dimensions to 2D for visualization.
        
        Args:
            embeddings: High-dimensional embeddings (n_samples, n_dims)
            method: "umap", "tsne", or "pca"
            
        Returns:
            2D coordinates (n_samples, 2)
            
        Note:
            UMAP requires the optional umap-learn package.
            Falls back to PCA if UMAP is requested but unavailable.
        """
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        if method == "umap":
            if not _HAS_UMAP:
                # Fallback to PCA when UMAP not available
                import warnings
                warnings.warn(
                    "umap-learn not installed (requires Python < 3.13). "
                    "Falling back to PCA for dimension reduction.",
                    UserWarning,
                )
                reducer = PCA(n_components=2, random_state=42)
                return reducer.fit_transform(embeddings)

            import umap
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(self.UMAP_N_NEIGHBORS, len(embeddings) - 1),
                min_dist=self.UMAP_MIN_DIST,
                metric="cosine",
                random_state=42,
            )
            return reducer.fit_transform(embeddings)

        elif method == "tsne":
            # Perplexity must be less than n_samples
            perplexity = min(self.TSNE_PERPLEXITY, len(embeddings) - 1)
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42,
            )
            return reducer.fit_transform(embeddings)

        elif method == "pca":
            reducer = PCA(n_components=2, random_state=42)
            return reducer.fit_transform(embeddings)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'umap', 'tsne', or 'pca'.")

    def _find_optimal_eps(self, normalized_embeddings: np.ndarray, min_samples: int) -> float:
        """Find optimal eps for DBSCAN using knee detection on k-distance graph.
        
        Uses the "knee" method: finds the point of maximum curvature in the
        sorted k-nearest neighbor distances, which typically indicates the
        optimal eps threshold.
        
        Args:
            normalized_embeddings: Normalized embeddings
            min_samples: Minimum samples parameter for DBSCAN
            
        Returns:
            Optimal eps value
        """
        from sklearn.neighbors import NearestNeighbors

        k = min(min_samples + 1, len(normalized_embeddings) - 1)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(normalized_embeddings)
        distances, _ = nn.kneighbors(normalized_embeddings)

        # Get k-th neighbor distances (sorted)
        k_distances = np.sort(distances[:, -1])

        # Find knee point using simple curvature method
        # Calculate second derivative (curvature) at each point
        n = len(k_distances)
        if n < 3:
            # Fallback to median if too few points
            return float(np.median(k_distances))

        # Use a sliding window to find maximum curvature
        # Curvature approximated as second difference
        diffs1 = np.diff(k_distances)
        diffs2 = np.diff(diffs1)

        # Find point of maximum curvature (knee)
        # Add small epsilon to avoid division by zero
        curvature = np.abs(diffs2) / (diffs1[1:] + 1e-10)

        # Find index of maximum curvature
        knee_idx = np.argmax(curvature)

        # Use the distance at the knee point
        eps = float(k_distances[knee_idx])

        # Ensure eps is reasonable (not too small or too large)
        min_eps = float(np.percentile(k_distances, 10))
        max_eps = float(np.percentile(k_distances, 80))
        eps = np.clip(eps, min_eps, max_eps)

        return eps

    def cluster(
        self,
        embeddings: np.ndarray,
        method: Literal["hdbscan", "dbscan", "kmeans"] = "hdbscan",
        n_clusters: int | None = None,
        eps: float | None = None,
        min_samples: int | None = None,
    ) -> np.ndarray:
        """Cluster embeddings using HDBSCAN, DBSCAN, or KMeans.
        
        Args:
            embeddings: Embeddings to cluster (n_samples, n_dims)
            method: "hdbscan" (auto k), "dbscan" (auto k), or "kmeans" (manual k)
            n_clusters: Number of clusters for KMeans (ignored for density methods)
            eps: Eps parameter for DBSCAN (auto-selected if None)
            min_samples: Min samples for DBSCAN/HDBSCAN (uses default if None)
            
        Returns:
            Cluster labels (n_samples,). -1 indicates noise for density methods.
            
        Note:
            HDBSCAN requires the optional hdbscan package.
            Falls back to DBSCAN if HDBSCAN is requested but unavailable.
        """
        from sklearn.cluster import DBSCAN, KMeans
        from sklearn.preprocessing import StandardScaler

        min_samples = min_samples or self.HDBSCAN_MIN_SAMPLES

        if method == "hdbscan":
            if not _HAS_HDBSCAN:
                # Fallback to DBSCAN when HDBSCAN not available
                import warnings
                warnings.warn(
                    "hdbscan not installed (requires Python < 3.13). "
                    "Falling back to DBSCAN for clustering.",
                    UserWarning,
                )
                # Normalize for DBSCAN
                scaler = StandardScaler()
                normalized = scaler.fit_transform(embeddings)

                # Auto-select eps if not provided
                if eps is None:
                    eps = self._find_optimal_eps(normalized, min_samples)
                    # Store for potential logging
                    self._last_eps = eps
                else:
                    self._last_eps = eps

                clusterer = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    metric="euclidean",
                )
                return clusterer.fit_predict(normalized)

            import hdbscan
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(self.HDBSCAN_MIN_CLUSTER_SIZE, 2),
                min_samples=min_samples,
                metric="euclidean",
            )
            return clusterer.fit_predict(embeddings)

        elif method == "dbscan":
            scaler = StandardScaler()
            normalized = scaler.fit_transform(embeddings)

            # Auto-select eps if not provided
            if eps is None:
                eps = self._find_optimal_eps(normalized, min_samples)
                # Store for potential logging
                self._last_eps = eps
            else:
                self._last_eps = eps

            clusterer = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric="euclidean",
            )
            return clusterer.fit_predict(normalized)

        elif method == "kmeans":
            k = n_clusters or self.DEFAULT_N_CLUSTERS
            # Ensure k doesn't exceed number of samples
            k = min(k, len(embeddings))

            clusterer = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10,
            )
            return clusterer.fit_predict(embeddings)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'hdbscan', 'dbscan', or 'kmeans'.")

    def get_cluster_summaries(
        self,
        papers: list[dict[str, Any]],
        labels: np.ndarray,
    ) -> list[ClusterSummary]:
        """Generate summary statistics for each cluster.
        
        Args:
            papers: Original paper dictionaries
            labels: Cluster labels from clustering
            
        Returns:
            List of ClusterSummary objects
        """
        unique_labels = sorted(set(labels))
        summaries = []

        for cluster_id in unique_labels:
            # Get indices of papers in this cluster
            indices = [i for i, l in enumerate(labels) if l == cluster_id]

            # Get papers in this cluster
            cluster_papers = [papers[i] for i in indices]

            # Sort by citation count descending
            sorted_papers = sorted(
                cluster_papers,
                key=lambda p: p.get("citation_count", 0),
                reverse=True,
            )

            # Take top 3 for summary
            top_papers = [
                {
                    "title": p.get("title", "Unknown"),
                    "year": p.get("year"),
                    "citations": p.get("citation_count", 0),
                }
                for p in sorted_papers[:3]
            ]

            # Generate label
            if cluster_id == -1:
                label = "Noise (unclustered)"
            else:
                label = f"Cluster {cluster_id}"

            summaries.append(ClusterSummary(
                cluster_id=cluster_id,
                label=label,
                count=len(indices),
                paper_indices=indices,
                top_papers=top_papers,
            ))

        return summaries

    @staticmethod
    def get_available_features() -> dict[str, bool]:
        """Check which optional clustering features are available.
        
        Returns:
            Dict with 'umap' and 'hdbscan' availability flags
        """
        return {
            "umap": _HAS_UMAP,
            "hdbscan": _HAS_HDBSCAN,
        }

    def run_full_pipeline(
        self,
        papers: list[dict[str, Any]],
        cluster_method: Literal["hdbscan", "dbscan", "kmeans"] = "hdbscan",
        dim_method: Literal["umap", "tsne", "pca"] = "umap",
        n_clusters: int | None = None,
        eps: float | None = None,
        min_samples: int | None = None,
    ) -> ClusteringResult:
        """Run the full clustering pipeline.
        
        Args:
            papers: List of paper dictionaries
            cluster_method: "hdbscan", "dbscan", or "kmeans"
            dim_method: "umap", "tsne", or "pca"
            n_clusters: Number of clusters (kmeans only)
            eps: Eps parameter for DBSCAN (auto-selected if None)
            min_samples: Min samples for DBSCAN/HDBSCAN (uses default if None)
            
        Returns:
            Complete ClusteringResult with all data
            
        Note:
            If umap-learn or hdbscan are not installed, automatic
            fallbacks to PCA and DBSCAN are used respectively.
        """
        # Step 1: Embed papers
        embeddings = self.embed_papers(papers)

        # Step 2: Reduce dimensions for visualization
        coords_2d = self.reduce_dimensions(embeddings, method=dim_method)

        # Step 3: Cluster (on full embeddings, not 2D)
        labels = self.cluster(
            embeddings,
            method=cluster_method,
            n_clusters=n_clusters,
            eps=eps,
            min_samples=min_samples,
        )

        # Step 4: Generate summaries
        summaries = self.get_cluster_summaries(papers, labels)

        # Count actual clusters (excluding noise for HDBSCAN)
        actual_n_clusters = len([s for s in summaries if s.cluster_id != -1])

        return ClusteringResult(
            method=cluster_method,
            dim_reduction=dim_method,
            n_clusters=actual_n_clusters,
            labels=labels,
            coords_2d=coords_2d,
            cluster_summaries=summaries,
            papers=papers,
        )

    def to_json(self, result: ClusteringResult) -> dict[str, Any]:
        """Convert ClusteringResult to JSON-serializable dict.
        
        Args:
            result: ClusteringResult from run_full_pipeline
            
        Returns:
            JSON-serializable dictionary
        """
        def _to_native(val: Any) -> Any:
            """Convert numpy types to native Python types."""
            if isinstance(val, (np.integer, np.int32, np.int64)):
                return int(val)
            elif isinstance(val, (np.floating, np.float32, np.float64)):
                return float(val)
            elif isinstance(val, np.ndarray):
                return val.tolist()
            return val

        clusters_json = []
        for summary in result.cluster_summaries:
            cluster_papers = [result.papers[i] for i in summary.paper_indices]
            clusters_json.append({
                "id": _to_native(summary.cluster_id),
                "label": summary.label,
                "count": summary.count,
                "top_papers": summary.top_papers,
                "papers": [
                    {
                        "paper_id": p.get("paper_id", ""),
                        "title": p.get("title", "Unknown"),
                        "year": _to_native(p.get("year")),
                        "citation_count": _to_native(p.get("citation_count", 0)),
                        "x": float(result.coords_2d[i, 0]),
                        "y": float(result.coords_2d[i, 1]),
                    }
                    for i, p in zip(summary.paper_indices, cluster_papers)
                ],
            })

        return {
            "method": result.method,
            "dim_reduction": result.dim_reduction,
            "n_clusters": _to_native(result.n_clusters),
            "total_papers": len(result.papers),
            "clusters": clusters_json,
        }
