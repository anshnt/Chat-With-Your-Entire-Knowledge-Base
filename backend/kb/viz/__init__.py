"""Corpus visualisation: projection, clustering, and the document graph."""

from kb.viz.clustering import Cluster, cluster_corpus, distinctive_terms, kmeans, suggest_k
from kb.viz.corpus_map import CorpusMap, CorpusMapBuilder, MapPoint, document_graph
from kb.viz.projection import Projection, normalize_to_unit_square, project
from kb.viz.render import cluster_colour, render_corpus_map

__all__ = [
    "Cluster",
    "CorpusMap",
    "CorpusMapBuilder",
    "MapPoint",
    "Projection",
    "cluster_colour",
    "cluster_corpus",
    "distinctive_terms",
    "document_graph",
    "kmeans",
    "normalize_to_unit_square",
    "project",
    "render_corpus_map",
    "suggest_k",
]
