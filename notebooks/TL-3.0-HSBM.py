import numpy as np
import pandas as pd
import seaborn as sns

from anytree import NodeMixin, Node, RenderTree, LevelOrderGroupIter, PostOrderIter
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed, select_dimension
from graspy.cluster import GaussianCluster, KMeansCluster
from graspy.models import BaseGraphEstimator
from graspy.utils import augment_diagonal


class BaseTree(NodeMixin):
    def __init__(self, X, root_inds=None, parent=None):
        super().__init__()
        self.X = X
        self.parent = parent

    def _count_levels(self):
        levels = []
        for group in LevelOrderGroupIter(self):
            levels.append(group)

    def _assignment_to_tree(self, block_inds, block_vert_inds):
        # input is cluster assignment e.g., generated from GMM
        parents = {}
        root = Node("root", parent=None)
        for i in block_inds:
            parent = block_inds[i]
            parents[i] = Node(parent, parent=root)
            for j in range(len(block_vert_inds[i])):
                child = block_vert_inds[i][j]
                child = Node(child, parent=parents[i])
        assignment_tree = RenderTree(root)
        return assignment_tree

    def build_linkage(self):
        pass
        # merge trees
        # match level name to clusters 


class RecursiveCluster(BaseTree):
    def __init__(
        self,
        label_init=None,
        embed_method="ase",
        cluster_method="gmm",
        n_init=None,
        bandwidth=None,
        selection_criteria="bic",
        embed_kws={},
        cluster_kws={},
    ):
        super().__init__()
        self.label_init = label_init
        self.embed_method = embed_method
        self.cluster_method = cluster_method
        self.n_init = n_init
        self.bandwidth = bandwidth
        self.selection_criteria = selection_criteria
        self.embed_kws = embed_kws
        self.cluster_kws = cluster_kws

    def fit(self, graph):
        # check level
        # if level>1: for subgraph in graph: _cluster_to_subgraphs(subgraph)

        # check bic of current model -> bic1
        # then cluster & calculate bic again -> bic2
        # bic_ratio = bic2 / bic1
        # or simply check if bic2 < bic1

        return self


    def _embed(self, graph, n_components):
        embed_graph = augment_diagonal(graph, weight=self.diag_aug_weight)
        if self.embed_method == "ase":
            embed = AdjacencySpectralEmbed(
                n_components=n_components,
                **self.embed_kws
            )
        elif self.embed_method == "lse":
            embed = LaplacianSpectralEmbed(
                n_components=n_components,
                **self.embed_kws
            )

        latent = embed.fit_transform(embed_graph)
        if isinstance(latent, tuple):
            latent = np.concatenate(latent, axis=-1)
        # TODO: processing on latent

    def _cluster_to_subgraphs(self, graph, assignment_tree=None, reembed=True):
        if assignment_tree is None:
            levels = 1
        else:
            levels = _count_levels(assignment_tree)

        if len(levels) == 1:
            n_components = None
            latent = _embed(graph, n_components)
        elif len(levels) > 1:
            if reembed is True:
                max_dim = 0
                for inds in sub_vert_inds:
                    subgraph = graph[np.ix_(inds, inds)]
                    sublatent = _embed(subgraph, n_components=None)  # why not use select_dimension?
                    if max_dim < sublatent.shape[1]:
                        max_dim = sublatent.shape[1]
                latent = _embed(graph, max_dim)
            else:
                latent = graph

        if self.cluster_method == "gmm":
            cluster = GaussianCluster(
                min_components=self.n_subgraphs,
                max_components=self.n_subgraphs,
                n_init=10,
                **self.cluster_kws
            )
        if self.cluster_method == "sphere-kmeans":
            cluster = KMeansCluster(
                max_clusters=self.n_subgraphs, kmeans_kws=self.cluster_kws
            )
        # TODO: multiple inits w/ Kmeans?

        cluster.fit(latent)
        vertex_assignments = cluster.predict(latent)
        sub_vert_inds, sub_inds, sub_inv = _get_block_indices(vertex_assignments)
        assignment_tree = build_linkage(vertex_assignments)
        bic_ = cluster.model_.bic(latent)

        return assignment_tree, bic_

    def _get_block_indices(self, y):
        block_labels, block_inv, block_sizes = np.unique(
            y, return_inverse=True, return_counts=True
        )

        n_blocks = len(block_labels)
        block_inds = range(n_blocks)

        block_vert_inds = []
        for i in block_inds:
            inds = np.where(block_inv == i)[0]
            block_vert_inds.append(inds)

        return block_vert_inds, block_inds, block_inv


class HSBMEstimator(RecursiveCluster):
     def __init__(
        self,
        label_init=None,
        embed_method="ase",
        cluster_method="gmm",
        n_init=None,
        bandwidth=None,
        selection_criteria="bic",
        embed_kws={},
        cluster_kws={},
    ):
        super().__init__()

    def fit(self):
        pass

    def _cluster_to_motifs(self):
        subgraph_dissimilarities = _compute_subgraph_dissimilarities(
            subgraph_latents, sub_inds, self.bandwidth
        )
        agglom = AgglomerativeClustering(
            n_clusters=self.n_subgroups,
            affinity="precomputed",
            linkage=self.linkage,
            compute_full_tree=True,
        )
        

