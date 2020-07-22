import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from anytree import NodeMixin, RenderTree, LevelOrderGroupIter
# from scipy.cluster.hierarchy import dendrogram, linkage
# from sklearn.cluster import AgglomerativeClustering

from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed, select_dimension
from graspy.cluster import GaussianCluster, KMeansCluster, AutoGMMCluster
from graspy.models import BaseGraphEstimator
from graspy.utils import augment_diagonal, pass_to_ranks


class RecursiveCluster(NodeMixin):
    def __init__(
        self,
        name,
        X,
        adj=None,
        label_init=None,
        selection_criteria="bic",
        embed_method="ASE",
        cluster_method="GMM",
        root_inds=None,
        n_init=1,
        ifembed=False,
        reembed=False,
        parent=None,
        min_clusters=1,
        max_clusters=10,
        n_components=None,
        n_elbows=2,
        normalize=False,
        regularizer=None,
        min_split=10,
        plus_c=False,
    ):
        super().__init__()
        self.name = name
        self.parent = parent
        self.ifembed = ifembed
        self.reembed = reembed
        self.n_init = n_init
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.n_components = n_components
        self.n_elbows = n_elbows
        self.normalize = normalize
        self.embed_method = embed_method
        self.cluster_method = cluster_method
        self.regularizer = regularizer
        self.plus_c = plus_c
        self.min_split = min_split
        self.selection_criteria = selection_criteria
        self.X = X
        self.adj = adj

        # if cluster_method not in ["GMM", "KMeans", "Spherical-Kmeans"]:
        #     msg = "clustering method must be one of [GMM, Kmeans, Spherical-Kmeans]"
        #     raise ValueError(msg)

        # if embed_method not in ["ASE", "LSE"]:
        #     msg = "embedding method must be either ASE or LSE"
        #     raise ValueError(msg)

        labels = range(len(self.X))
        labels = pd.DataFrame(labels)
        labels.columns = ["inds"]
        self.labels = labels

    def fit(self):
        while True:
            all_k = []
            for i, node in enumerate(self.get_lowest_level()):
                node.fit_candidates()
                node.select_model()
                all_k.append(node.k_)
            self.collect_labels()
            if all_k.count(1) == len(all_k):
                break

        mask = self.labels.columns.str.endswith(('pred'))
        self.labels = self.labels.loc[:, ~mask]

    def fit_candidates(self):
        if hasattr(self, "k_") and self.k_ == 0:
            return

        if self.is_root:
            if self.ifembed is True:
                X = self._embed()
            else:
                X = self.X
        elif self.is_root is False:
            if self.reembed is True:
                n_components = self._choose_ncomponents()
                X = self._embed(n_components=n_components)
            else:
                X = self.X

        if np.linalg.norm(X) < 1:
            X = X / np.linalg.norm(X)

        # self.X_ = X
        results = self._cluster(
            X,
            cluster_method="GMM",
            n_init=self.n_init,
        )

        self.results_ = results

    def _to_adjacency(self):
        if self.adj is None:
            pass
        # TODO: convert data matrix e.g. distance or similarity matrix
        # to adjacency matrix

    def _choose_ncomponents(self):
        max_dim = 0
        for j, node in enumerate(self.get_lowest_level()):
            dim = select_dimension(node.X, n_elbows=node.n_elbows)[0][-1]
            if max_dim < dim:
                max_dim = dim

        return max_dim

    def _embed(self, n_components=None):
        if hasattr(self, "adj"):
            adj = self.adj
        else:
            adj = self._to_adjacency()

        embed_adj = pass_to_ranks(adj)
        if self.plus_c:
            embed_adj += 1 / adj.size

        if n_components is None:
            n_components = self.n_components

        if self.embed_method == "ASE":
            embedder = AdjacencySpectralEmbed(
                n_components=n_components, n_elbows=self.n_elbows
            )
            embed = embedder.fit_transform(embed_adj)
        elif self.embed_method == "LSE":
            embedder = LaplacianSpectralEmbed(
                n_components=n_components,
                n_elbows=self.n_elbows,
                regularizer=self.regularizer,
            )
            embed = embedder.fit_transform(embed_adj)

        if self.normalize:
            row_norms = np.linalg.norm(embed, axis=1)
            embed /= row_norms[:, None]
       
        return embed

    def _cluster(self, embed, cluster_method=None, n_init=None):
        if self.cluster_method == "GMM":
            cluster = AutoGMMCluster(
                min_components=1, max_components=self.max_clusters
            )
            cluster.fit(embed)
            model = cluster.model_
            bic_ = model.bic(embed)
            lik_ = model.score(embed)
            k = cluster.n_components_
            pred = cluster.fit_predict(embed)

        res = {
            "bic": -bic_,
            "lik": lik_,
            "k": k,
            "model": model,
            "pred": pred,
        }
        # TODO: if cluster_method == "Spherical-KMeans":
        # TODO: if cluster_method == "KMeans":

        return res

    def _model_metrics(self):
        results = self.results_
        model = results["model"]
        pred = results["pred"]
        _n_components = results["k"]

        return model, pred, _n_components

    def select_model(self, k=None):
        if k is None:
            model, pred, k = self._model_metrics()

        self.k_ = k
        self.children = []
        if k > 1:
            self.model_ = model
            self.pred_ = pred
            root_labels = self.root.labels

            pred_name = f"{self.depth + 1}_pred"
            if pred_name not in root_labels.columns:
                root_labels[pred_name] = ""
            root_labels.loc[self.root_inds, pred_name] = pred.astype(str)

            uni_labels = np.unique(pred).astype(str)

            self.children = []
            for i, ul in enumerate(uni_labels):
                new_labels = root_labels[
                    (root_labels[pred_name] == ul)
                    & (root_labels.index.isin(self.root_inds.index))
                ]
                new_root_inds = new_labels["inds"]
                new_name = self.name + "-" + str(ul)
                new_x = self.root.X[new_root_inds]
                # new_adj = self.root.adj[np.ix_(new_root_inds, new_root_inds)]
                if len(new_labels) > self.min_split:
                    RecursiveCluster(
                        new_name,
                        X=new_x,
                        selection_criteria="bic",
                        embed_method="ASE",
                        cluster_method="GMM",
                        root_inds=new_root_inds,
                        n_init=1,
                        reembed=False,
                        parent=self,
                        min_clusters=1,
                        max_clusters=3,
                        n_components=None,
                        n_elbows=2,
                        normalize=False,
                        regularizer=None,
                        plus_c=True,
                        min_split=1,
                    )

    def collect_labels(self):
        labels = self.root.labels
        labels[f"lvl0_labels"] = "0"
        for i in range(1, self.height + 1):
            labels[f"lvl{i}_labels"] = labels[f"lvl{i-1}_labels"] + "-" + labels[f"{i}_pred"]

    def predict(self):
        n_levels = self.height
        labels = self.labels
        for i in range(1, n_levels+1):
            uni_labels = np.unique(labels[f"lvl{i}_labels"])
            labels["new"] = ''
            for j in range(len(uni_labels)):
                mask = labels[f"lvl{i}_labels"] == uni_labels[j]
                labels[mask] = labels[mask].assign(new=j)
            labels.rename(columns={"new": f"lvl{i}_cum_clusters"}, inplace=True)

        mask = labels.columns.str.endswith(('pred'))
        labels = labels.loc[:, ~mask]

        return labels

    def get_lowest_level(self):
        level_it = LevelOrderGroupIter(self)
        last = next(level_it)
        nxt = last
        while nxt is not None:
            last = nxt
            nxt = next(level_it, None)
        return last


x = np.zeros((15,2))
x[:3,:] = [[1,0], [2,0], [3,0]]
x[3:6,0] = x[:3,0] + 15
x[6:12,0] = x[:6,0] + 50
x[12:15,0] = x[9:12,0] + (x[9,0] - x[6,0])

# plt.scatter(x[:,0], x[:,1])

rc = RecursiveCluster("0", X=x, max_clusters=3, min_split=1, ifembed=False)
np.random.seed(8888)
rc.fit()
print(rc.predict())

