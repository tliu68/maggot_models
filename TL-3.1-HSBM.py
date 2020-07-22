import numpy as np
import pandas as pd
import warnings
# import matplotlib.pyplot as plt

from anytree import NodeMixin, RenderTree, LevelOrderGroupIter
# from scipy.cluster.hierarchy import dendrogram, linkage
# from sklearn.cluster import AgglomerativeClustering

from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed, select_dimension
from graspy.cluster import GaussianCluster, KMeansCluster, AutoGMMCluster
from graspy.models import BaseGraphEstimator
from graspy.utils import augment_diagonal, pass_to_ranks
from sklearn.utils import check_array


def _check_common_inputs(
        n_components, min_components, max_components, embed_kws, cluster_kws
        ):
    if not isinstance(n_components, int) and n_components is not None:
        raise TypeError("n_components must be an int or None")
    elif n_components is not None and n_components < 1:
        raise ValueError("n_components must be > 0")

    if not isinstance(min_components, int):
        raise TypeError("min_components must be an int")
    elif min_components < 1:
        raise ValueError("min_components must be > 0")

    if not isinstance(max_components, int):
        raise TypeError("max_components must be an int")
    elif max_components < 1:
        raise ValueError("max_components must be > 0")
    elif max_components < min_components:
        raise ValueError("max_components must be >= min_components")

    if not isinstance(cluster_kws, dict):
        raise TypeError("cluster_kws must be a dict")

    if not isinstance(embed_kws, dict):
        raise TypeError("embed_kws must be a dict")


class RecursiveCluster(NodeMixin):
    def __init__(
        self,
        name,
        X,
        adj=None,
        selection_criteria="bic",
        embed_method="ASE",
        cluster_method="GMM",
        root_inds=None,
        n_init=1,
        ifembed=False,
        reembed=False,
        parent=None,
        min_components=1,
        max_components=None,
        n_components=None,
        n_elbows=2,
        normalize=False,
        regularizer=None,
        min_split=10,
        plus_c=False,
        embed_kws={},
        cluster_kws={},
    ):
        super().__init__()

        _check_common_inputs(
            n_components, min_components, max_components,
            embed_kws, cluster_kws)

        if cluster_method not in ["GMM", "KMeans", "Spherical-Kmeans"]:
            msg = "clustering method must be one of {GMM, Kmeans, Spherical-Kmeans}"
            raise ValueError(msg)

        if embed_method not in ["ASE", "LSE"]:
            msg = "embedding method must be either ASE or LSE"
            raise ValueError(msg)

        self.name = name
        self.parent = parent
        self.ifembed = ifembed
        self.reembed = reembed
        self.n_init = n_init
        self.min_components = min_components
        self.max_components = max_components
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
        self.embed_kws = embed_kws
        self.cluster_kws = cluster_kws

        labels = range(len(self.X))
        labels = pd.DataFrame(labels)
        labels.columns = ["inds"]
        self.labels = labels

        if root_inds is None:
            root_inds = labels["inds"]
        self.root_inds = root_inds

    def fit(self, X):
        self.fit_predict(X)
        return self

    def fit_predict(self, X):
        X = check_array(X, dtype=[np.float64, np.float32], 
                        ensure_min_samples=1)

        if self.max_components is None:
            lower_ncomponents = 1
            upper_ncomponents = self.min_components
        else:
            lower_ncomponents = self.min_components
            upper_ncomponents = self.max_components

        if upper_ncomponents > X.shape[0]:
            if self.max_components is None:
                msg = "if max_components is None then min_components must be >= "
                msg += "n_samples, but min_components = {}, n_samples = {}".format(
                    upper_ncomponents, X.shape[0]
                )
            else:
                msg = "max_components must be >= n_samples, but max_components = "
                msg += "{}, n_samples = {}".format(upper_ncomponents, X.shape[0])
            raise ValueError(msg)
        elif lower_ncomponents > X.shape[0]:
            msg = "min_components must be <= n_samples, but min_components = "
            msg += "{}, n_samples = {}".format(upper_ncomponents, X.shape[0])
            raise ValueError(msg)

        # TODO: check if the total # of clusters exceeds
        # max_components or max_levels?
        while True:
            all_k = []
            for i, node in enumerate(self.get_lowest_level()):
                if not hasattr(node, "k_") or node.k_ > 1:
                    node.fit_candidates()
                    node.select_model()
                    all_k.append(node.k_)
            self.collect_labels()
            if all_k.count(1) == len(all_k):
                break

        self._cum_clusters()
        return self.labels

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
                X = self._embed()
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

    def _embed(self, n_components=None):
        # TODO: embed on X?
        if hasattr(self, "adj"):
            adj = self.adj
        else:
            adj = self._to_adjacency()

        embed_adj = pass_to_ranks(adj)
        if self.plus_c:
            embed_adj += 1 / adj.size

        if n_components is None:
            n_components = self.n_components
            if self.n_components is None:
                max_dim = 0
                for j, node in enumerate(self.get_lowest_level()):
                    dim = select_dimension(node.X, n_elbows=node.n_elbows)[0][-1]
                    if max_dim < dim:
                        max_dim = dim
                n_components = max_dim

        if self.embed_method == "ASE":
            embedder = AdjacencySpectralEmbed(
                n_components=n_components, **self.embed_kws
            )
            embed = embedder.fit_transform(embed_adj)
        elif self.embed_method == "LSE":
            embedder = LaplacianSpectralEmbed(
                n_components=n_components, **self.embed_kws
            )
            embed = embedder.fit_transform(embed_adj)

        if self.normalize:
            row_norms = np.linalg.norm(embed, axis=1)
            embed /= row_norms[:, None]

        return embed

    def _cluster(self, embed, cluster_method=None, n_init=None):
        if self.cluster_method == "GMM":
            # TODO: self.max_components should be the sum of
            # numbers of all clusters at each level...?
            cluster = AutoGMMCluster(
                min_components=1, max_components=self.max_components,
                **self.cluster_kws)
            cluster.fit(embed)
            model = cluster.model_
            bic = model.bic(embed)
            lik = model.score(embed)
            k = cluster.n_components_
            pred = cluster.fit_predict(embed)

        res = {
            "bic": -bic,
            "lik": lik,
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
                        min_components=1,
                        max_components=3,
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

    def _cum_clusters(self):
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

rc = RecursiveCluster("0", X=x, max_components=3, min_split=1, ifembed=False)
np.random.seed(8888)
print(rc.fit_predict(x))
