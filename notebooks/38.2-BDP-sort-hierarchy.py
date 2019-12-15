# %% [markdown]
# # Load and import
import os
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from graspy.plot import gridplot, heatmap, pairplot
from graspy.simulations import sbm
from graspy.utils import binarize, get_lcc, is_fully_connected
from src.data import load_everything
from src.hierarchy import normalized_laplacian, signal_flow
from src.utils import savefig

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)
SAVEFIGS = True
DEFAULT_FMT = "png"
DEFUALT_DPI = 150

plt.style.use("seaborn-white")
sns.set_palette("deep")
sns.set_context("talk", font_scale=1)


def stashfig(name, **kws):
    if SAVEFIGS:
        savefig(name, foldername=FNAME, fmt=DEFAULT_FMT, dpi=DEFUALT_DPI, **kws)


# %% [markdown]
# # null simulation


def get_feedforward_B(low_p, diag_p, feedforward_p, n_blocks=5):
    B = np.zeros((n_blocks, n_blocks))
    B += low_p
    B -= np.diag(np.diag(B))
    B -= np.diag(np.diag(B, k=1), k=1)
    B += np.diag(diag_p * np.ones(n_blocks))
    B += np.diag(feedforward_p * np.ones(n_blocks - 1), k=1)
    return B


# block_probs = get_feedforward_B(low_p, diag_p, feedforward_p)
def get_recurrent_feedforward_B(low_p, diag_p, feedforward_p, n_blocks=5):
    alternating = np.ones(2 * n_blocks - 1)
    alternating[1::2] = 0
    late_alternating = np.ones(2 * n_blocks - 1)
    late_alternating[::2] = 0
    B = np.zeros((2 * n_blocks, 2 * n_blocks))
    B += low_p
    B -= np.diag(np.diag(B))
    B -= np.diag(np.diag(B, k=1), k=1)
    B -= np.diag(np.diag(B, k=-1), k=-1)
    B += np.diag(diag_p * np.ones(2 * n_blocks))
    B += np.diag((feedforward_p - low_p) * alternating[:-1], k=2)
    B += np.diag(diag_p * alternating, k=1)
    B += np.diag(low_p * late_alternating, k=1)
    B += np.diag(diag_p * alternating, k=-1)
    B += np.diag(low_p * late_alternating, k=-1)
    return B


def n_to_labels(n):
    n = np.array(n)
    n_cumsum = n.cumsum()
    labels = np.zeros(n.sum(), dtype=np.int64)
    for i in range(1, len(n)):
        labels[n_cumsum[i - 1] : n_cumsum[i]] = i
    return labels


# %% [markdown]
# #
low_p = 0.01
diag_p = 0.2
feedforward_p = 0.3
n_blocks = 5


block_probs = get_feedforward_B(low_p, diag_p, feedforward_p, n_blocks=n_blocks)
plt.figure(figsize=(10, 10))
sns.heatmap(block_probs, annot=True, cmap="Reds", cbar=False)
plt.title("Feedforward block probability matrix")
stashfig("ffw-B")

#%%
community_sizes = np.empty(2 * n_blocks, dtype=int)
n_feedforward = 100
n_feedback = 100
community_sizes[::2] = n_feedforward
community_sizes[1::2] = n_feedback
community_sizes = n_blocks * [n_feedforward]
labels = n_to_labels(community_sizes)


A = sbm(community_sizes, block_probs, directed=True, loops=False)
n_verts = A.shape[0]

perm_inds = np.random.permutation(n_verts)
A_perm = A[np.ix_(perm_inds, perm_inds)]
heatmap(A, cbar=False, title="Feedforward SBM")
stashfig("ffSBM")

heatmap(A_perm, cbar=False, title="Feedforward SBM, shuffled")
stashfig("ffSBM-shuffle")

true_z = signal_flow(A)
sort_inds = np.argsort(true_z)[::-1]
heatmap(
    A[np.ix_(sort_inds, sort_inds)],
    cbar=False,
    title=r"Feedforward SBM, sorted by $A$ signal flow",
)
stashfig("ffSBM-adj-sf")

A_fake = A.copy().ravel()
np.random.shuffle(A_fake)
A_fake = A_fake.reshape((n_verts, n_verts))
fake_z = signal_flow(A_fake)
sort_inds = np.argsort(fake_z)[::-1]
heatmap(
    A_fake[np.ix_(sort_inds, sort_inds)],
    cbar=False,
    title="Random network, sorted by signal flow",
)
stashfig("random-sf")

adj_df = pd.DataFrame()
adj_df["Signal flow"] = true_z
adj_df["Input"] = "Adjacency"
adj_df["Block"] = labels

# fg = sns.FacetGrid(dist_df, col="Label", col_wrap=2, aspect=2, hue="Label")
# fg.map(sns.distplot, "Signal flow")
# stashfig("sf-dists")

# %% [markdown]
# # try with p_hat
from graspy.embed import AdjacencySpectralEmbed

latent = AdjacencySpectralEmbed(n_components=n_blocks).fit_transform(A)
P_hat = latent[0] @ latent[1].T
# P_hat -= P_hat.min()
heatmap(P_hat, title=r"$\hat{P}$")
stashfig("p-hat")
true_z = signal_flow(P_hat)
sort_inds = np.argsort(true_z)[::-1]
heatmap(
    A[np.ix_(sort_inds, sort_inds)],
    cbar=False,
    title=r"Feedforward SBM, sorted by $\hat{P}$ signal flow",
)
stashfig("ffSBM-phat-sf")

A_fake = A.copy().ravel()
np.random.shuffle(A_fake)
A_fake = A_fake.reshape((n_verts, n_verts))
fake_z = signal_flow(A_fake)
sort_inds = np.argsort(fake_z)[::-1]
heatmap(
    A_fake[np.ix_(sort_inds, sort_inds)],
    cbar=False,
    title="Random network, sorted by signal flow",
)
stashfig("random-sf")

p_hat_df = pd.DataFrame()
p_hat_df["Signal flow"] = true_z
p_hat_df["Input"] = r"$\hat{P}$"
p_hat_df["Block"] = labels

# %% [markdown]
# # Plot distributions of both against each other
total_df = pd.concat((adj_df, p_hat_df))

fg = sns.FacetGrid(
    total_df,
    col="Input",
    row="Block",
    aspect=2,
    hue="Block",
    margin_titles=True,
    sharex=True,
)
fg = fg.map(sns.distplot, "Signal flow")
fg.set(yticks=())
stashfig("sf-dists-separate")

fg = sns.FacetGrid(
    total_df, col="Input", aspect=2, hue="Block", margin_titles=True, sharex=True
)
fg = fg.map(sns.distplot, "Signal flow")
fg.add_legend()
fg.set(yticks=())
stashfig("sf-dists-squished")


# %%
from graspy.models import SBMEstimator

sbm = SBMEstimator()
sbm.fit(A)
B_hat = sbm.block_p_

sns.heatmap(B_hat)

z = signal_flow(B_hat)
sort_inds = np.argsort(z)[::-1]
plt.figure()
sns.heatmap(B_hat[np.ix_(sort_inds, sort_inds)])
