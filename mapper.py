# illustrative implementation of the Mapper algorithm from "Topological Methods for the Analysis of High Dimensional Data Sets and
# 3D Object Recognition", Eurographics Symposium on Point-Based Graphics, 2007.


from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

from itertools import product
from sklearn.cluster import DBSCAN


def euclidean_distances(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return cdist(X, X)


# Lens class holds and applies filters
class Lens:

    def __init__(self, filters, names=None): 
        if callable(filters):
            filters = [filters]
        self.filters = list(filters)
        self.names = names or [f"f{i}" for i in range(len(self.filters))]

    #D is pre-computed distance matrix, used in many filters (density, eccentricity, graph Laplacian)
    def transform(self, X: np.ndarray, D: np.ndarray | None = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if D is None:
            D = euclidean_distances(X)
        cols = []
        for filt in self.filters:
            v = np.asarray(filt(X, D), dtype=float)
            if v.ndim == 1:
                v = v[:, None] #reshape to (N, 1) for stacking
            cols.append(v)
        return np.hstack(cols)


# Filters 

#project onto a single coordinate axis (height)
def coordinate(axis: int = 0):
    return lambda X, D: X[:, axis]


#use a pre-computed filter values (ignores X, D)
def precomputed(values):
    values = np.asarray(values, dtype=float)
    return lambda X, D: values


#dist from point
def distance_from_point(p):
    p = np.asarray(p, dtype=float)
    return lambda X, D: np.linalg.norm(X - p, axis=1)

#dist from origin
def l2norm():
    return lambda X, D: np.linalg.norm(X, axis=1)

#first n components of PCA
def pca(n_components: int = 1):
    from sklearn.decomposition import PCA

    def f(X, D):
        return PCA(n_components=n_components).fit_transform(X)

    return f

#gaussian kernel density estimate, f(x) = sum exp(-d(x,y)^2 / eps).
#larger eps means smoother estimate
def density(eps: float | None = None):
    def f(X, D):
        e = eps if eps is not None else np.median(D[D > 0]) ** 2
        return np.exp(-(D ** 2) / e).sum(axis=1)

    return f

#opopposite from density, large for points far from center
# avg distance to all other points
#(mean d(x,y)^p)^(1/p)
def eccentricity(p: float = 1.0):
    def f(X, D):
        if np.isinf(p):
            return D.max(axis=1)
        return (np.mean(D ** p, axis=1)) ** (1.0 / p)

    return f

#the graph laplacian filter used in paper
def graph_laplacian(n_eigen: int = 2, eps: float | None = None):
    def f(X, D):
        e = eps if eps is not None else np.median(D[D > 0]) ** 2
        W = np.exp(-(D ** 2) / e)
        d = W.sum(axis=1)
        dinv = 1.0 / np.sqrt(d)
        L = np.eye(W.shape[0]) - (dinv[:, None] * W * dinv[None, :])
        vals, vecs = np.linalg.eigh(L)
        return vecs[:, 1:1 + n_eigen]

    return f


# Cover   
# overlapping windows over the range of the filter
class Cover:

    # builds windows from filter values f, covers of inverse images of f
    def fit(self, f):
        raise NotImplementedError

    # assigns points to windows
    def assign(self, f):
        raise NotImplementedError

    def fit_assign(self, f):
        return self.fit(f).assign(f)


#n overlapping intervals from lo to hi
#returns array of shape (n, 2) of [left, right] intervals
def _intervals(lo, hi, n, overlap):
    if n == 1:
        return np.array([[lo, hi]])
    step = (hi - lo) / n           
    padding = 0.5 * overlap * step    
    centers = lo + (np.arange(n) + 0.5) * step
    left = centers - step / 2 - padding
    right = centers + step / 2 + padding
    return np.column_stack([left, right])

class IntervalCover(Cover):

    def __init__(self, n_intervals: int = 10, overlap: float = 0.3):
        self.n_intervals = n_intervals
        self.overlap = overlap
        self.bounds_ = None

    #needs 1d filter, f is a 1d array of filter values
    def fit(self, f):
        f = np.asarray(f).reshape(len(f), -1)
        self.bounds_ = _intervals(f.min(), f.max(),
                                  self.n_intervals, self.overlap)
        return self

    def assign(self, f):
        v = np.asarray(f).reshape(-1)
        return [np.where((v >= lo) & (v <= hi))[0]
                for lo, hi in self.bounds_]


class RectangleCover(Cover):

    def __init__(self, n_per_axis=(8, 8), overlap: float = 0.3):
        if np.isscalar(n_per_axis):
            n_per_axis = (int(n_per_axis), int(n_per_axis))
        self.n_per_axis = tuple(n_per_axis)
        self.overlap = overlap
        self.bx_ = self.by_ = None

    #needs 2d filter, f is an (N, 2) array of filter values
    def fit(self, f):
        f = np.asarray(f)
        self.bx_ = _intervals(f[:, 0].min(), f[:, 0].max(),
                              self.n_per_axis[0], self.overlap)
        self.by_ = _intervals(f[:, 1].min(), f[:, 1].max(),
                              self.n_per_axis[1], self.overlap)
        return self

    def assign(self, f):
        f = np.asarray(f)
        out = []
        for lx, rx in self.bx_:
            in_x = (f[:, 0] >= lx) & (f[:, 0] <= rx) #filters points in x range
            for ly, ry in self.by_:
                in_y = (f[:, 1] >= ly) & (f[:, 1] <= ry) #filters points in y range
                out.append(np.where(in_x & in_y)[0])
        return out


class HexagonCover(Cover):
    def __init__(self, n: int = 8, overlap: float = 0.3):
        self.n = n
        self.overlap = overlap
        self.centers_ = None
        self.R_ = None  # circumradius with overlap 

    #needs 2d filter, f is an (N, 2) array of filter values
    def fit(self, f):
        f = np.asarray(f)
        x0, x1 = f[:, 0].min(), f[:, 0].max()
        y0, y1 = f[:, 1].min(), f[:, 1].max()
        W = max(x1 - x0, 1e-9)
        # pointy top hexagons
        R = W / (self.n * np.sqrt(3)) 
        dx = np.sqrt(3) * R          # column spacing
        dy = 1.5 * R                 # row spacing
        centers = []
        row = 0
        y = y0 - dy
        while y <= y1 + dy:
            offset = (dx / 2) if (row % 2) else 0.0
            x = x0 - dx
            while x <= x1 + dx:
                centers.append((x + offset, y))
                x += dx
            y += dy
            row += 1
        self.centers_ = np.array(centers)
        self.R_ = R * (1.0 + self.overlap)  
        return self

    def _inside(self, q):
        a = np.sqrt(3) / 2 * self.R_         
        c60, s60 = 0.5, np.sqrt(3) / 2
        return ((np.abs(q[:, 0]) <= a) &
                (np.abs(q[:, 0] * c60 + q[:, 1] * s60) <= a) &
                (np.abs(-q[:, 0] * c60 + q[:, 1] * s60) <= a))

    def assign(self, f):
        f = np.asarray(f)
        out = []
        for cx, cy in self.centers_:
            idx = np.where(self._inside(f - np.array([cx, cy])))[0]
            out.append(idx)
        # keep only non empty windows 
        return [w for w in out if len(w) > 0]

#built from overlapping angular intervals, axes are cicular
class CircularCover(Cover):

    #either 1d or 2d filter; angles are assumed to live in [0, 2*pi)
    def __init__(self, n_per_axis=(6, 6), overlap: float = 0.4):
        if np.isscalar(n_per_axis):
            n_per_axis = (int(n_per_axis),)
        self.n_per_axis = tuple(n_per_axis)
        self.overlap = overlap

    def fit(self, f):
        f = np.asarray(f)
        self.centers_ = [np.arange(n) * 2 * np.pi / n for n in self.n_per_axis]
        self.steps_ = [2 * np.pi / n for n in self.n_per_axis]
        return self

    def _axis_member(self, theta, center, step):
        d = (theta - center + np.pi) % (2 * np.pi) - np.pi
        half = 0.5 * self.overlap * step
        return np.abs(d) <= step / 2 + half

    def assign(self, f):
        f = np.asarray(f)
        masks = []  
        for ax, (centers, step) in enumerate(zip(self.centers_, self.steps_)):
            masks.append([self._axis_member(f[:, ax], c, step) for c in centers])
        out = []
        for combo in product(*[range(len(m)) for m in masks]):
            mask = np.ones(f.shape[0], dtype=bool)
            for ax, ci in enumerate(combo):
                mask &= masks[ax][ci]
            idx = np.where(mask)[0]
            if len(idx) > 0:
                out.append(idx)
        return out


# Clustering 

#DBSCAN with precomuted distance matrix, noise points become their own singleton clusters (marked -1)
def dbscan(eps: float = 0.5, min_samples: int = 3):

    def f(idx, D):
        Dsub = D[np.ix_(idx, idx)] #submatrix of distances for points in this window, ix are indices
        lab = DBSCAN(eps=eps, min_samples=min_samples,
                     metric="precomputed").fit_predict(Dsub)
        nxt = lab.max() + 1 if (lab >= 0).any() else 0
        for i in np.where(lab < 0)[0]:   # dbscan groups all isolated points, split them into singletons
            lab[i] = nxt
            nxt += 1
        return lab

    return f


def single_linkage(n_bins: int = 10):
    def f(idx, D):
        Dsub = D[np.ix_(idx, idx)]
        n = len(idx)
        if n == 1:
            return np.array([0])
        condensed = Dsub[np.triu_indices(n, k=1)]
        Z = linkage(condensed, method="single")
        heights = Z[:, 2]
        lo, hi = heights.min(), heights.max()
        if hi == lo:
            return np.zeros(n, dtype=int)
        edges = np.linspace(lo, hi, n_bins + 1)
        counts, _ = np.histogram(heights, bins=edges)
        empty = np.where(counts == 0)[0]
        threshold = edges[empty[0]] if len(empty) else hi
        return fcluster(Z, t=threshold, criterion="distance") - 1
    return f


# Mapper
class Mapper:

    def __init__(self, lens: Lens, cover: Cover, clusterer,
                 min_cluster_size: int = 1):
        self.lens = lens
        self.cover = cover
        self.clusterer = clusterer
        self.min_cluster_size = min_cluster_size

    def fit(self, X, D=None):
        X = np.asarray(X, dtype=float)
        if D is None:
            D = euclidean_distances(X)
        self.X_, self.D_ = X, D

        f = self.lens.transform(X, D)
        self.f_ = f
        self.cover.fit(f)
        windows = self.cover.assign(f)

        nodes = []                       # list of node metadata
        for w_idx in windows:
            if len(w_idx) == 0:
                continue
            labels = self.clusterer(w_idx, D)
            for lab in np.unique(labels):
                members = w_idx[labels == lab]
                if len(members) < self.min_cluster_size:
                    continue                       # drop trivial clusters
                nodes.append({
                    "members": members,
                    "size": int(len(members)),
                    "filter_mean": f[members].mean(axis=0),
                    "centroid": X[members].mean(axis=0),
                })
        self.nodes_ = nodes

        point_to_nodes = {} #point index -> list of node indices
        for nid, nd in enumerate(nodes):
            for m in nd["members"]:
                point_to_nodes.setdefault(int(m), []).append(nid)
        self.point_to_nodes_ = point_to_nodes
        return self

    # to gudhi simplex tree
    def to_simplex_tree(self, max_dim: int = 3):
        import gudhi

        st = gudhi.SimplexTree()
        for nid in range(len(self.nodes_)):
            st.insert([nid])                     
        for nid_list in self.point_to_nodes_.values():
            s = sorted(set(nid_list))
            if len(s) - 1 <= max_dim:
                st.insert(s)                      
            else:                                
                st.insert(s, filtration=0.0)
        return st

    # builds 1d graph
    def to_networkx(self):
        import networkx as nx

        G = nx.Graph()
        for nid, nd in enumerate(self.nodes_):
            G.add_node(nid, size=nd["size"],
                       filter_mean=nd["filter_mean"],
                       centroid=nd["centroid"], members=nd["members"])
        for nid_list in self.point_to_nodes_.values():
            s = sorted(set(nid_list))
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    G.add_edge(s[i], s[j])
        self.graph_ = G
        return G

    def betti_numbers(self, max_dim: int = 3):
        st = self.to_simplex_tree(max_dim=max_dim)
        st.compute_persistence(persistence_dim_max=True)
        return st.betti_numbers()


# visualization helpers

# index -> cords, centeres node cords to the mean of its members cords
def positions_from_coords(mapper: Mapper, coords):
    coords = np.asarray(coords, dtype=float)
    return {nid: coords[nd["members"]].mean(axis=0)
            for nid, nd in enumerate(mapper.nodes_)}


def plot_graph(mapper, ax=None, pos=None, color_by="filter_mean",
               filter_index=0, cmap="jet", size_scale=20.0, edge_color="0.4",
               with_labels=False, colorbar=True):
    import matplotlib.pyplot as plt
    import networkx as nx

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    G = mapper.to_networkx()
    if pos is None:
        pos = nx.spring_layout(G, seed=0)

    if size_scale is None:
        sizes = 300
    else:
        sizes = np.array([G.nodes[n]["size"] for n in G.nodes]) * size_scale
    if color_by == "filter_mean":
        colors = np.array([np.atleast_1d(G.nodes[n]["filter_mean"])[filter_index]
                           for n in G.nodes])
    elif color_by == "size":
        colors = np.array([G.nodes[n]["size"] for n in G.nodes])
    else:
        colors = "tab:blue"

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color)
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes,
                                   node_color=colors, cmap=cmap)
    if with_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
    if colorbar and color_by in ("filter_mean", "size") and nodes is not None:
        plt.colorbar(nodes, ax=ax, shrink=0.7, label=color_by)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


#draws points and graph on top
def overlay_on_points(mapper, coords2d, ax=None, point_color="0.8",
                      point_size=6, **kwargs):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    coords2d = np.asarray(coords2d, dtype=float)
    ax.scatter(coords2d[:, 0], coords2d[:, 1], s=point_size,
               c=point_color, zorder=0)
    pos = positions_from_coords(mapper, coords2d)
    plot_graph(mapper, ax=ax, pos=pos, **kwargs)
    return ax



def plot_complex_3d(mapper, coords3d, ax=None, color_by=None, cmap="jet",
                    point_color="0.85", point_size=2, point_alpha=0.2,
                    node_size=35, face_alpha=0.55, edge_color="0.3",
                    show_points=True, max_dim=2):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

    coords3d = np.asarray(coords3d, dtype=float)
    node_xyz = np.array(list(positions_from_coords(mapper, coords3d).values()))
    idx = 0 if color_by is None else int(color_by)
    node_c = np.array([np.atleast_1d(nd["filter_mean"])[idx]
                       for nd in mapper.nodes_])

    st = mapper.to_simplex_tree(max_dim=max_dim)
    edges = [tuple(s) for s, _ in st.get_simplices() if len(s) == 2]
    tris = [tuple(s) for s, _ in st.get_simplices() if len(s) == 3]

    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect(np.ptp(coords3d, axis=0))

    if show_points:
        ax.scatter(coords3d[:, 0], coords3d[:, 1], coords3d[:, 2],
                   c=point_color, s=point_size, alpha=point_alpha)

    if tris:
        polys = [node_xyz[list(t)] for t in tris]
        fmean = node_c[[list(t) for t in tris]].mean(1)
        facec = plt.get_cmap(cmap)((fmean - node_c.min())
                                   / (np.ptp(node_c) + 1e-9))
        ax.add_collection3d(Poly3DCollection(polys, facecolors=facec,
                                             edgecolors="k", linewidths=0.25,
                                             alpha=face_alpha))

    if edges:
        ax.add_collection3d(Line3DCollection(
            [node_xyz[list(e)] for e in edges],
            colors=edge_color, linewidths=0.6))

    ax.scatter(*node_xyz.T, c=node_c, cmap=cmap, s=node_size, depthshade=False)
    return ax
