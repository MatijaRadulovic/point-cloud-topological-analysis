# Point Cloud Topological Analysis

This project is an illustrative implementation of the **Mapper algorithm**, a method from topological data analysis used to summarize the shape of high-dimensional data sets.

The implementation is based on the paper:

**G. Singh, F. Mémoli, and G. Carlsson**,  
*Topological Methods for the Analysis of High Dimensional Data Sets and 3D Object Recognition*,  
Eurographics Symposium on Point-Based Graphics, 2007, pp. 91–100.  
DOI: `10.2312/SPBG/SPBG07/091-100`

The goal of the project is to demonstrate how a point cloud can be transformed into a graph or simplicial complex that captures important topological features of the data, such as connected components, loops, and higher-dimensional structures.

---

## Project Overview

The Mapper algorithm gives a compressed topological representation of a data set. Instead of analyzing the full point cloud directly, Mapper builds a graph or simplicial complex by combining three main ideas:

1. **Filter functions**, also called lenses, which assign numerical values to data points.
2. **Covers** of the filter image, usually by overlapping intervals, rectangles, hexagons, or circular regions.
3. **Clustering** inside the inverse image of each cover element.

Each cluster becomes a node in the Mapper graph. Two nodes are connected if the corresponding clusters share at least one original data point. In this way, the resulting graph describes the global shape of the data.

---

## Mathematical Idea

Let `X` be a finite point cloud and let `f: X -> R^d` be a filter function.

The Mapper construction follows these steps:

1. Compute the filter values `f(X)`.
2. Cover the image `f(X)` by overlapping sets.
3. Pull each cover element back to the original data set using the inverse image `f^{-1}(U)`.
4. Cluster the points inside each inverse image.
5. Create one node for each cluster.
6. Connect two nodes if their clusters overlap in the original data.

The overlap is essential. Without overlapping cover elements, Mapper would usually produce disconnected local summaries instead of a meaningful global structure.

---

## Implemented Features

The project contains a compact but flexible implementation of Mapper.

Implemented filter functions include:

* coordinate projection,
* distance from a fixed point,
* Euclidean norm,
* PCA projection,
* density estimate,
* eccentricity,
* graph Laplacian eigenvectors,
* precomputed filter values.

Implemented covers include:

* interval cover for one-dimensional filters,
* rectangle cover for two-dimensional filters,
* hexagon cover for two-dimensional filters,
* circular cover for angular data.

Implemented clustering methods include:

* DBSCAN with a precomputed distance matrix,
* single-linkage clustering.

The implementation can also convert the Mapper output into:

* a NetworkX graph,
* a GUDHI simplex tree,

and compute:

* Betti numbers of the Mapper complex.

---

## Project Structure

```text
.
├── mapper.py              # Core Mapper implementation
├── mapper_demo.ipynb      # Notebook with demonstrations and explanations
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .gitignore             # Files ignored by Git
```

Optional data files, such as 3D mesh files, may be placed in a local `data/` directory. The notebook is written so that examples depending on such files can be skipped if the data is not available.

---

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook mapper_demo.ipynb
```

Then run the notebook cells from top to bottom.

The notebook demonstrates the Mapper algorithm on several examples, including:

* a noisy circle,
* different filter functions,
* different covers,
* interactive parameter changes,
* two-dimensional filters,
* a torus example,
* optional 3D mesh data.

---

## Running a Basic Import Test

To check that the main implementation can be imported correctly, run:

```bash
python -m py_compile mapper.py
python -c "import mapper as mp; print('Mapper import OK')"
```

---

## Example Usage

A minimal example of using the implementation is:

```python
import numpy as np
import mapper as mp

# Generate a noisy circle
theta = np.linspace(0, 2 * np.pi, 300, endpoint=False)
X = np.column_stack([np.cos(theta), np.sin(theta)])
X += 0.05 * np.random.randn(*X.shape)

# Define the Mapper pipeline
lens = mp.Lens(mp.coordinate(1), names=["height"])
cover = mp.IntervalCover(n_intervals=8, overlap=0.4)
clusterer = mp.dbscan(eps=0.25, min_samples=3)

mapper = mp.Mapper(lens=lens, cover=cover, clusterer=clusterer)
mapper.fit(X)

# Convert to graph
G = mapper.to_networkx()

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
print("Betti numbers:", mapper.betti_numbers())
```

For a noisy circle, a good Mapper construction should often produce a graph that reflects the circular structure of the data.

---

## Main Components

### Lens

The `Lens` class stores one or more filter functions and applies them to the data set. A filter may be one-dimensional or multidimensional.

Examples:

```python
mp.coordinate(0)
mp.coordinate(1)
mp.l2norm()
mp.distance_from_point([0, 0])
mp.pca(n_components=2)
mp.density()
mp.eccentricity()
mp.graph_laplacian()
```

### Cover

A cover divides the range of the filter function into overlapping regions.

Examples:

```python
mp.IntervalCover(n_intervals=10, overlap=0.3)
mp.RectangleCover(n_per_axis=(8, 8), overlap=0.3)
mp.HexagonCover(n=8, overlap=0.3)
mp.CircularCover(n_per_axis=(6, 6), overlap=0.4)
```

### Clustering

Inside each inverse image of a cover element, the algorithm clusters the corresponding points.

Examples:

```python
mp.dbscan(eps=0.5, min_samples=3)
mp.single_linkage(n_bins=10)
```

### Mapper

The `Mapper` class combines the lens, cover, and clustering method.

```python
mapper = mp.Mapper(
    lens=lens,
    cover=cover,
    clusterer=clusterer,
    min_cluster_size=1
)
mapper.fit(X)
```

After fitting, the result can be converted to a graph:

```python
G = mapper.to_networkx()
```

or to a simplex tree:

```python
st = mapper.to_simplex_tree()
```

Betti numbers can be computed using:

```python
mapper.betti_numbers()
```

---

## Visualization

The project includes helper functions for visualizing Mapper graphs.

For a two-dimensional point cloud:

```python
mp.overlay_on_points(mapper, X)
```

For drawing only the Mapper graph:

```python
mp.plot_graph(mapper)
```

For three-dimensional data:

```python
mp.plot_complex_3d(mapper, coords3d)
```

These visualizations are used in the notebook to show how the Mapper graph changes when different filters, covers, and clustering parameters are chosen.

---

## Interpretation of Mapper Output

The Mapper graph is not just a visualization. It is a topological summary of the data.

Typical interpretations are:

* connected components in the graph may correspond to separated regions in the data,
* loops in the graph may indicate circular or cyclic structure,
* branching may indicate several different geometric or statistical regimes,
* large nodes correspond to clusters containing many data points,
* node colors can show average filter values.

For example, when Mapper is applied to a point cloud sampled from a circle, the output should ideally contain one dominant loop. This agrees with the fact that a circle has one connected component and one one-dimensional hole.

---

## Notes on Parameters

The Mapper output depends strongly on several parameters:

* the choice of filter function,
* the number of cover intervals or regions,
* the amount of overlap,
* the clustering method,
* clustering parameters such as `eps` and `min_samples` for DBSCAN.

A small number of intervals may produce an overly simple graph. A large number of intervals may produce a fragmented graph. Similarly, too little overlap may disconnect the Mapper graph, while too much overlap may create too many edges.

Because of this, Mapper should usually be explored interactively with several parameter choices.

---

## Dependencies

The main dependencies are:

* `numpy`
* `scipy`
* `networkx`
* `matplotlib`
* `scikit-learn`
* `gudhi`
* `trimesh`
* `ipywidgets`
* `notebook`
* `pytest`

They are listed in `requirements.txt`.

---

## Testing

At minimum, the implementation can be checked by compiling and importing the main module:

```bash
python -m py_compile mapper.py
python -c "import mapper as mp; print('Mapper import OK')"
```

If test files are added later, they can be run with:

```bash
pytest
```

---

## References

This project is based on:

```bibtex
@inproceedings{singh2007mapper,
  author    = {Singh, Gurjeet and M{\'e}moli, Facundo and Carlsson, Gunnar},
  title     = {Topological Methods for the Analysis of High Dimensional Data Sets and 3D Object Recognition},
  booktitle = {Eurographics Symposium on Point-Based Graphics},
  pages     = {91--100},
  year      = {2007},
  doi       = {10.2312/SPBG/SPBG07/091-100}
}
```
