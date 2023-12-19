# classix.jl

[![License: MIT](https://anaconda.org/conda-forge/classixclustering/badges/license.svg)](https://github.com/nla-group/classix/blob/master/LICENSE)

__classix.jl__ is Julia package for clustering algorithm  __CLASSIX__. __CLASSIX__ is a fast, memory-efficient, and explainable clustering algorithm. Here are a few highlights:

- Ability to cluster low and high-dimensional data of arbitrary shape efficiently
- Ability to detect and deal with outliers in the data
- Ability to provide textual and visual explanations for the clusters
- Full reproducibility of all tests in the accompanying paper
- Support of Cython compilation

__CLASSIX__ is a contrived acronym of *CLustering by Aggregation with Sorting-based Indexing* and the letter *X* for *explainability*. 

## Install

__classix.jl__ is registered Julia package, one can simply install via:

```julia
pkg> add ABBAj
```

##   Quick start

Here is an example of CLASSIX clustering a synthetic dataset: 

```julia
using Classix

data = [8.0391 11.3790 9.7221;
9.8023    8.9418   10.7015;
-9.7180  -10.2991  -10.8314;
-9.9665   -9.9771  -10.9792;
8.7922    9.5314    7.9482;
12.9080    9.7275    9.6462;
10.8252   11.0984    9.1764;
-11.5771   -8.8725  -11.7502;
-9.4920   -9.6498  -10.2857;
-11.3337  -10.2620  -11.1564]

labels, explain, out = classix(data, radius=0.2, minPts=1, merge_tiny_groups=true)

```


## Contribution
Any form of contribution is welcome. We particularly welcome the contribution of new `demos` in the form of Jupyter Notebooks. Feel free to post issues and pull requests if you want to assist in documentation or code. To contribute, please fork the project and pull a request for your changes. We will strive to work through any issues and requests and get your code merged into the main branch. Contributors will be acknowledged in the release notes. 



### Reference

Xinye, C., Güttel, S. Fast and explainable clustering based on sorting. arXiv:2202.01456, 1–25 (2022). https://arxiv.org/abs/2202.01456
