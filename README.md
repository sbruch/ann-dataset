A lightweight research library for managing Approximate Nearest Neighbor search datasets.

It offers the following features:
  * Storage of dense, sparse, and dense-sparse vector sets;
  * Storage of query sets with ground-truth (i.e., exact nearest neighbors) according to different metrics;
  * Basic functionality such as computing recall given a retrieved set; and,
  * Serialization into and deserialization from HDF5 file format.

Find out more on [crates.io](https://docs.rs/crate/ann_dataset/).

## Example usage

It is straightforward to read an ANN dataset. The code snippet
below gives a concise example.

```rust
use ann_dataset::{AnnDataset, Hdf5File, InMemoryAnnDataset, Metric, 
                  PointSet, QuerySet, GroundTruth};

// Load the dataset.
let dataset = InMemoryAnnDataset::<f32>::read(path_to_hdf5)
    .expect("Failed to read the dataset.");

// Get a reference to the data points.
let data_points: &PointSet<_> = dataset.get_data_points();

// Get the test query set.
let test: &QuerySet<_> = dataset.get_test_query_set()
    .expect("Failed to load test query set.");
let test_queries: &PointSet<_> = test.get_points();
let gt: &GroundTruth = test.get_ground_truth(Metric::InnerProduct)
    .expect("Failed to load ground truth for InnerProduct search.");

// Compute recall, assuming `retrieved_set` is &[Vec<usize>],
// where the `i`-th entry is a list of ids of retrieved points
// for the `i`-th query.
let recall = gt.mean_recall(retrieved_set);
```
