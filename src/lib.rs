//! A lightweight research library for managing Approximate Nearest Neighbor search datasets.
//!
//! It offers the following features:
//!  * Storage of dense, sparse, and dense-sparse vector sets;
//!  * Storage of query sets with ground-truth (i.e., exact nearest neighbors) according to
//!    different metrics;
//!  * Basic functionality such as computing recall given a retrieved set; and,
//!  * Serialization into and deserialization from HDF5 file format.
//!
//! ## Example usage
//! It is straightforward to read an ANN dataset. The code snippet
//! below gives a concise example.
//!
//! ```no_run
//! use ann_dataset::{AnnDataset, Hdf5File, InMemoryAnnDataset, Metric,
//!                   PointSet, QuerySet, GroundTruth};
//!
//! // Load the dataset.
//! let dataset = InMemoryAnnDataset::<f32>::read("")
//!     .expect("Failed to read the dataset.");
//!
//! // Get a reference to the data points.
//! let data_points: &PointSet<_> = dataset.get_data_points();
//!
//! // Get the test query set.
//! let test: &QuerySet<_> = dataset.get_test_query_set()
//!     .expect("Failed to load test query set.");
//! let test_queries: &PointSet<_> = test.get_points();
//! let gt: &GroundTruth = test.get_ground_truth(&Metric::InnerProduct)
//!     .expect("Failed to load ground truth for InnerProduct search.");
//!
//! // Compute recall, where the argument is &[Vec<usize>],
//! // where the `i`-th entry is a list of ids of retrieved points
//! // for the `i`-th query.
//! let recall = gt.mean_recall(&[]);
//! ```
mod data;
mod io;
mod types;

pub use crate::data::in_memory_dataset::{
    InMemoryAnnDataset, PointSetIterator, PointSetMutableIterator,
};
pub use crate::data::AnnDataset;

pub use crate::types::ground_truth::GroundTruth;
pub use crate::types::point_set::PointSet;
pub use crate::types::query_set::QuerySet;
pub use crate::types::Metric;

pub use crate::io::Hdf5File;
pub use crate::io::Hdf5Serialization;
