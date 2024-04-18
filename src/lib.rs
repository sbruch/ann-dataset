//! A lightweight research library for managing Approximate Nearest Neighbor search datasets.
//!
//! It offers the following features:
//!  * Storage of dense, sparse, and dense-sparse vector sets;
//!  * Storage of query sets with ground-truth (i.e., exact nearest neighbors) according to
//!    different metrics;
//!  * Basic functionality such as computing recall given a retrieved set; and,
//!  * Serialization into and deserialization from HDF5 file format.

mod data;
mod types;
mod io;

pub use crate::data::AnnDataset;
pub use crate::data::dataset::InMemoryAnnDataset;

pub use crate::types::Metric;
pub use crate::types::point_set::PointSet;
pub use crate::types::query_set::QuerySet;
pub use crate::types::ground_truth::GroundTruth;

pub use crate::io::Hdf5Serialization;
pub use crate::io::Hdf5File;
