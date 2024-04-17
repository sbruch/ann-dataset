//! A lightweight wrapper around the HDF5 file format that adds support for dense
//! and sparse vector sets. This research library is tailored towards the Approximate
//! Nearest Neighbor search use case.
//!
//! ## Creating an ANN dataset
//!
//! ```no_run
//! use ann_dataset::{VectorSet, AnnDataset};
//!
//! let dense_set = ndarray::Array2::<f32>::eye(5);
//!
//! let mut sparse_set = sprs::TriMat::new((4, 4));
//! sparse_set.add_triplet(0, 0, 3.0_f32);
//! sparse_set.add_triplet(1, 2, 2.0);
//! sparse_set.add_triplet(3, 0, - 2.0);
//! let sparse_set: sprs::CsMat<_ > = sparse_set.to_csr();
//!
//! let mut dataset = AnnDataset::<f32>::new();
//! dataset.add("dense", VectorSet::Dense(dense_set)).unwrap();
//! dataset.add("sparse", VectorSet::Sparse(sparse_set.clone())).unwrap();
//!
//! dataset.write("dataset.hdf5").expect("failed to serialize dataset");
//! ```
//!
//! ## Reading an ANN dataset
//! ```no_run
//! use ann_dataset::{VectorSet, AnnDataset};
//!
//! let dataset = AnnDataset::<f32>::load("dataset.hdf5").expect("failed to load dataset");
//! println!("{}", dataset);
//!
//! let dense_set = dataset.read("dense_set").expect("failed to read vector set");
//! let sparse_set = dataset.read("sparse_set").expect("failed to read vector set");
//! ```

pub(crate) mod data;

pub use crate::data::types::VectorSet;
pub use crate::data::dataset::AnnDataset;
