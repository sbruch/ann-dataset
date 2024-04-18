use ndarray::Array2;

pub mod point_set;
pub mod query_set;

/// Collection of metrics and distance functions that characterize an ANN search.
#[derive(Eq, PartialEq, Hash, Debug)]
pub enum Metric {
    Euclidean,
    Cosine,
    InnerProduct,
}

#[doc(hidden)]
#[allow(dead_code)]
pub(crate) type GroundTruth = Array2<usize>;
