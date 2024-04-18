pub mod point_set;
pub mod query_set;
pub mod ground_truth;

/// Collection of metrics and distance functions that characterize an ANN search.
#[derive(Eq, PartialEq, Hash, Debug)]
pub enum Metric {
    Euclidean,
    Cosine,
    InnerProduct,
}
