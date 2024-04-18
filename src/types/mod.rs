use std::fmt::{Display, Formatter};
use std::str::FromStr;
use anyhow::anyhow;

pub mod point_set;
pub mod query_set;
pub mod ground_truth;

/// Collection of metrics and distance functions that characterize an ANN search.
#[derive(Eq, PartialEq, Hash, Debug, Clone)]
pub enum Metric {
    Hamming,
    Euclidean,
    Cosine,
    InnerProduct,
}

impl Display for Metric {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl FromStr for Metric {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Hamming" => Ok(Metric::Hamming),
            "Euclidean" => Ok(Metric::Euclidean),
            "Cosine" => Ok(Metric::Cosine),
            "InnerProduct" => Ok(Metric::InnerProduct),
            _ => Err(anyhow!("invalid metric"))
        }
    }
}
