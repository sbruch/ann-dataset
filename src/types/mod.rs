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
            "Hamming" | "hamming" => Ok(Metric::Hamming),
            "Euclidean" | "euclidean" => Ok(Metric::Euclidean),
            "Cosine" | "cosine" => Ok(Metric::Cosine),
            "InnerProduct" | "inner-product" | "dot-product" => Ok(Metric::InnerProduct),
            _ => Err(anyhow!("Metric must be one of [hamming|euclidean|cosine|inner-product]"))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;
    use crate::Metric;

    #[test]
    fn test_from_str() {
        assert_eq!(Metric::Cosine, Metric::from_str("cosine").unwrap());
        assert_eq!(Metric::Cosine, Metric::from_str("Cosine").unwrap());
        assert_eq!(Metric::InnerProduct, Metric::from_str("inner-product").unwrap());
        assert_eq!(Metric::InnerProduct, Metric::from_str("InnerProduct").unwrap());
        assert_eq!(Metric::InnerProduct, Metric::from_str("dot-product").unwrap());
        assert_eq!(Metric::Euclidean, Metric::from_str("euclidean").unwrap());
        assert_eq!(Metric::Euclidean, Metric::from_str("Euclidean").unwrap());
        assert!(Metric::from_str("foo").is_err());
    }
}
