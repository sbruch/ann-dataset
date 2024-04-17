use ndarray::Array2;
use sprs::CsMat;

/// A set of vectors represented as a matrix, where each row corresponds with a single vector.
pub enum VectorSet<DataType> {
    /// Dense vector sets are stored as an `ndarray::Array2` object.
    Dense(Array2<DataType>),

    /// Sparse vector sets are stored as a `sprs::CsMat` object.
    Sparse(CsMat<DataType>),
}

pub(crate) type GroundTruth = Array2<usize>;
