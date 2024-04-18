use ndarray::{Array2, Axis};
use sprs::CsMat;
use anyhow::{anyhow, Result};

/// A set of points (dense, sparse, or both) represented as a matrix,
/// where each row corresponds to a single vector.
pub struct PointSet<DataType: Clone> {
    dense: Option<Array2<DataType>>,
    sparse: Option<CsMat<DataType>>,
}

impl<DataType: Clone> PointSet<DataType> {
    /// Creates a point set.
    ///
    /// Returns an error if both `dense` and `sparse` vector sets are empty, or if they are both
    /// provided, the number of rows of the `dense` and `sparse` sets do not match.
    pub fn new(dense: Option<Array2<DataType>>, sparse: Option<CsMat<DataType>>) -> Result<PointSet<DataType>> {
        if dense.is_none() && sparse.is_none() {
            return Err(anyhow!("Both dense and sparse sets are empty."));
        }
        if dense.is_some() && sparse.is_some() {
            let dense = dense.as_ref().unwrap();
            let sparse = sparse.as_ref().unwrap();
            if dense.nrows() != sparse.rows() {
                return Err(anyhow!("There are {} dense vectors but {} sparse vectors!", dense.nrows(), sparse.rows()));
            }
        }
        Ok(PointSet { dense, sparse })
    }

    /// Returns the number of points in the point set.
    pub fn num_points(&self) -> usize {
        if let Some(dense) = self.dense.as_ref() {
            return dense.nrows();
        }
        if let Some(sparse) = self.sparse.as_ref() {
            return sparse.rows();
        }
        0_usize
    }

    /// Returns the dense sub-vectors.
    pub fn get_dense(&self) -> Option<&Array2<DataType>> { self.dense.as_ref() }

    /// Returns the sparse sub-vectors.
    pub fn get_sparse(&self) -> Option<&CsMat<DataType>> { self.sparse.as_ref() }

    /// Selects a subset of points with the given ids.
    pub fn select(&self, ids: &[usize]) -> PointSet<DataType> {
        let dense = match self.dense.as_ref() {
            None => { None }
            Some(dense) => { Some(dense.select(Axis(0), ids)) }
        };

        let sparse = match self.sparse.as_ref() {
            None => { None }
            Some(sparse) => {
                let mut nnzs = ids.iter().map(|&index| {
                    sparse.indptr().index(index + 1) - sparse.indptr().index(index)
                }).collect::<Vec<_>>();

                let indices = ids.iter().enumerate().map(|(i, &index)| {
                    let begin = sparse.indptr().index(index);
                    let end = begin + nnzs[i];
                    sparse.indices()[begin..end].to_vec()
                }).flatten().collect::<Vec<_>>();

                let data = ids.iter().enumerate().map(|(i, &index)| {
                    let begin = sparse.indptr().index(index);
                    let end = begin + nnzs[i];
                    sparse.data()[begin..end].to_vec()
                }).flatten().collect::<Vec<_>>();

                let mut acc = 0_usize;
                for x in &mut nnzs {
                    acc += *x;
                    *x = acc;
                }
                nnzs.insert(0, 0);

                Some(CsMat::new((ids.len(), sparse.shape().1), nnzs, indices, data))
            }
        };

        PointSet { dense, sparse }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, Axis};
    use sprs::{CsMat, TriMat};
    use crate::types::point_set::PointSet;

    #[test]
    fn test_new() {
        let dense = Array2::<f32>::eye(5);

        let mut sparse = TriMat::new((4, 4));
        sparse.add_triplet(0, 0, 3.0_f32);
        sparse.add_triplet(1, 2, 2.0);
        sparse.add_triplet(3, 0, -2.0);
        let sparse: CsMat<_> = sparse.to_csr();

        assert!(PointSet::<f32>::new(None, None).is_err());
        assert!(PointSet::new(Some(dense.clone()), None).is_ok());
        assert!(PointSet::new(None, Some(sparse.clone())).is_ok());
        assert!(PointSet::new(Some(dense.clone()), Some(sparse.clone())).is_err());

        let dense = Array2::<f32>::eye(4);
        assert!(PointSet::new(Some(dense.clone()), Some(sparse.clone())).is_ok());
    }

    #[test]
    fn test_subset() {
        let dense = Array2::<f32>::eye(10);

        let mut sparse = TriMat::new((10, 4));
        sparse.add_triplet(0, 0, 3.0_f32);
        sparse.add_triplet(1, 2, 2.0);
        sparse.add_triplet(3, 0, -2.0);
        sparse.add_triplet(9, 2, 3.4);
        let sparse: CsMat<_> = sparse.to_csr();

        let point_set = PointSet::new(Some(dense.clone()), Some(sparse.clone()));
        assert!(point_set.is_ok());
        let point_set = point_set.unwrap();

        let subset = point_set.select(&[9]);
        assert_eq!(subset.get_dense().unwrap(), dense.select(Axis(0), &[9]));

        let mut sparse_subset = TriMat::new((1, 4));
        sparse_subset.add_triplet(0, 2, 3.4);
        let sparse_subset: CsMat<_> = sparse_subset.to_csr();
        assert_eq!(subset.get_sparse().unwrap(), &sparse_subset);

        let subset = point_set.select(&[0, 3, 9]);
        assert_eq!(subset.get_dense().unwrap(), dense.select(Axis(0), &[0, 3, 9]));

        let mut sparse_subset = TriMat::new((3, 4));
        sparse_subset.add_triplet(0, 0, 3.0_f32);
        sparse_subset.add_triplet(1, 0, -2.0);
        sparse_subset.add_triplet(2, 2, 3.4);
        let sparse_subset: CsMat<_> = sparse_subset.to_csr();
        assert_eq!(subset.get_sparse().unwrap(), &sparse_subset);
    }
}
