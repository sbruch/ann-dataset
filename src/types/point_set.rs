use crate::Hdf5Serialization;
use anyhow::{anyhow, Result};
use hdf5::{Group, H5Type};
use linfa_linalg::norm::Norm;
use ndarray::{Array1, Array2, Axis, Zip};
use sprs::CsMat;
use std::fmt::{Display, Formatter};
use std::iter::zip;

const DENSE: &str = "dense";
const SPARSE: &str = "sparse";
const SPARSE_INDPTR: &str = "indptr";
const SPARSE_INDICES: &str = "indices";
const SPARSE_DATA: &str = "data";
const SPARSE_SHAPE: &str = "shape";

/// A set of points (dense, sparse, or both) represented as a matrix,
/// where each row corresponds to a single vector.
#[derive(Eq, PartialEq, Debug, Clone)]
pub struct PointSet<DataType: Clone> {
    dense: Option<Array2<DataType>>,
    sparse: Option<CsMat<DataType>>,
}

impl<DataType: Clone> PointSet<DataType> {
    /// Creates a point set.
    ///
    /// Returns an error if both `dense` and `sparse` vector sets are empty, or if they are both
    /// provided, the number of rows of the `dense` and `sparse` sets do not match.
    pub fn new(
        dense: Option<Array2<DataType>>,
        sparse: Option<CsMat<DataType>>,
    ) -> Result<PointSet<DataType>> {
        if dense.is_none() && sparse.is_none() {
            return Err(anyhow!("Both dense and sparse sets are empty."));
        }
        if dense.is_some() && sparse.is_some() {
            let dense = dense.as_ref().unwrap();
            let sparse = sparse.as_ref().unwrap();
            if dense.nrows() != sparse.rows() {
                return Err(anyhow!(
                    "There are {} dense vectors but {} sparse vectors!",
                    dense.nrows(),
                    sparse.rows()
                ));
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

    /// Returns the number of dense dimensions.
    pub fn num_dense_dimensions(&self) -> usize {
        if let Some(dense) = self.dense.as_ref() {
            return dense.ncols();
        }
        0_usize
    }

    /// Returns the number of sparse dimensions.
    pub fn num_sparse_dimensions(&self) -> usize {
        if let Some(sparse) = self.sparse.as_ref() {
            return sparse.cols();
        }
        0_usize
    }

    /// Returns the total number of dimensions.
    pub fn num_dimensions(&self) -> usize {
        self.num_sparse_dimensions() + self.num_dense_dimensions()
    }

    /// Returns the dense sub-vectors.
    pub fn get_dense(&self) -> Option<&Array2<DataType>> {
        self.dense.as_ref()
    }

    /// Returns the sparse sub-vectors.
    pub fn get_sparse(&self) -> Option<&CsMat<DataType>> {
        self.sparse.as_ref()
    }

    /// Selects a subset of points with the given ids.
    pub fn select(&self, ids: &[usize]) -> PointSet<DataType> {
        let dense = self.dense.as_ref().map(|dense| dense.select(Axis(0), ids));

        let sparse = match self.sparse.as_ref() {
            None => None,
            Some(sparse) => {
                let mut nnzs = ids
                    .iter()
                    .map(|&index| sparse.indptr().index(index + 1) - sparse.indptr().index(index))
                    .collect::<Vec<_>>();

                let indices = ids
                    .iter()
                    .enumerate()
                    .flat_map(|(i, &index)| {
                        let begin = sparse.indptr().index(index);
                        let end = begin + nnzs[i];
                        sparse.indices()[begin..end].to_vec()
                    })
                    .collect::<Vec<_>>();

                let data = ids
                    .iter()
                    .enumerate()
                    .flat_map(|(i, &index)| {
                        let begin = sparse.indptr().index(index);
                        let end = begin + nnzs[i];
                        sparse.data()[begin..end].to_vec()
                    })
                    .collect::<Vec<_>>();

                let mut acc = 0_usize;
                for x in &mut nnzs {
                    acc += *x;
                    *x = acc;
                }
                nnzs.insert(0, 0);

                Some(CsMat::new(
                    (ids.len(), sparse.shape().1),
                    nnzs,
                    indices,
                    data,
                ))
            }
        };

        PointSet { dense, sparse }
    }
}

impl PointSet<f32> {
    /// Returns the L2 norm of the points.
    pub fn l2_norm(&self) -> Array1<f32> {
        let dense_l2_squared = if let Some(dense) = self.dense.as_ref() {
            Array1::from(
                dense
                    .axis_iter(Axis(0))
                    .map(|point| point.norm_l2().powi(2))
                    .collect::<Vec<_>>(),
            )
        } else {
            Array1::<f32>::zeros(self.num_points())
        };

        let sparse_l2_squared = if let Some(sparse) = self.sparse.as_ref() {
            Array1::from(
                sparse
                    .outer_iterator()
                    .map(|point| point.l2_norm().powi(2))
                    .collect::<Vec<_>>(),
            )
        } else {
            Array1::<f32>::zeros(self.num_points())
        };

        let mut l2_norm = dense_l2_squared + sparse_l2_squared;
        l2_norm.mapv_inplace(|v| v.sqrt());
        l2_norm
    }

    /// Normalizes all points by their L2 norm and modifies the `PointSet` in place.
    pub fn l2_normalize_inplace(&mut self) {
        let norms = self.l2_norm();
        if let Some(dense) = self.dense.as_mut() {
            Zip::from(norms.view())
                .and(dense.axis_iter_mut(Axis(0)))
                .par_for_each(|&norm, mut point| {
                    point.mapv_inplace(|x| x / norm);
                });
        }
        if let Some(sparse) = self.sparse.as_mut() {
            zip(norms.iter(), sparse.outer_iterator_mut()).for_each(|(&norm, mut point)| {
                point.map_inplace(|&x| x / norm);
            });
        }
    }
}

impl<DataType: Clone + H5Type> Hdf5Serialization for PointSet<DataType> {
    type Object = PointSet<DataType>;

    fn serialize(&self, group: &mut Group) -> Result<()> {
        if let Some(dense) = self.dense.as_ref() {
            let dataset = group
                .new_dataset::<DataType>()
                .shape(dense.shape())
                .create(format!("{}-{}", Self::label(), DENSE).as_str())?;
            dataset.write(dense)?;
        }

        if let Some(sparse) = self.sparse.as_ref() {
            let group = group.create_group(format!("{}-{}", Self::label(), SPARSE).as_str())?;
            let shape = group.new_attr::<usize>().shape(2).create(SPARSE_SHAPE)?;
            shape.write(&[sparse.shape().0, sparse.shape().1])?;

            let indptr = group
                .new_dataset::<usize>()
                .shape(sparse.indptr().len())
                .create(SPARSE_INDPTR)?;
            indptr.write(sparse.indptr().as_slice().unwrap())?;

            let indices = group
                .new_dataset::<usize>()
                .shape(sparse.indices().len())
                .create(SPARSE_INDICES)?;
            indices.write(sparse.indices())?;

            let data = group
                .new_dataset::<DataType>()
                .shape(sparse.data().len())
                .create(SPARSE_DATA)?;
            data.write(sparse.data())?;
        }
        Ok(())
    }

    fn deserialize(group: &Group) -> Result<Self::Object> {
        let dataset = group.dataset(format!("{}-{}", Self::label(), DENSE).as_str());
        let dense = match dataset {
            Ok(dataset) => {
                let vectors: Vec<DataType> = dataset.read_raw::<DataType>()?;
                let num_dimensions: usize = dataset.shape()[1];
                let vector_count = vectors.len() / num_dimensions;
                Some(Array2::from_shape_vec(
                    (vector_count, num_dimensions),
                    vectors,
                )?)
            }
            Err(_) => None,
        };

        let sparse_group = group.group(format!("{}-{}", Self::label(), SPARSE).as_str());
        let sparse = match sparse_group {
            Ok(sparse_group) => {
                let shape = sparse_group.attr(SPARSE_SHAPE)?.read_raw::<usize>()?;
                if shape.len() != 2 {
                    return Err(anyhow!(
                        "Corrupt shape for sparse dataset '{}'",
                        group.name()
                    ));
                }

                let indptr = sparse_group.dataset(SPARSE_INDPTR)?.read_raw::<usize>()?;
                let indices = sparse_group.dataset(SPARSE_INDICES)?.read_raw::<usize>()?;
                let data: Vec<DataType> =
                    sparse_group.dataset(SPARSE_DATA)?.read_raw::<DataType>()?;
                Some(CsMat::new((shape[0], shape[1]), indptr, indices, data))
            }
            Err(_) => None,
        };

        Ok(PointSet { dense, sparse })
    }

    fn label() -> String {
        "point-set".to_string()
    }
}

impl<DataType: Clone> Display for PointSet<DataType> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let dense = match self.dense.as_ref() {
            None => "is empty".to_string(),
            Some(dense) => {
                format!("has shape [{}, {}]", dense.shape()[0], dense.shape()[1])
            }
        };

        let sparse = match self.sparse.as_ref() {
            None => "is empty".to_string(),
            Some(sparse) => {
                format!("has shape [{}, {}]", sparse.rows(), sparse.cols())
            }
        };

        write!(f, "Dense set {}; Sparse set {}", dense, sparse)
    }
}

#[cfg(test)]
mod tests {
    use crate::types::point_set::PointSet;
    use crate::Hdf5Serialization;
    use approx_eq::assert_approx_eq;
    use hdf5::File;
    use ndarray::{Array2, Axis};
    use sprs::{CsMat, TriMat};
    use std::iter::zip;
    use tempdir::TempDir;

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
        assert_eq!(
            subset.get_dense().unwrap(),
            dense.select(Axis(0), &[0, 3, 9])
        );

        let mut sparse_subset = TriMat::new((3, 4));
        sparse_subset.add_triplet(0, 0, 3.0_f32);
        sparse_subset.add_triplet(1, 0, -2.0);
        sparse_subset.add_triplet(2, 2, 3.4);
        let sparse_subset: CsMat<_> = sparse_subset.to_csr();
        assert_eq!(subset.get_sparse().unwrap(), &sparse_subset);
    }

    #[test]
    fn test_num_dimensions() {
        let dense = Array2::<f32>::eye(10);

        let mut sparse = TriMat::new((10, 4));
        sparse.add_triplet(0, 0, 3.0_f32);
        sparse.add_triplet(1, 2, 2.0);
        sparse.add_triplet(3, 0, -2.0);
        let sparse: CsMat<_> = sparse.to_csr();

        let point_set = PointSet::new(Some(dense), Some(sparse)).unwrap();
        assert_eq!(14, point_set.num_dimensions());
        assert_eq!(10, point_set.num_dense_dimensions());
        assert_eq!(4, point_set.num_sparse_dimensions());
    }

    #[test]
    fn test_hdf5() {
        let dense = Array2::<f32>::eye(10);

        let mut sparse = TriMat::new((10, 4));
        sparse.add_triplet(0, 0, 3.0_f32);
        sparse.add_triplet(1, 2, 2.0);
        sparse.add_triplet(3, 0, -2.0);
        let sparse: CsMat<_> = sparse.to_csr();

        let point_set = PointSet::new(Some(dense), Some(sparse)).unwrap();

        let dir = TempDir::new("pointset_test_hdf5").unwrap();
        let path = dir.path().join("ann-dataset.hdf5");
        let path = path.to_str().unwrap();
        let hdf5 = File::create(path).unwrap();

        let mut group = hdf5.group("/").unwrap();
        assert!(point_set.serialize(&mut group).is_ok());
        let point_set_copy = PointSet::<f32>::deserialize(&group).unwrap();
        assert_eq!(&point_set, &point_set_copy);

        let mut group = group.create_group("/nested").unwrap();
        assert!(point_set.serialize(&mut group).is_ok());
        let point_set_copy = PointSet::<f32>::deserialize(&group).unwrap();
        assert_eq!(&point_set, &point_set_copy);
    }

    #[test]
    fn test_hdf5_dense() {
        let dense = Array2::<f32>::eye(10);
        let point_set = PointSet::new(Some(dense), None).unwrap();

        let dir = TempDir::new("pointset_test_hdf5").unwrap();
        let path = dir.path().join("ann-dataset.hdf5");
        let path = path.to_str().unwrap();
        let hdf5 = File::create(path).unwrap();

        let mut group = hdf5.group("/").unwrap();
        assert!(point_set.serialize(&mut group).is_ok());
        let point_set_copy = PointSet::<f32>::deserialize(&group).unwrap();
        assert_eq!(&point_set, &point_set_copy);

        let mut group = group.create_group("/nested").unwrap();
        assert!(point_set.serialize(&mut group).is_ok());
        let point_set_copy = PointSet::<f32>::deserialize(&group).unwrap();
        assert_eq!(&point_set, &point_set_copy);
    }

    #[test]
    fn test_hdf5_sparse() {
        let mut sparse = TriMat::new((10, 4));
        sparse.add_triplet(0, 0, 3.0_f32);
        sparse.add_triplet(1, 2, 2.0);
        sparse.add_triplet(3, 0, -2.0);
        let sparse: CsMat<_> = sparse.to_csr();

        let point_set = PointSet::new(None, Some(sparse)).unwrap();

        let dir = TempDir::new("pointset_test_hdf5").unwrap();
        let path = dir.path().join("ann-dataset.hdf5");
        let path = path.to_str().unwrap();
        let hdf5 = File::create(path).unwrap();

        let mut group = hdf5.group("/").unwrap();
        assert!(point_set.serialize(&mut group).is_ok());
        let point_set_copy = PointSet::<f32>::deserialize(&group).unwrap();
        assert_eq!(&point_set, &point_set_copy);

        let mut group = group.create_group("/nested").unwrap();
        assert!(point_set.serialize(&mut group).is_ok());
        let point_set_copy = PointSet::<f32>::deserialize(&group).unwrap();
        assert_eq!(&point_set, &point_set_copy);
    }

    #[test]
    fn test_l2_norm() {
        let dense = Array2::<f32>::eye(10);

        let mut sparse = TriMat::new((10, 4));
        sparse.add_triplet(0, 0, 3.0_f32);
        sparse.add_triplet(1, 2, 2.0);
        sparse.add_triplet(3, 0, -2.0);
        let sparse: CsMat<_> = sparse.to_csr();

        let point_set = PointSet::new(Some(dense.clone()), None).unwrap();
        zip(vec![1.0; 10], point_set.l2_norm().to_vec()).for_each(|e| {
            assert_approx_eq!(e.0 as f64, e.1 as f64, 0.01);
        });

        let point_set = PointSet::new(Some(dense.clone()), Some(sparse.clone())).unwrap();
        zip(
            vec![3.16, 2.23, 1.0, 2.23, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            point_set.l2_norm().to_vec(),
        )
        .for_each(|e| {
            assert_approx_eq!(e.0, e.1 as f64, 0.01);
        });

        let point_set = PointSet::new(None, Some(sparse.clone())).unwrap();
        zip(
            vec![3.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            point_set.l2_norm().to_vec(),
        )
        .for_each(|e| {
            assert_approx_eq!(e.0, e.1 as f64, 0.01);
        });
    }

    #[test]
    fn test_l2_normalize_inplace() {
        let dense = Array2::<f32>::eye(10);

        let mut sparse = TriMat::new((10, 4));
        sparse.add_triplet(0, 0, 3.0_f32);
        sparse.add_triplet(1, 2, 2.0);
        sparse.add_triplet(3, 0, -2.0);
        let sparse: CsMat<_> = sparse.to_csr();

        let mut point_set = PointSet::new(Some(dense.clone()), None).unwrap();
        point_set.l2_normalize_inplace();
        zip(vec![1.0; 10], point_set.l2_norm().to_vec()).for_each(|e| {
            assert_approx_eq!(e.0 as f64, e.1 as f64, 0.01);
        });

        let mut point_set = PointSet::new(Some(dense.clone()), Some(sparse.clone())).unwrap();
        point_set.l2_normalize_inplace();
        zip(vec![1.0; 10], point_set.l2_norm().to_vec()).for_each(|e| {
            assert_approx_eq!(e.0, e.1 as f64, 0.01);
        });

        let mut point_set = PointSet::new(None, Some(sparse.clone())).unwrap();
        point_set.l2_normalize_inplace();
        zip(
            vec![1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            point_set.l2_norm().to_vec(),
        )
        .for_each(|e| {
            assert_approx_eq!(e.0, e.1 as f64, 0.01);
        });
    }
}
