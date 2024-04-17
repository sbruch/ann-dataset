use std::collections::HashMap;
use std::fmt;
use std::fmt::Formatter;
use std::string::ToString;
use anyhow::anyhow;
use hdf5::{File, H5Type};
use ndarray::Array2;
use sprs::CsMat;
use crate::data::types::{GroundTruth, VectorSet};

const SPARSE_INDPTR: &str = "indptr";
const SPARSE_INDICES: &str = "indices";
const SPARSE_DATA: &str = "data";
const SPARSE_SHAPE: &str = "shape";
const GROUP_ROOT: &str = "/";
const GROUP_SEPARATOR: &str = "/";
const GROUP_GT: &str = "ground_truths";

/// Encapsulates an ANN dataset.
pub struct AnnDataset<DataType> {
    vector_sets: HashMap<String, VectorSet<DataType>>,
    ground_truths: HashMap<String, GroundTruth>,
}

impl<DataType: H5Type> AnnDataset<DataType> {
    /// Creates an empty dataset.
    pub fn new() -> AnnDataset<DataType> {
        AnnDataset {
            vector_sets: HashMap::new(),
            ground_truths: HashMap::new(),
        }
    }

    /// Loads an `AnnDataset` from the given hdf5 file.
    pub fn load(path: &str) -> anyhow::Result<AnnDataset<DataType>> {
        let hdf5_dataset = File::open(path)?;

        let mut ann_dataset = Self::new();

        let datasets = hdf5_dataset.datasets()?;
        datasets.iter().try_for_each(|dataset| {
            let name = dataset.name();
            let name = name.strip_prefix(GROUP_ROOT).unwrap();

            let vectors = dataset.read_raw::<DataType>()?;
            let num_dimensions: usize = dataset.shape()[1];
            let vector_count = vectors.len() / num_dimensions;
            let vectors = Array2::from_shape_vec((vector_count, num_dimensions), vectors)?;

            ann_dataset.add_points(name, VectorSet::Dense(vectors))?;
            anyhow::Ok(())
        })?;

        let groups = hdf5_dataset.groups()?;
        groups.iter().try_for_each(|group| {
            if group.name() == GROUP_ROOT { return anyhow::Ok(()); }

            let name = group.name();
            let name = name.strip_prefix(GROUP_ROOT).unwrap();
            if name == GROUP_GT { return anyhow::Ok(()); }

            let shape = group.attr(SPARSE_SHAPE)?.read_raw::<usize>()?;
            if shape.len() != 2 {
                return Err(anyhow!("Corrupt shape for sparse dataset '{}'", group.name()));
            }

            let indptr = group.dataset(SPARSE_INDPTR)?.read_raw::<usize>()?;
            let indices = group.dataset(SPARSE_INDICES)?.read_raw::<usize>()?;
            let data = group.dataset(SPARSE_DATA)?.read_raw::<DataType>()?;
            let vectors = CsMat::new((shape[0], shape[1]), indptr, indices, data);

            ann_dataset.add_points(name, VectorSet::Sparse(vectors))?;
            anyhow::Ok(())
        })?;

        let gt_group = hdf5_dataset.group(GROUP_GT)?;
        gt_group.datasets()?.iter().try_for_each(|dataset| {
            let name = dataset.name();
            let name = name.strip_prefix(GROUP_ROOT).unwrap();
            let name = name.strip_prefix(GROUP_GT).unwrap();
            let name = name.strip_prefix(GROUP_SEPARATOR).unwrap();

            let vectors = dataset.read_raw::<usize>()?;
            let num_dimensions: usize = dataset.shape()[1];
            let vector_count = vectors.len() / num_dimensions;
            let vectors = Array2::from_shape_vec((vector_count, num_dimensions), vectors)?;

            ann_dataset.add_ground_truth(name, vectors)?;
            anyhow::Ok(())
        })?;

        Ok(ann_dataset)
    }

    /// Reads a vector set with the given label.
    pub fn read_points(&self, label: &str) -> anyhow::Result<&VectorSet<DataType>> {
        match self.vector_sets.get(label) {
            None => { Err(anyhow!("Vector set {} does not exist", label)) }
            Some(set) => { Ok(set) }
        }
    }

    /// Reads a vector set with the given label.
    pub fn read_ground_truth(&self, label: &str) -> anyhow::Result<&GroundTruth> {
        match self.ground_truths.get(label) {
            None => { Err(anyhow!("Ground-truth set {} does not exist", label)) }
            Some(set) => { Ok(set) }
        }
    }

    /// Adds a vector set with the given label to the dataset,
    /// or replaces the set if it already exists.
    pub fn add_points(&mut self, label: &str, vector_set: VectorSet<DataType>) -> anyhow::Result<()> {
        self.vector_sets.insert(label.to_string(), vector_set);
        Ok(())
    }

    /// Adds a set of ground truths or replaces the set if it already exists.
    pub fn add_ground_truth(&mut self, label: &str, gt: GroundTruth) -> anyhow::Result<()> {
        self.ground_truths.insert(label.to_string(), gt);
        Ok(())
    }

    /// Stores the dataset in an hdf5 file at the given path.
    pub fn write(&self, path: &str) -> anyhow::Result<()> {
        let file = File::create(path)?;
        self.vector_sets.iter().try_for_each(|(label, vector_set)| {
            match vector_set {
                VectorSet::Dense(dense_set) => {
                    let dataset = file.new_dataset::<DataType>()
                        .shape(dense_set.shape())
                        .create(label.as_str())?;
                    dataset.write(dense_set)?;
                }
                VectorSet::Sparse(sparse_set) => {
                    let group = file.create_group(label)?;

                    let shape = group.new_attr::<usize>().shape(2).create(SPARSE_SHAPE)?;
                    shape.write(&[sparse_set.shape().0, sparse_set.shape().1])?;

                    let indptr = group.new_dataset::<usize>()
                        .shape(sparse_set.indptr().len())
                        .create(SPARSE_INDPTR)?;
                    indptr.write(sparse_set.indptr().as_slice().unwrap())?;

                    let indices = group.new_dataset::<usize>()
                        .shape(sparse_set.indices().len())
                        .create(SPARSE_INDICES)?;
                    indices.write(sparse_set.indices())?;

                    let data = group.new_dataset::<DataType>()
                        .shape(sparse_set.data().len())
                        .create(SPARSE_DATA)?;
                    data.write(sparse_set.data())?;
                }
            }
            anyhow::Ok(())
        })?;

        let gt_group = file.create_group(GROUP_GT)?;
        self.ground_truths.iter().try_for_each(|(label, gt)| {
            let dataset = gt_group.new_dataset::<usize>()
                .shape(gt.shape())
                .create(label.as_str())?;
            dataset.write(gt)?;
            anyhow::Ok(())
        })?;
        file.close()?;
        Ok(())
    }
}

impl<DataType> fmt::Display for AnnDataset<DataType> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut dense_labels: Vec<String> = vec![];
        let mut sparse_labels: Vec<String> = vec![];
        self.vector_sets.iter().for_each(|entry| {
            match entry.1 {
                VectorSet::Dense(set) => {
                    dense_labels.push(format!("  - {} (dense) with shape {:?}",
                                              entry.0.to_string(), set.shape()));
                }
                VectorSet::Sparse(set) => {
                    sparse_labels.push(format!("  - {} (sparse) with shape [{}, {}]",
                                               entry.0.to_string(), set.rows(), set.cols()));
                }
            }
        });

        let ground_truths: Vec<String> = self.ground_truths.iter().map(|entry| {
            format!("  - {} (ground-truth) with shape {:?}",
                    entry.0.to_string(), entry.1.shape())
        }).collect();

        write!(f, "There are a total of {} datasets: \n{}\n{}\n{}",
               self.vector_sets.len(),
               dense_labels.join("\n"),
               sparse_labels.join("\n"),
               ground_truths.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, ArrayView2};
    use sprs::{CsMat, CsMatView, TriMat};
    use tempdir::TempDir;
    use crate::data::dataset::AnnDataset;
    use crate::data::types::VectorSet;

    fn assert_eq_dense(expected: ArrayView2<f32>, value: &VectorSet<f32>) {
        match value {
            VectorSet::Dense(set) => {
                assert_eq!(expected, set);
            }
            VectorSet::Sparse(_) => {
                panic!("Dense set was turned into a sparse set!");
            }
        }
    }

    fn assert_eq_sparse(expected: CsMatView<f32>, value: &VectorSet<f32>) {
        match value {
            VectorSet::Dense(_) => {
                panic!("Sparse set was turned into a dense set!");
            }
            VectorSet::Sparse(set) => {
                assert_eq!(expected, set.view());
            }
        }
    }

    #[test]
    fn test_new() {
        let set_a = Array2::<f32>::eye(5);
        let set_b = Array2::<usize>::ones((5, 10));

        let mut set_c = TriMat::new((4, 4));
        set_c.add_triplet(0, 0, 3.0_f32);
        set_c.add_triplet(1, 2, 2.0);
        set_c.add_triplet(3, 0, -2.0);
        let set_c: CsMat<_> = set_c.to_csr();

        let mut dataset = AnnDataset::<f32>::new();
        assert!(dataset.add_points("a", VectorSet::Dense(set_a.clone())).is_ok());
        assert!(dataset.add_ground_truth("b", set_b.clone()).is_ok());
        assert!(dataset.add_points("c", VectorSet::Sparse(set_c.clone())).is_ok());

        let a = dataset.read_points("a");
        assert!(a.is_ok());
        assert_eq_dense(set_a.view(), &a.unwrap());

        let b = dataset.read_ground_truth("b");
        assert!(b.is_ok());
        assert_eq!(set_b.view(), b.unwrap());
        assert!(dataset.read_points("b").is_err());

        let c = dataset.read_points("c");
        assert!(c.is_ok());
        assert_eq_sparse(set_c.view(), &c.unwrap());
    }

    #[test]
    fn test_nonexistent() {
        let dataset = AnnDataset::<f32>::new();
        assert!(dataset.read_points("a").is_err());
        assert!(dataset.read_ground_truth("a").is_err());
    }

    #[test]
    fn test_replace() {
        let set_a = Array2::<f32>::eye(5);

        let mut set_b = TriMat::new((4, 4));
        set_b.add_triplet(0, 0, 3.0_f32);
        set_b.add_triplet(1, 2, 2.0);
        set_b.add_triplet(3, 0, -2.0);
        let set_b: CsMat<_> = set_b.to_csr();

        let mut dataset = AnnDataset::<f32>::new();
        assert!(dataset.add_points("a", VectorSet::Dense(set_a.clone())).is_ok());
        assert!(dataset.add_points("a", VectorSet::Sparse(set_b.clone())).is_ok());

        let a = dataset.read_points("a");
        assert!(a.is_ok());
        assert_eq_sparse(set_b.view(), &a.unwrap());
    }

    #[test]
    fn test_write() {
        let set_a = Array2::<f32>::eye(5);
        let set_b = Array2::<usize>::ones((5, 10));

        let mut set_c = TriMat::new((4, 4));
        set_c.add_triplet(0, 0, 3.0_f32);
        set_c.add_triplet(1, 2, 2.0);
        set_c.add_triplet(3, 0, -2.0);
        let set_c: CsMat<_> = set_c.to_csr();

        let mut set_d = TriMat::new((4, 4));
        set_d.add_triplet(0, 3, 2.0_f32);
        set_d.add_triplet(1, 1, 1.0);
        set_d.add_triplet(3, 0, -1.0);
        let set_d: CsMat<_> = set_d.to_csr();

        let mut dataset = AnnDataset::<f32>::new();
        assert!(dataset.add_points("a", VectorSet::Dense(set_a.clone())).is_ok());
        assert!(dataset.add_ground_truth("b", set_b.clone()).is_ok());
        assert!(dataset.add_points("c", VectorSet::Sparse(set_c.clone())).is_ok());
        assert!(dataset.add_points("d", VectorSet::Sparse(set_d.clone())).is_ok());

        let dir = TempDir::new("test_write").unwrap();
        let path = dir.path().join("ann-dataset.hdf5");
        let path = path.to_str().unwrap();

        let result = dataset.write(path);
        assert!(result.is_ok());

        // Next, load the dataset and assert that vector sets are intact.
        let dataset = AnnDataset::<f32>::load(path);
        assert!(dataset.is_ok());
        let dataset = dataset.unwrap();

        let a = dataset.read_points("a");
        assert!(a.is_ok());
        assert_eq_dense(set_a.view(), &a.unwrap());

        let b = dataset.read_ground_truth("b");
        assert!(b.is_ok());
        assert_eq!(set_b.view(), b.unwrap());

        let c = dataset.read_points("c");
        assert!(c.is_ok());
        assert_eq_sparse(set_c.view(), &c.unwrap());

        let d = dataset.read_points("d");
        assert!(d.is_ok());
        assert_eq_sparse(set_d.view(), &d.unwrap());
    }
}
