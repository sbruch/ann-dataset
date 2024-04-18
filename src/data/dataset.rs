use std::collections::HashMap;
use std::fmt;
use std::fmt::Formatter;
use std::string::ToString;
use anyhow::{anyhow, Result};
use hdf5::{File, Group, H5Type};
use crate::{Hdf5Serialization, PointSet, QuerySet};
use crate::data::AnnDataset;
use crate::io::Hdf5File;

const QUERY_SETS: &str = "query_sets";

/// An ANN dataset.
#[derive(Eq, PartialEq, Debug, Clone)]
pub struct InMemoryAnnDataset<DataType: Clone + H5Type> {
    data_points: PointSet<DataType>,
    query_sets: HashMap<String, QuerySet<DataType>>,
}

impl<DataType: Clone + H5Type> InMemoryAnnDataset<DataType> {
    /// Creates an `AnnDataset` object.
    ///
    /// Here is a simple example:
    /// ```rust
    /// use ndarray::Array2;
    /// use sprs::{CsMat, TriMat};
    /// use ann_dataset::{InMemoryAnnDataset, PointSet};
    ///
    /// let dense = Array2::<f32>::eye(10);
    /// let mut sparse = TriMat::new((10, 4));
    /// sparse.add_triplet(0, 0, 3.0_f32);
    /// sparse.add_triplet(1, 2, 2.0);
    /// sparse.add_triplet(3, 0, -2.0);
    /// sparse.add_triplet(9, 2, 3.4);
    /// let sparse: CsMat<_> = sparse.to_csr();
    ///
    /// let data_points = PointSet::new(Some(dense.clone()), Some(sparse.clone()))
    ///     .expect("Failed to create PointSet.");
    ///
    /// let dataset = InMemoryAnnDataset::create(data_points);
    /// ```
    pub fn create(data_points: PointSet<DataType>) -> InMemoryAnnDataset<DataType> {
        InMemoryAnnDataset {
            data_points,
            query_sets: HashMap::new(),
        }
    }
}

impl<DataType: Clone + H5Type> AnnDataset<DataType> for InMemoryAnnDataset<DataType> {
    fn get_data_points(&self) -> &PointSet<DataType> { &self.data_points }

    /// Adds a new query set to the dataset with the given `label` or replaces one if it already
    /// exists.
    ///
    /// Consider the following example:
    /// ```rust
    /// use ndarray::Array2;
    /// use ann_dataset::{AnnDataset, InMemoryAnnDataset, PointSet, QuerySet};
    ///
    /// let dense = Array2::<f32>::eye(10);
    /// let data_points = PointSet::new(Some(dense.clone()), None)
    ///     .expect("Failed to create PointSet.");
    /// let query_points = data_points.clone();
    ///
    /// let mut dataset = InMemoryAnnDataset::create(data_points);
    ///
    /// let query_set = QuerySet::new(query_points);
    /// dataset.add_query_set("train", query_set);
    /// ```
    fn add_query_set(&mut self, label: &str, query_set: QuerySet<DataType>) {
        self.query_sets.insert(label.to_string(), query_set);
    }

    fn get_query_set(&self, label: &str) -> Result<&QuerySet<DataType>> {
        match self.query_sets.get(label) {
            None => { Err(anyhow!("Query set {} does not exist", label)) }
            Some(set) => { Ok(set) }
        }
    }
}

impl<DataType: Clone + H5Type> Hdf5Serialization for InMemoryAnnDataset<DataType> {
    type Object = InMemoryAnnDataset<DataType>;

    fn serialize(&self, group: &mut Group) -> Result<()> {
        self.data_points.serialize(group)?;

        let query_group = group.create_group(QUERY_SETS)?;
        self.query_sets.iter().try_for_each(|entry| {
            let mut grp = query_group.create_group(entry.0)?;
            entry.1.serialize(&mut grp)?;
            anyhow::Ok(())
        })?;
        Ok(())
    }

    fn deserialize(group: &Group) -> Result<Self::Object> {
        let data_points = PointSet::<DataType>::deserialize(group)?;

        let mut query_sets: HashMap<String, QuerySet<DataType>> = HashMap::new();
        let query_group = group.group(QUERY_SETS)?;
        query_group.groups()?.iter().try_for_each(|grp| {
            let name = grp.name();
            let name = name.split("/").last().unwrap();
            let query_set = QuerySet::<DataType>::deserialize(grp)?;
            query_sets.insert(name.to_string(), query_set);
            anyhow::Ok(())
        })?;

        Ok(InMemoryAnnDataset { data_points, query_sets })
    }

    fn label() -> String {
        "ann-dataset".to_string()
    }
}

impl<DataType: Clone + H5Type> Hdf5File for InMemoryAnnDataset<DataType> {
    type Object = InMemoryAnnDataset<DataType>;

    fn write(&self, path: &str) -> Result<()> {
        let file = File::create(path)?;
        let mut root = file.group("/")?;
        Hdf5Serialization::serialize(self, &mut root)?;
        file.close()?;
        Ok(())
    }

    fn read(path: &str) -> Result<Self::Object> {
        let hdf5_dataset = File::open(path)?;
        let root = hdf5_dataset.group("/")?;
        <InMemoryAnnDataset::<DataType> as Hdf5Serialization>::deserialize(&root)
    }
}

impl<DataType: Clone + H5Type> fmt::Display for InMemoryAnnDataset<DataType> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Point Set: {}\n{}",
               self.data_points,
               self.query_sets.iter()
                   .map(|entry| format!("{}: {}", entry.0, entry.1))
                   .collect::<Vec<_>>()
                   .join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use sprs::{CsMat, TriMat};
    use tempdir::TempDir;
    use crate::data::dataset::InMemoryAnnDataset;
    use crate::{Hdf5File, PointSet, QuerySet};
    use crate::data::AnnDataset;

    fn sample_data_points() -> PointSet<f32> {
        let dense_set = Array2::random((4, 10), Uniform::new(0.0, 1.0));

        let mut sparse_set = TriMat::new((4, 4));
        sparse_set.add_triplet(0, 0, 3.0_f32);
        sparse_set.add_triplet(1, 2, 2.0);
        sparse_set.add_triplet(3, 0, -2.0);
        let sparse_set: CsMat<_> = sparse_set.to_csr();

        PointSet::new(Some(dense_set), Some(sparse_set)).unwrap()
    }

    #[test]
    fn test_create() {
        let data_points = sample_data_points();
        let dataset = InMemoryAnnDataset::<f32>::create(data_points.clone());
        let copy = dataset.get_data_points();
        assert_eq!(&data_points, copy);
    }

    #[test]
    fn test_query_points() {
        let data_points = sample_data_points();
        let mut dataset = InMemoryAnnDataset::<f32>::create(data_points.clone());

        assert!(dataset.get_train_query_set().is_err());
        assert!(dataset.get_validation_query_set().is_err());
        assert!(dataset.get_test_query_set().is_err());

        let query_points = sample_data_points();
        dataset.add_train_query_set(QuerySet::new(query_points.clone()));
        assert!(dataset.get_train_query_set().is_ok());
        let copy = dataset.get_train_query_set().unwrap();
        assert_eq!(&query_points, copy.get_points());

        // Replace an existing query set.
        let query_points = sample_data_points();
        dataset.add_train_query_set(QuerySet::new(query_points.clone()));
        assert!(dataset.get_train_query_set().is_ok());
        let copy = dataset.get_train_query_set().unwrap();
        assert_eq!(&query_points, copy.get_points());
    }

    #[test]
    fn test_write() {
        let data_points = sample_data_points();
        let mut dataset = InMemoryAnnDataset::<f32>::create(data_points.clone());
        let query_points = sample_data_points();
        dataset.add_train_query_set(QuerySet::new(query_points.clone()));

        let dir = TempDir::new("test_write").unwrap();
        let path = dir.path().join("ann-dataset.hdf5");
        let path = path.to_str().unwrap();

        let result = dataset.write(path);
        assert!(result.is_ok());

        // Next, load the dataset and assert that vector sets are intact.
        let dataset = InMemoryAnnDataset::<f32>::read(path);
        assert!(dataset.is_ok());
        let dataset = dataset.unwrap();

        assert_eq!(&data_points, dataset.get_data_points());
        assert_eq!(&query_points, dataset.get_train_query_set().unwrap().get_points());
    }
}
