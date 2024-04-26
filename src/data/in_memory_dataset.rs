use crate::data::AnnDataset;
use crate::io::Hdf5File;
use crate::{Hdf5Serialization, PointSet, QuerySet};
use anyhow::{anyhow, Result};
use hdf5::{File, Group, H5Type};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::fmt::Formatter;
use std::sync::mpsc::Receiver;

const QUERY_SETS: &str = "query_sets";

/// An ANN dataset.
#[derive(Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct InMemoryAnnDataset<DataType: Clone> {
    data_points: PointSet<DataType>,
    query_sets: HashMap<String, QuerySet<DataType>>,
}

impl<DataType: Clone> InMemoryAnnDataset<DataType> {
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

pub struct PointSetIterator<'a, DataType: Clone> {
    point_set: &'a PointSet<DataType>,
    consumed: bool,
}

impl<'a, DataType: Clone> Iterator for PointSetIterator<'a, DataType> {
    type Item = &'a PointSet<DataType>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.consumed {
            return None;
        }
        self.consumed = true;
        Some(self.point_set)
    }
}

pub struct PointSetMutableIterator<'a, DataType: Clone> {
    point_set: &'a mut PointSet<DataType>,
    consumed: bool,
}

impl<'a, DataType: Clone> Iterator for PointSetMutableIterator<'a, DataType> {
    type Item = &'a mut PointSet<DataType>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.consumed {
            return None;
        }
        self.consumed = true;
        unsafe {
            let ptr: *mut PointSet<DataType> = self.point_set;
            Some(&mut *ptr)
        }
    }
}

impl<DataType: Clone + Sync + Send + 'static> AnnDataset<DataType>
    for InMemoryAnnDataset<DataType>
{
    type DataPointIterator<'a> = PointSetIterator<'a, DataType> where DataType: 'a;
    type DataPointMutableIterator<'a> = PointSetMutableIterator<'a, DataType> where DataType: 'a;

    fn iter(&self) -> Self::DataPointIterator<'_> {
        PointSetIterator {
            point_set: &self.data_points,
            consumed: false,
        }
    }

    fn iter_mut<'a>(&'a mut self) -> Self::DataPointMutableIterator<'_> {
        PointSetMutableIterator::<'a> {
            point_set: &mut self.data_points,
            consumed: false,
        }
    }

    fn num_data_points(&self) -> usize {
        self.data_points.num_points()
    }

    fn get_data_points(&self) -> &PointSet<DataType> {
        &self.data_points
    }

    fn get_data_points_mut(&mut self) -> &mut PointSet<DataType> {
        &mut self.data_points
    }

    fn select(&self, ids: &[usize]) -> PointSet<DataType> {
        self.data_points.select(ids)
    }

    fn num_query_points(&self, label: &str) -> Result<usize> {
        match self.query_sets.get(label) {
            None => Err(anyhow!("Query set {} does not exist", label)),
            Some(set) => Ok(set.get_points().num_points()),
        }
    }

    /// Consumes a set of `QuerySet` objects to build a unified query set labeled as `label`.
    ///
    /// If a set with label `label` already exists, this method discards the existing set
    /// and replaces it with the new set.
    ///
    /// Consider the following example:
    /// ```rust
    /// use std::sync::mpsc::channel;
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
    /// let (tx, rx) = channel::<QuerySet<f32>>();
    ///  tx.send(query_set).expect("Failed to send query set to channel.");
    ///  drop(tx);
    ///  dataset.add_query_sets("train", rx)
    ///     .expect("Failed to add query set to the dataset.");
    /// ```
    fn add_query_sets(
        &mut self,
        label: &str,
        query_sets: Receiver<QuerySet<DataType>>,
    ) -> Result<()> {
        if self.query_sets.contains_key(label) {
            self.query_sets.remove(label);
        }

        for query_set in query_sets {
            if let Some(set) = self.query_sets.get_mut(label) {
                set.is_appendable(&query_set)?;
                set.append(&query_set)?;
            } else {
                self.query_sets.insert(label.to_string(), query_set);
            }
        }
        Ok(())
    }

    fn get_query_set(&self, label: &str) -> Result<&QuerySet<DataType>> {
        match self.query_sets.get(label) {
            None => Err(anyhow!("Query set {} does not exist", label)),
            Some(set) => Ok(set),
        }
    }
}

impl<DataType: Clone + H5Type> Hdf5Serialization for InMemoryAnnDataset<DataType> {
    type Object = InMemoryAnnDataset<DataType>;

    fn add_to(&self, group: &mut Group) -> Result<()> {
        self.data_points.add_to(group)?;

        let query_group = group.create_group(QUERY_SETS)?;
        self.query_sets.iter().try_for_each(|entry| {
            let mut grp = query_group.create_group(entry.0)?;
            entry.1.add_to(&mut grp)?;
            anyhow::Ok(())
        })?;
        Ok(())
    }

    fn read_from(group: &Group) -> Result<Self::Object> {
        let data_points = PointSet::<DataType>::read_from(group)?;

        let mut query_sets: HashMap<String, QuerySet<DataType>> = HashMap::new();
        let query_group = group.group(QUERY_SETS)?;
        query_group.groups()?.iter().try_for_each(|grp| {
            let name = grp.name();
            let name = name.split('/').last().unwrap();
            let query_set = QuerySet::<DataType>::read_from(grp)?;
            query_sets.insert(name.to_string(), query_set);
            anyhow::Ok(())
        })?;

        Ok(InMemoryAnnDataset {
            data_points,
            query_sets,
        })
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
        Hdf5Serialization::add_to(self, &mut root)?;
        file.close()?;
        Ok(())
    }

    fn read(path: &str) -> Result<Self::Object> {
        let hdf5_dataset = File::open(path)?;
        let root = hdf5_dataset.group("/")?;
        <InMemoryAnnDataset<DataType> as Hdf5Serialization>::read_from(&root)
    }
}

impl<DataType: Clone> fmt::Display for InMemoryAnnDataset<DataType> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Point Set: {}\n{}",
            self.data_points,
            self.query_sets
                .iter()
                .map(|entry| format!("{}: {}", entry.0, entry.1))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::data::in_memory_dataset::InMemoryAnnDataset;
    use crate::data::AnnDataset;
    use crate::{Hdf5File, PointSet, QuerySet};
    use ndarray::Array2;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use sprs::{CsMat, TriMat};
    use tempdir::TempDir;

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
    #[allow(deprecated)]
    fn test_create() {
        let data_points = sample_data_points();
        let dataset = InMemoryAnnDataset::<f32>::create(data_points.clone());
        let copy = dataset.get_data_points();
        assert_eq!(&data_points, copy);

        assert_eq!(4, dataset.num_data_points());
        assert!(dataset.num_query_points("nonexistent").is_err());
    }

    #[test]
    fn test_iter() {
        let data_points = sample_data_points();
        let dataset = InMemoryAnnDataset::<f32>::create(data_points.clone());
        assert_eq!(1_usize, dataset.iter().count());

        for point_set in dataset.iter() {
            assert_eq!(&data_points, point_set);
        }
    }

    #[test]
    fn test_iter_mut() {
        let data_points = sample_data_points();
        let mut dataset = InMemoryAnnDataset::<f32>::create(data_points.clone());
        assert_eq!(1_usize, dataset.iter_mut().count());

        for point_set in dataset.iter_mut() {
            assert_eq!(&data_points, point_set);
            point_set.l2_normalize_inplace();
        }

        let mut data_points = data_points.clone();
        data_points.l2_normalize_inplace();
        for point_set in dataset.iter() {
            assert_eq!(&data_points, point_set);
        }
    }

    #[test]
    fn test_query_points() {
        let data_points = sample_data_points();
        let mut dataset = InMemoryAnnDataset::<f32>::create(data_points.clone());

        assert!(dataset.get_train_query_set().is_err());
        assert!(dataset.get_validation_query_set().is_err());
        assert!(dataset.get_test_query_set().is_err());

        let query_points = sample_data_points();
        assert!(dataset
            .add_train_query_set(QuerySet::new(query_points.clone()))
            .is_ok());
        assert!(dataset.get_train_query_set().is_ok());
        assert_eq!(4, dataset.num_train_query_points().unwrap());
        let copy = dataset.get_train_query_set().unwrap();
        assert_eq!(&query_points, copy.get_points());

        // Replace an existing query set.
        let query_points = sample_data_points();
        assert!(dataset
            .add_train_query_set(QuerySet::new(query_points.clone()))
            .is_ok());
        assert!(dataset.get_train_query_set().is_ok());
        let copy = dataset.get_train_query_set().unwrap();
        assert_eq!(&query_points, copy.get_points());
    }

    #[test]
    fn test_write() {
        let data_points = sample_data_points();
        let mut dataset = InMemoryAnnDataset::<f32>::create(data_points.clone());
        let query_points = sample_data_points();
        assert!(dataset
            .add_train_query_set(QuerySet::new(query_points.clone()))
            .is_ok());

        let dir = TempDir::new("test_write").unwrap();
        let path = dir.path().join("ann-dataset.hdf5");
        let path = path.to_str().unwrap();

        let result = dataset.write(path);
        assert!(result.is_ok());

        // Next, load the dataset and assert that vector sets are intact.
        let dataset = InMemoryAnnDataset::<f32>::read(path);
        assert!(dataset.is_ok());
        let dataset = dataset.unwrap();

        assert_eq!(&data_points, dataset.iter().next().unwrap());
        assert_eq!(
            &query_points,
            dataset.get_train_query_set().unwrap().get_points()
        );
    }
}
