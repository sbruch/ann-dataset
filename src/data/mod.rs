pub mod in_memory_dataset;

use crate::{PointSet, QuerySet};
use std::sync::mpsc::{channel, Receiver};

const TRAIN_QUERY_SET: &str = "train_query_set";
const VALIDATION_QUERY_SET: &str = "validation_query_set";
const TEST_QUERY_SET: &str = "test_query_set";

pub trait AnnDataset<DataType: Clone + Sync + Send + 'static> {
    type DataPointIterator<'a>: Iterator
    where
        DataType: 'a,
        Self: 'a;

    type DataPointMutableIterator<'a>: Iterator
    where
        DataType: 'a,
        Self: 'a;

    /// Provides an `Iterator` over chunks of data points as `PointSet` objects.
    fn iter(&self) -> Self::DataPointIterator<'_>;

    /// Provides a mutable `Iterator` over chunks of data points as `PointSet` objects.
    ///
    /// It is important to note that, this only provides a mutable view of objects in memory.
    /// In other words, any modifications to the underlying object persist so long as the object
    /// resides in memory. The moment an object is evicted from memory, any and all
    /// modifications to it would be discarded.
    fn iter_mut(&mut self) -> Self::DataPointMutableIterator<'_>;

    /// Returns the total number of data points in the dataset.
    fn num_data_points(&self) -> usize;

    /// Returns all data points.
    fn get_data_points(&self) -> &PointSet<DataType>;

    /// Returns a mutable view of all data points.
    fn get_data_points_mut(&mut self) -> &mut PointSet<DataType>;

    /// Selects a subset of data points.
    fn select(&self, ids: &[usize]) -> PointSet<DataType>;

    /// Returns the total number of query points labeled with `label`.
    fn num_query_points(&self, label: &str) -> anyhow::Result<usize>;

    /// Convenience method that returns the total number of train query points.
    fn num_train_query_points(&self) -> anyhow::Result<usize> {
        self.num_query_points(TRAIN_QUERY_SET)
    }

    /// Convenience method that returns the total number of validation query points.
    fn num_validation_query_points(&self) -> anyhow::Result<usize> {
        self.num_query_points(VALIDATION_QUERY_SET)
    }

    /// Convenience method that returns the total number of test query points.
    fn num_test_query_points(&self) -> anyhow::Result<usize> {
        self.num_query_points(TEST_QUERY_SET)
    }

    /// Consumes a set of `QuerySet` objects to build a unified query set labeled as `label`,
    /// and adds it to the dataset.
    ///
    /// If a set with label `label` already exists, this method discards the existing set
    /// and replaces it with the new set.
    fn add_query_sets(
        &mut self,
        label: &str,
        query_sets: Receiver<QuerySet<DataType>>,
    ) -> anyhow::Result<()>;

    /// Adds a single `QuerySet` to the dataset with the given `label` or replaces it if it already
    /// exists.
    fn add_query_set(&mut self, label: &str, query_set: QuerySet<DataType>) -> anyhow::Result<()> {
        let (tx, rx) = channel::<QuerySet<DataType>>();
        tx.send(query_set)?;
        drop(tx);
        self.add_query_sets(label, rx)
    }

    /// Convenience method to add a "train" query set.
    fn add_train_query_set(&mut self, query_set: QuerySet<DataType>) -> anyhow::Result<()> {
        self.add_query_set(TRAIN_QUERY_SET, query_set)
    }

    /// Convenience method to add a "validation" query set.
    fn add_validation_query_set(&mut self, query_set: QuerySet<DataType>) -> anyhow::Result<()> {
        self.add_query_set(VALIDATION_QUERY_SET, query_set)
    }

    /// Convenience method to add a "test" query set.
    fn add_test_query_set(&mut self, query_set: QuerySet<DataType>) -> anyhow::Result<()> {
        self.add_query_set(TEST_QUERY_SET, query_set)
    }

    fn get_query_set(&self, label: &str) -> anyhow::Result<&QuerySet<DataType>>;

    /// Convenience method that returns the "train" `QuerySet`.
    fn get_train_query_set(&self) -> anyhow::Result<&QuerySet<DataType>> {
        self.get_query_set(TRAIN_QUERY_SET)
    }

    /// Convenience method that returns the "validation" `QuerySet`.
    fn get_validation_query_set(&self) -> anyhow::Result<&QuerySet<DataType>> {
        self.get_query_set(VALIDATION_QUERY_SET)
    }

    /// Convenience method that returns the "test" `QuerySet`.
    fn get_test_query_set(&self) -> anyhow::Result<&QuerySet<DataType>> {
        self.get_query_set(TEST_QUERY_SET)
    }
}
