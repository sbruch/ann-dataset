pub mod in_memory_dataset;

use crate::{PointSet, QuerySet};

const TRAIN_QUERY_SET: &str = "train_query_set";
const VALIDATION_QUERY_SET: &str = "validation_query_set";
const TEST_QUERY_SET: &str = "test_query_set";

pub trait AnnDataset<DataType: Clone> {
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
    fn iter_mut(&mut self) -> Self::DataPointMutableIterator<'_>;

    /// Returns all data points.
    #[deprecated(since = "0.1.3", note = "Please use `iter()` instead.")]
    fn get_data_points(&self) -> &PointSet<DataType>;

    /// Returns a mutable view of all data points.
    #[deprecated(since = "0.1.3", note = "Please use `iter_mut()` instead.")]
    fn get_data_points_mut(&mut self) -> &mut PointSet<DataType>;

    /// Selects a subset of data points.
    fn select(&self, ids: &[usize]) -> PointSet<DataType>;

    /// Adds a new query set to the dataset with the given `label` or replaces one if it already
    /// exists.
    fn add_query_set(&mut self, label: &str, query_set: QuerySet<DataType>);

    /// Convenience method to add a "train" query set.
    fn add_train_query_set(&mut self, query_set: QuerySet<DataType>) {
        self.add_query_set(TRAIN_QUERY_SET, query_set);
    }

    /// Convenience method to add a "validation" query set.
    fn add_validation_query_set(&mut self, query_set: QuerySet<DataType>) {
        self.add_query_set(VALIDATION_QUERY_SET, query_set);
    }

    /// Convenience method to add a "test" query set.
    fn add_test_query_set(&mut self, query_set: QuerySet<DataType>) {
        self.add_query_set(TEST_QUERY_SET, query_set);
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
