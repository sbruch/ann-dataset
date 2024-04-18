use anyhow::{anyhow, Result};
use std::collections::HashMap;
use ndarray::Array2;
use crate::PointSet;
use crate::types::ground_truth::GroundTruth;
use crate::types::Metric;

/// A set of query points (dense, sparse, or both) and their exact nearest neighbors for various
/// metrics.
#[derive(Eq, PartialEq)]
pub struct QuerySet<DataType: Clone> {
    points: PointSet<DataType>,
    neighbors: HashMap<Metric, GroundTruth>,
}

impl<DataType: Clone> QuerySet<DataType> {
    /// Creates a new QuerySet from a set of query points.
    pub fn new(points: PointSet<DataType>) -> QuerySet<DataType> {
        QuerySet {
            points,
            neighbors: HashMap::new(),
        }
    }

    /// Returns the set of query points.
    pub fn get_points(&self) -> &PointSet<DataType> { &self.points }

    /// Adds a set of exact nearest neighbors to the query set, as solutions to ANN with the given
    /// metric.
    ///
    /// Returns an error if the number of rows in `neighbors` does not match the number of query
    /// points.
    pub fn add_ground_truth(&mut self, metric: Metric, neighbors: Array2<usize>) -> Result<()> {
        if neighbors.nrows() != self.points.num_points() {
            return Err(anyhow!(
                "Number of rows in `neighbors` ({}) must match the \
                number of query points in the set {}.",
                neighbors.nrows(), self.points.num_points()));
        }
        self.neighbors.insert(metric, GroundTruth::new(neighbors));
        Ok(())
    }

    /// Returns the set of exact nearest neighbors for ANN search with the given metric; or an error
    /// if the query set does not have the solution.
    pub fn get_ground_truth(&self, metric: &Metric) -> Result<&GroundTruth> {
        if let Some(gt) = self.neighbors.get(metric) {
            return Ok(gt);
        }
        Err(anyhow!("No solution to ANN with {:?} was provided.", metric))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use crate::{PointSet, QuerySet};
    use crate::types::Metric::{Cosine, Euclidean, InnerProduct};

    #[test]
    fn test_new() {
        let dense = Array2::<f64>::eye(5);
        let queries = PointSet::<f64>::new(Some(dense), None).unwrap();
        let mut query_set = QuerySet::new(queries);

        assert!(query_set.add_ground_truth(InnerProduct, Array2::<usize>::eye(3)).is_err());

        assert!(query_set.add_ground_truth(
            InnerProduct, Array2::<usize>::zeros((5, 1))).is_ok());
        assert!(query_set.add_ground_truth(
            Euclidean, Array2::<usize>::ones((5, 1))).is_ok());

        assert!(query_set.get_ground_truth(&Cosine).is_err());
        assert_eq!(query_set.get_ground_truth(&Euclidean).unwrap().get_neighbors(),
                   Array2::<usize>::ones((5, 1)));
        assert_eq!(query_set.get_ground_truth(&InnerProduct).unwrap().get_neighbors(),
                   Array2::<usize>::zeros((5, 1)));
    }
}
