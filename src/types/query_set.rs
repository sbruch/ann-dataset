use crate::types::ground_truth::GroundTruth;
use crate::types::Metric;
use crate::{Hdf5Serialization, PointSet};
use anyhow::{anyhow, Result};
use hdf5::{Group, H5Type};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

const QUERIES: &str = "queries";
const GROUND_TRUTH: &str = "gt";

/// A set of query points (dense, sparse, or both) and their exact nearest neighbors for various
/// metrics.
#[derive(Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
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
    pub fn get_points(&self) -> &PointSet<DataType> {
        &self.points
    }

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
                neighbors.nrows(),
                self.points.num_points()
            ));
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
        Err(anyhow!(
            "No solution to ANN with {:?} was provided.",
            metric
        ))
    }
}

impl<DataType: Clone + H5Type> Hdf5Serialization for QuerySet<DataType> {
    type Object = QuerySet<DataType>;

    fn add_to(&self, group: &mut Group) -> Result<()> {
        let mut query_group = group.create_group(QUERIES)?;
        self.points.add_to(&mut query_group)?;

        let gt_group = group.create_group(GROUND_TRUTH)?;
        self.neighbors.iter().try_for_each(|entry| {
            let mut grp = gt_group.create_group(entry.0.to_string().as_str())?;
            entry.1.add_to(&mut grp)?;
            anyhow::Ok(())
        })?;

        Ok(())
    }

    fn read_from(group: &Group) -> Result<Self::Object> {
        let query_group = group.group(QUERIES)?;
        let points = PointSet::<DataType>::read_from(&query_group)?;

        let mut neighbors: HashMap<Metric, GroundTruth> = HashMap::new();
        let gt_group = group.group(GROUND_TRUTH)?;
        gt_group.groups()?.iter().try_for_each(|grp| {
            let name = grp.name();
            let name = name.split('/').last().unwrap();
            let metric = Metric::from_str(name)?;
            let gt = GroundTruth::read_from(grp)?;
            neighbors.insert(metric, gt);
            anyhow::Ok(())
        })?;

        Ok(QuerySet { points, neighbors })
    }

    fn label() -> String {
        "query-set".to_string()
    }
}

impl<DataType: Clone> Display for QuerySet<DataType> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Query points: {}\nGround-truths: {}",
            self.points,
            self.neighbors
                .iter()
                .map(|entry| format!("{}: {}", entry.0, entry.1))
                .collect::<Vec<_>>()
                .join("; ")
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::types::Metric::{Cosine, Euclidean, InnerProduct};
    use crate::{Hdf5Serialization, PointSet, QuerySet};
    use hdf5::File;
    use ndarray::Array2;
    use tempdir::TempDir;

    #[test]
    fn test_new() {
        let dense = Array2::<f64>::eye(5);
        let queries = PointSet::<f64>::new(Some(dense), None).unwrap();
        let mut query_set = QuerySet::new(queries);

        assert!(query_set
            .add_ground_truth(InnerProduct, Array2::<usize>::eye(3))
            .is_err());

        assert!(query_set
            .add_ground_truth(InnerProduct, Array2::<usize>::zeros((5, 1)))
            .is_ok());
        assert!(query_set
            .add_ground_truth(Euclidean, Array2::<usize>::ones((5, 1)))
            .is_ok());

        assert!(query_set.get_ground_truth(&Cosine).is_err());
        assert_eq!(
            query_set
                .get_ground_truth(&Euclidean)
                .unwrap()
                .get_neighbors(),
            Array2::<usize>::ones((5, 1))
        );
        assert_eq!(
            query_set
                .get_ground_truth(&InnerProduct)
                .unwrap()
                .get_neighbors(),
            Array2::<usize>::zeros((5, 1))
        );
    }

    #[test]
    fn test_hdf5() {
        let dense = Array2::<f64>::eye(5);
        let queries = PointSet::<f64>::new(Some(dense), None).unwrap();
        let mut query_set = QuerySet::new(queries);

        assert!(query_set
            .add_ground_truth(InnerProduct, Array2::<usize>::zeros((5, 1)))
            .is_ok());
        assert!(query_set
            .add_ground_truth(Euclidean, Array2::<usize>::ones((5, 1)))
            .is_ok());

        let dir = TempDir::new("pointset_test_hdf5").unwrap();
        let path = dir.path().join("ann-dataset.hdf5");
        let path = path.to_str().unwrap();
        let hdf5 = File::create(path).unwrap();

        let mut group = hdf5.group("/").unwrap();
        assert!(query_set.add_to(&mut group).is_ok());
        let query_set_copy = QuerySet::<f64>::read_from(&group).unwrap();
        assert_eq!(&query_set, &query_set_copy);

        let mut group = group.create_group("nested").unwrap();
        assert!(query_set.add_to(&mut group).is_ok());
        let query_set_copy = QuerySet::<f64>::read_from(&group).unwrap();
        assert_eq!(&query_set, &query_set_copy);
    }

    #[test]
    fn test_hdf5_no_gt() {
        let dense = Array2::<f64>::eye(5);
        let queries = PointSet::<f64>::new(Some(dense), None).unwrap();
        let query_set = QuerySet::new(queries);

        let dir = TempDir::new("pointset_test_hdf5").unwrap();
        let path = dir.path().join("ann-dataset.hdf5");
        let path = path.to_str().unwrap();
        let hdf5 = File::create(path).unwrap();

        let mut group = hdf5.group("/").unwrap();
        assert!(query_set.add_to(&mut group).is_ok());
        let query_set_copy = QuerySet::<f64>::read_from(&group).unwrap();
        assert_eq!(&query_set, &query_set_copy);

        let mut group = group.create_group("nested").unwrap();
        assert!(query_set.add_to(&mut group).is_ok());
        let query_set_copy = QuerySet::<f64>::read_from(&group).unwrap();
        assert_eq!(&query_set, &query_set_copy);
    }
}
