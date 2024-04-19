use std::cmp::min;
use std::fmt::{Display, Formatter};
use ndarray::{Array2, ArrayView2};
use roaring::RoaringBitmap;
use anyhow::{anyhow, Result};
use hdf5::Group;
use crate::Hdf5Serialization;

/// Defines the exact nearest neighbors.
#[derive(Eq, PartialEq, Default, Debug, Clone)]
pub struct GroundTruth(Array2<usize>);

impl GroundTruth {
    pub fn new(neighbors: Array2<usize>) -> GroundTruth {
        GroundTruth(neighbors)
    }

    /// Returns the set of neighbors.
    pub fn get_neighbors(&self) -> ArrayView2<usize> { self.0.view() }

    /// Computes recall given a retrieved set.
    ///
    /// Returns an error if the number of queries does not match between `retrieved_set`
    /// and the exact neighbor set stored in this object.
    pub fn mean_recall(&self, retrieved_set: &[Vec<usize>]) -> Result<f32> {
        if retrieved_set.len() != self.0.nrows() {
            return Err(anyhow!(
                "Retrieved set has {} queries, but expected {} queries",
                retrieved_set.len(), self.0.nrows()));
        }

        if retrieved_set.is_empty() {
            return Ok(1_f32);
        }
        let k = min(retrieved_set[0].len(), self.0.ncols());

        let recall = retrieved_set.iter()
            .enumerate()
            .map(|(i, set)| {
                let intersection_len =
                    RoaringBitmap::from_iter(self.0.row(i).iter().map(|x| *x as u32).take(k))
                        .intersection_len(&RoaringBitmap::from_iter(set.iter().map(|x| *x as u32).take(k))) as f64;
                intersection_len / k as f64
            }).sum::<f64>();
        Ok(recall as f32 / retrieved_set.len() as f32)
    }
}

impl Hdf5Serialization for GroundTruth {
    type Object = GroundTruth;

    fn serialize(&self, group: &mut Group) -> Result<()> {
        let dataset = group.new_dataset::<usize>()
            .shape(self.0.shape())
            .create(Self::label().as_str())?;
        dataset.write(self.0.view())?;
        Ok(())
    }

    fn deserialize(group: &Group) -> Result<Self::Object> {
        let dataset = group.dataset(Self::label().as_str())?;

        let vectors = dataset.read_raw::<usize>()?;
        let num_dimensions: usize = dataset.shape()[1];
        let vector_count = vectors.len() / num_dimensions;
        let vectors = Array2::from_shape_vec((vector_count, num_dimensions), vectors)?;

        Ok(GroundTruth(vectors))
    }

    fn label() -> String {
        "ground-truth".to_string()
    }
}

impl Display for GroundTruth {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Shape [{}, {}]", self.0.shape()[0], self.0.shape()[1])
    }
}

#[cfg(test)]
mod tests {
    use approx_eq::assert_approx_eq;
    use hdf5::File;
    use ndarray::Array2;
    use tempdir::TempDir;
    use crate::Hdf5Serialization;
    use crate::types::ground_truth::GroundTruth;

    #[test]
    fn test_recall() {
        let gt = GroundTruth::new(
            Array2::from_shape_vec(
                (3, 3),
                vec![1_usize, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap());
        assert!(gt.mean_recall(&vec![]).is_err());

        let recall = gt.mean_recall(
            &vec![vec![1_usize], vec![5], vec![1]]);
        assert_approx_eq!(recall.unwrap().into(), 0.333, 0.01);

        let recall = gt.mean_recall(
            &vec![vec![1_usize, 2], vec![5, 6], vec![1, 8]]);
        assert_approx_eq!(recall.unwrap().into(), 0.666, 0.01);
    }

    #[test]
    fn test_hdf5() {
        let gt = GroundTruth::new(
            Array2::from_shape_vec(
                (3, 3),
                vec![1_usize, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap());

        let dir = TempDir::new("gt_test_hdf5").unwrap();
        let path = dir.path().join("ann-dataset.hdf5");
        let path = path.to_str().unwrap();
        let hdf5 = File::create(path).unwrap();

        let mut group = hdf5.group("/").unwrap();
        assert!(gt.serialize(&mut group).is_ok());

        let gt_copy = GroundTruth::deserialize(&group).unwrap();
        assert_eq!(&gt, &gt_copy);

        let group = hdf5.group("/").unwrap();
        let mut group = group.create_group("nested").unwrap();
        assert!(gt.serialize(&mut group).is_ok());

        let gt_copy = GroundTruth::deserialize(&group).unwrap();
        assert_eq!(&gt, &gt_copy);
    }
}
