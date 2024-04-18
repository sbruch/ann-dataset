use std::cmp::min;
use ndarray::{Array2, ArrayView2};
use roaring::RoaringBitmap;
use anyhow::{anyhow, Result};

/// Defines the exact nearest neighbors.
#[derive(Eq, PartialEq, Default, Debug)]
pub struct GroundTruth(Array2<usize>);

impl GroundTruth {
    pub fn new(neighbors: Array2<usize>) -> GroundTruth {
        GroundTruth(neighbors)
    }

    /// Returns the set of neighbors.
    pub fn get_neighbors(&self) -> ArrayView2<usize> { self.0.view() }

    /// Computes recall given a retrieved set.
    ///
    /// Returns an error if the number of queries does not match between the retrieved `top_k` set
    /// and the exact neighbor set stored in this object.
    pub fn mean_recall(&self, top_k: &[Vec<usize>]) -> Result<f32> {
        if top_k.len() != self.0.nrows() {
            return Err(anyhow!(
                "Retrieved set has {} queries, but expected {} queries",
                top_k.len(), self.0.nrows()));
        }

        if top_k.is_empty() {
            return Ok(1_f32);
        }
        let k = min(top_k[0].len(), self.0.ncols());

        let recall = top_k.iter()
            .enumerate()
            .map(|(i, set)| {
                let intersection_len =
                    RoaringBitmap::from_iter(self.0.row(i).iter().map(|x| *x as u32).take(k))
                        .intersection_len(&RoaringBitmap::from_iter(set.iter().map(|x| *x as u32).take(k))) as f64;
                intersection_len / k as f64
            }).sum::<f64>();
        Ok(recall as f32 / top_k.len() as f32)
    }
}

#[cfg(test)]
mod tests {
    use approx_eq::assert_approx_eq;
    use ndarray::Array2;
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
}
