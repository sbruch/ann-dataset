A lightweight wrapper around the HDF5 file format that
adds support for dense and sparse vector sets.
This research library is tailored towards the Approximate
Nearest Neighbor search use case.

## Creating an ANN dataset

```rust
let dense_set = ndarray::Array2::<f32>::eye(5);

let mut sparse_set = sprs::TriMat::new((4, 4));
sparse_set.add_triplet(0, 0, 3.0_f32);
sparse_set.add_triplet(1, 2, 2.0);
sparse_set.add_triplet(3, 0, - 2.0);
let sparse_set: sprs::CsMat<_ > = sparse_set.to_csr();

let mut dataset = AnnDataset::<f32>::new();
dataset.add("dense", VectorSet::Dense(dense_set))?;
dataset.add("sparse", VectorSet::Sparse(sparse_set.clone()))?;

dataset.write("dataset.hdf5");
```

## Reading an ANN dataset
```rust
let dataset = AnnDataset::<f32>::load("dataset.hdf5") ?;
println!("{}", dataset);

let dense_set = dataset.read("dense_set")?;
let sparse_set = dataset.read("sparse_set")?;
```
