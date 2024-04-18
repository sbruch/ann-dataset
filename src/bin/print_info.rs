use clap::Parser;
use ann_dataset::{AnnDataset, Hdf5File};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Path to an hdf5 file containing the data.
    #[clap(long, required = true)]
    path: String,
}

fn main() {
    let args = Args::parse();
    let dataset = AnnDataset::<f32>::read(args.path.as_str())
        .expect("Unable to load the dataset");
    println!("{}", dataset);
}
