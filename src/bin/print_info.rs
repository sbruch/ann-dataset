use ann_dataset::{Hdf5File, InMemoryAnnDataset};
use clap::Parser;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Path to an hdf5 file containing the data.
    #[clap(long, required = true)]
    path: String,
}

fn main() {
    let args = Args::parse();
    let dataset =
        InMemoryAnnDataset::<f32>::read(args.path.as_str()).expect("Unable to load the dataset");
    println!("{}", dataset);
}
