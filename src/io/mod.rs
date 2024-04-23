use hdf5::Group;

pub trait Hdf5Serialization {
    type Object;

    /// Adds `Object` to the given HDF5 `group`.
    fn add_to(&self, group: &mut Group) -> anyhow::Result<()>;

    /// Deserializes `group` into the `Object`.
    fn read_from(group: &Group) -> anyhow::Result<Self::Object>;

    /// Returns the label of `Object` in the HDF5 file.
    fn label() -> String;
}

pub trait Hdf5File {
    type Object;

    /// Stores `Object` as an HDF5 file at `path`.
    fn write(&self, path: &str) -> anyhow::Result<()>;

    /// Reads `Object` from HDF5 file at `path`.
    fn read(path: &str) -> anyhow::Result<Self::Object>;
}
