use hdf5::Group;

pub trait Hdf5Serialization {
    type Object;

    /// Adds `Object` to the given HDF5 `group`.
    fn write(&self, group: &mut Group) -> anyhow::Result<()>;

    /// Deserializes `group` into the `Object`.
    fn read(group: &Group) -> anyhow::Result<Self::Object>;

    /// Returns the label of `Object` in the HDF5 file.
    fn label() -> String;
}
