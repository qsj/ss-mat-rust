/// Class category for traversal classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClassCategory {
    String,
    StringValue,
    Class,
    Thread,
    ClassLoader,
    Collection,
    Map,
    Array,
    BoxedNumber,
    DirectByteBuffer,
    WeakReference,
    Other,
}
