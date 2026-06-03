use thiserror::Error;

#[derive(Error, Debug)]
pub enum HprofError {
    #[error("Failed to open HPROF file: {0}")]
    FileOpen(#[source] std::io::Error),

    #[error("HPROF file is corrupted: {0}")]
    Corrupted(String),

    #[error("Invalid HPROF format: {0}")]
    InvalidFormat(String),

    #[error("HPROF read error: {0}")]
    ReadError(#[source] std::io::Error),

    #[error("Unsupported HPROF version: {0}")]
    UnsupportedVersion(String),

    #[error("Object count overflow: exceeds i32 max")]
    ObjectOverflow,
}
