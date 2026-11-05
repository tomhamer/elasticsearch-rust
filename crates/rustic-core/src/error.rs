use thiserror::Error;

#[derive(Error, Debug)]
pub enum RusticError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Index not found: {0}")]
    IndexNotFound(String),

    #[error("Field not found: {0}")]
    FieldNotFound(String),

    #[error("Invalid query: {0}")]
    InvalidQuery(String),

    #[error("maxClauseCount is set to {max}")]
    TooManyClauses { max: usize, nested: bool },

    #[error("{0}")]
    IntervalExpansion(String),

    #[error("Segment error: {0}")]
    Segment(String),

    #[error("Memory allocation failed: {0}")]
    Memory(String),

    #[error("Corrupted index: {0}")]
    Corrupted(String),

    #[error("Unsupported operation: {0}")]
    Unsupported(String),

    /// A long operation was cancelled ([`crate::cancel`]).
    #[error("Cancelled: {0}")]
    Cancelled(String),
}

pub type Result<T> = std::result::Result<T, RusticError>;
