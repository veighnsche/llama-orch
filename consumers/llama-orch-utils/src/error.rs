use std::error::Error as StdError;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone)]
pub enum Error {
    Unimplemented { message: String },
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Unimplemented { message } => write!(f, "{}", message),
        }
    }
}

impl StdError for Error {} 
