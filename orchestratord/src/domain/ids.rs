#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TaskId(pub String);
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionId(pub String);
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArtifactId(pub String);

impl From<&str> for TaskId { fn from(s: &str) -> Self { Self(s.into()) } }
impl From<&str> for SessionId { fn from(s: &str) -> Self { Self(s.into()) } }
impl From<&str> for ArtifactId { fn from(s: &str) -> Self { Self(s.into()) } }
