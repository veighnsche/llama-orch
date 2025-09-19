//! observability-narration-core — shared, lightweight narration helper.

use tracing::field::display;
use tracing::{event, Level};

pub fn human<S: AsRef<str>>(actor: &str, action: &str, target: &str, msg: S) {
    event!(
        Level::INFO,
        actor = display(actor),
        action = display(action),
        target = display(target),
        human = display(msg.as_ref()),
    );
}
