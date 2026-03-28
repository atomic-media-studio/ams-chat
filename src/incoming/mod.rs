//! Inbound message classification and dispatch rules (HTTP vs UI vs system).

mod source;

pub use source::{should_dispatch_to_model, MessageSource};
