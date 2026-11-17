//! OffconAI Desktop: Native Rust + Slint workspace over typed local/gateway LLM backends.

pub mod backends;
pub mod catalog;
pub mod contracts;
pub mod controller;
pub mod epoch;
pub mod image_backend;
pub mod mcp;
pub mod persist;
pub mod process;
pub mod session;
pub mod transport;

slint::include_modules!();
