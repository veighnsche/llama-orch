// TEAM-286: Utility functions for WASM

use wasm_bindgen::prelude::*;

/// Log to browser console
///
/// TEAM-286: Helper for debugging
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

/// Log macro for easier usage
#[macro_export]
macro_rules! console_log {
    ($($t:tt)*) => {
        $crate::utils::log(&format!($($t)*))
    };
}
