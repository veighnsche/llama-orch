#[cfg(feature = "ts-types")]
fn main() {
    if let Err(e) = llama_orch_utils::export_ts_types() {
        eprintln!("export_ts_types failed: {}", e);
        std::process::exit(1);
    }
}

#[cfg(not(feature = "ts-types"))]
fn main() {
    eprintln!("error: export-ts binary requires --features ts-types");
    std::process::exit(1);
}
