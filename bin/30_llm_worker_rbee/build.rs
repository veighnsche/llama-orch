// TEAM-XXX: Generate build metadata using shadow-rs

fn main() {
    shadow_rs::new().expect("Failed to generate shadow-rs build metadata");
}
