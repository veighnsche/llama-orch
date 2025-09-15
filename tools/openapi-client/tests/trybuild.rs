#[test]
fn ui_passes() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/*.rs");
}
