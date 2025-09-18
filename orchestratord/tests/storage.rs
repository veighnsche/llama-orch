use orchestratord::infra::storage::inmem::InMemStore;
use orchestratord::infra::storage::fs::FsStore;
use orchestratord::ports::storage::ArtifactStore;

#[test]
fn inmem_store_round_trip_and_id_prefix() {
    let store = InMemStore::default();
    let doc = serde_json::json!({"k":"v","n":1});
    let id = store.put(doc.clone()).unwrap();
    assert!(id.starts_with("sha256:"));
    let got = store.get(&id).unwrap().unwrap();
    assert_eq!(got, doc);
}

#[test]
fn fs_store_round_trip_and_file_written() {
    let td = tempfile::tempdir().unwrap();
    let fs = FsStore::new(td.path()).unwrap();
    let doc = serde_json::json!({"a": [1,2,3], "b": {"x": true}});
    let id = fs.put(doc.clone()).unwrap();
    assert!(id.starts_with("sha256:"));

    // File exists at <root>/<hex>
    let hex = id.strip_prefix("sha256:").unwrap();
    let path = td.path().join(hex);
    assert!(path.exists());

    // Read back via store API
    let got = fs.get(&id).unwrap().unwrap();
    assert_eq!(got, doc);
}
