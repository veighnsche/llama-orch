use napi::bindgen_prelude::*;

// N-API wrappers mirroring TS shapes
#[napi(object)]
pub struct ReadRequestNapi {
    pub paths: Vec<String>,
    pub as_text: bool,
    pub encoding: Option<String>,
}

#[napi(object)]
pub struct FileBlobNapi {
    pub path: String,
    pub content: Option<String>,
    pub bytes: Option<Vec<u8>>, // Represented as number[] in JS
}

#[napi(object)]
pub struct ReadResponseNapi {
    pub files: Vec<FileBlobNapi>,
}

// Conversions (pure, total)
impl From<ReadRequestNapi> for llama_orch_utils::fs::file_reader::ReadRequest {
    fn from(value: ReadRequestNapi) -> Self {
        llama_orch_utils::fs::file_reader::ReadRequest {
            paths: value.paths,
            as_text: value.as_text,
            encoding: value.encoding,
        }
    }
}

impl From<llama_orch_utils::fs::file_reader::FileBlob> for FileBlobNapi {
    fn from(value: llama_orch_utils::fs::file_reader::FileBlob) -> Self {
        FileBlobNapi {
            path: value.path,
            content: value.content,
            bytes: value.bytes,
        }
    }
}

impl From<llama_orch_utils::fs::file_reader::ReadResponse> for ReadResponseNapi {
    fn from(value: llama_orch_utils::fs::file_reader::ReadResponse) -> Self {
        ReadResponseNapi {
            files: value.files.into_iter().map(FileBlobNapi::from).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn request_napi_to_core_parity() {
        let req = ReadRequestNapi {
            paths: vec!["a.txt".into(), "b.bin".into()],
            as_text: true,
            encoding: None,
        };
        let core: llama_orch_utils::fs::file_reader::ReadRequest = req.into();
        assert_eq!(core.paths, vec!["a.txt", "b.bin"]);
        assert!(core.as_text);
        assert_eq!(core.encoding, None);
    }

    #[test]
    fn response_core_to_napi_preserves_order_and_values() {
        // Create mixed FileBlob content
        let dir = tempdir().unwrap();
        let p1 = dir.path().join("x.txt");
        let p2 = dir.path().join("y.bin");
        fs::File::create(&p1).unwrap().write_all(b"hello").unwrap();
        fs::File::create(&p2).unwrap().write_all(&[1u8, 2, 3]).unwrap();

        let core = llama_orch_utils::fs::file_reader::ReadResponse {
            files: vec![
                llama_orch_utils::fs::file_reader::FileBlob {
                    path: p1.to_string_lossy().to_string(),
                    content: Some("hello".to_string()),
                    bytes: None,
                },
                llama_orch_utils::fs::file_reader::FileBlob {
                    path: p2.to_string_lossy().to_string(),
                    content: None,
                    bytes: Some(vec![1, 2, 3]),
                },
            ],
        };

        let napi_resp: ReadResponseNapi = core.into();
        assert_eq!(napi_resp.files.len(), 2);
        assert!(napi_resp.files[0].content.as_deref() == Some("hello"));
        assert!(napi_resp.files[0].bytes.is_none());
        assert_eq!(napi_resp.files[1].bytes.as_ref().unwrap(), &vec![1, 2, 3]);
        assert!(napi_resp.files[1].content.is_none());
    }
}
