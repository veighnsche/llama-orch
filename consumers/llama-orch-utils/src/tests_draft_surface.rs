#![allow(unused_imports)]
#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    // fs/file_reader
    use crate::fs::file_reader::{FileBlob, ReadRequest, ReadResponse};
    use crate::fs::file_reader as fs_file_reader;

    // fs/file_writer
    use crate::fs::file_writer::{WriteIn, WriteOut};
    use crate::fs::file_writer as fs_file_writer;

    // prompt/message
    use crate::prompt::message::{Message, MessageIn, Source};
    use crate::prompt::message as prompt_message;

    // prompt/thread
    use crate::prompt::thread::{ThreadIn, ThreadItem, ThreadOut};
    use crate::prompt::thread as prompt_thread;

    // model/define
    use crate::model::define::{ModelRef};
    use crate::model::define as model_define;

    // params/define
    use crate::params::define::{Params};
    use crate::params::define as params_define;

    // llm/invoke
    use crate::llm::invoke::{Choice, InvokeIn, InvokeOut, InvokeResult, Usage};
    use crate::llm::invoke as llm_invoke;
    use llama_orch_sdk::client::OrchestratorClient;

    // orch/response_extractor
    use crate::orch::response_extractor as response_extractor;

    #[test]
    fn draft_surface_compiles() {
        // Touch types to avoid unused warnings
        let _ = (
            size_of::<ReadRequest>(),
            size_of::<FileBlob>(),
            size_of::<ReadResponse>(),
            size_of::<WriteIn>(),
            size_of::<WriteOut>(),
            size_of::<MessageIn>(),
            size_of::<Message>(),
            size_of::<Source>(),
            size_of::<ThreadItem>(),
            size_of::<ThreadIn>(),
            size_of::<ThreadOut>(),
            size_of::<ModelRef>(),
            size_of::<Params>(),
            size_of::<InvokeIn>(),
            size_of::<InvokeOut>(),
            size_of::<InvokeResult>(),
            size_of::<Choice>(),
            size_of::<Usage>(),
        );

        // Assert function signatures via typed function pointers (no calls)
        let _fr_run: fn(ReadRequest) -> std::io::Result<ReadResponse> = fs_file_reader::run;
        let _fw_run: fn(WriteIn) -> std::io::Result<WriteOut> = fs_file_writer::run;
        let _pm_run: fn(MessageIn) -> std::io::Result<Message> = prompt_message::run;
        let _pt_run: fn(ThreadIn) -> std::io::Result<ThreadOut> = prompt_thread::run;
        let _md_run: fn(String, Option<String>, Option<String>) -> ModelRef = model_define::run;
        let _pd_run: fn(Params) -> Params = params_define::run;
        let _li_run: fn(&OrchestratorClient, InvokeIn) -> Result<InvokeOut, crate::error::Error> = llm_invoke::run;
        let _or_run: fn(&InvokeResult) -> String = response_extractor::run;

        let _ = (_fr_run, _fw_run, _pm_run, _pt_run, _md_run, _pd_run, _li_run, _or_run);
    }
}
