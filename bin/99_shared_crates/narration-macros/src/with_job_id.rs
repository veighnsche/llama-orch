// TEAM-328: Attribute macro to eliminate job_id context boilerplate
//! Attribute macro to automatically wrap async functions with narration context
//!
//! Eliminates the repetitive pattern of:
//! ```ignore
//! let ctx = config.job_id.as_ref().map(|jid| NarrationContext::new().with_job_id(jid));
//! let impl_fn = async { /* actual logic */ };
//! if let Some(ctx) = ctx {
//!     with_narration_context(ctx, impl_fn).await
//! } else {
//!     impl_fn.await
//! }
//! ```
//!
//! # Usage
//! ```ignore
//! #[with_job_id(config_param = "rebuild_config")]
//! pub async fn rebuild_with_hot_reload(
//!     rebuild_config: RebuildConfig,
//!     daemon_config: HttpDaemonConfig,
//! ) -> Result<bool> {
//!     // Your actual implementation here
//!     // n!() calls will automatically use job_id from rebuild_config
//!     Ok(true)
//! }
//! ```
//!
//! The macro will:
//! 1. Extract `job_id` from the specified config parameter
//! 2. Create NarrationContext if job_id exists
//! 3. Wrap your function body with `with_narration_context` if needed
//! 4. Otherwise execute directly

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, parse_quote, ItemFn, Meta};

pub fn with_job_id_impl(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input_fn = parse_macro_input!(item as ItemFn);

    // Parse attribute to get config parameter name
    let config_param = if attr.is_empty() {
        // Default: look for first parameter with "config" in the name
        None
    } else {
        let meta = parse_macro_input!(attr as Meta);
        match meta {
            Meta::NameValue(nv) => {
                if nv.path.is_ident("config_param") {
                    if let syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(s), .. }) = &nv.value {
                        Some(s.value())
                    } else {
                        panic!("#[with_job_id] config_param must be a string literal");
                    }
                } else {
                    panic!("#[with_job_id] only supports config_param attribute");
                }
            }
            _ => panic!("#[with_job_id] expects config_param = \"param_name\""),
        }
    };

    // Find the config parameter
    let config_ident = if let Some(name) = config_param {
        syn::Ident::new(&name, proc_macro2::Span::call_site())
    } else {
        // Auto-detect: find first parameter with "config" in name
        input_fn
            .sig
            .inputs
            .iter()
            .find_map(|arg| {
                if let syn::FnArg::Typed(pat_type) = arg {
                    if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                        let name = pat_ident.ident.to_string();
                        if name.contains("config") {
                            return Some(pat_ident.ident.clone());
                        }
                    }
                }
                None
            })
            .expect("#[with_job_id] requires a parameter with 'config' in the name, or specify config_param")
    };

    // TEAM-350: FIX - Properly wrap in with_narration_context()
    // TEAM-335 broke this by just setting a variable that nothing uses
    //
    // CRITICAL: We cannot create nested async blocks! That causes infinite recursion.
    // Instead, we create an inner sync function that returns the Future, then wrap that.
    
    let vis = &input_fn.vis;
    let sig = &input_fn.sig;
    let fn_name = &sig.ident;
    let inputs = &sig.inputs;
    let output = &sig.output;
    let generics = &sig.generics;
    let where_clause = &sig.generics.where_clause;
    let original_block = &input_fn.block;
    
    // Create inner function name
    let inner_name = syn::Ident::new(&format!("__{}_inner", fn_name), fn_name.span());

    // TEAM-350: Transform to avoid nested async blocks
    let expanded: proc_macro2::TokenStream = quote! {
        #vis #sig {
            // Inner async function with original logic
            async fn #inner_name #generics ( #inputs ) #output #where_clause {
                #original_block
            }
            
            // TEAM-350: Wrap with context if job_id exists
            if let Some(job_id) = #config_ident.job_id.as_ref() {
                let ctx = observability_narration_core::NarrationContext::new().with_job_id(job_id);
                observability_narration_core::with_narration_context(ctx, #inner_name(#config_ident)).await
            } else {
                // No job_id, call inner function directly
                #inner_name(#config_ident).await
            }
        }
    };

    TokenStream::from(expanded)
}
