//! Timeout Enforcer Macros - Ergonomic attribute macro for timeout enforcement
//!
//! Created by: TEAM-330
//!
//! # Purpose
//! Provides `#[with_timeout]` attribute macro to wrap async functions with
//! automatic timeout enforcement. This is syntactic sugar over the core
//! `TimeoutEnforcer` struct.
//!
//! # Design Philosophy
//! - **Core remains king**: `TimeoutEnforcer` struct is the source of truth
//! - **Macro is sugar**: Just reduces call-site boilerplate
//! - **Policy enforcement**: Makes timeouts mandatory by default
//! - **Zero runtime cost**: Expands to the same code you'd write manually
//!
//! # Usage
//! ```rust,ignore
//! use timeout_enforcer::with_timeout;
//! use anyhow::Result;
//!
//! #[with_timeout(secs = 10, label = "Slow operation")]
//! pub async fn slow_operation() -> Result<String> {
//!     tokio::time::sleep(std::time::Duration::from_secs(5)).await;
//!     Ok("done".into())
//! }
//! ```
//!
//! # Expands to
//! ```rust,ignore
//! pub async fn slow_operation() -> Result<String> {
//!     async fn __slow_operation_inner() -> Result<String> {
//!         tokio::time::sleep(std::time::Duration::from_secs(5)).await;
//!         Ok("done".into())
//!     }
//!     
//!     timeout_enforcer::TimeoutEnforcer::new(Duration::from_secs(10))
//!         .with_label("Slow operation")
//!         .enforce(__slow_operation_inner())
//!         .await
//! }
//! ```

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::Parser, parse_macro_input, punctuated::Punctuated, Expr, ExprLit, FnArg, ItemFn, Lit,
    Meta, Pat, Token,
};

/// Attribute macro to enforce timeouts on async functions
///
/// # Arguments
/// - `secs` (required): Timeout duration in seconds
/// - `label` (optional): Human-readable label for the operation
/// - `countdown` (optional): Show visual progress bar (default: false)
///
/// # Examples
///
/// ## Basic usage
/// ```rust,ignore
/// #[with_timeout(secs = 30)]
/// async fn my_operation() -> Result<()> {
///     // ... operation ...
///     Ok(())
/// }
/// ```
///
/// ## With label
/// ```rust,ignore
/// #[with_timeout(secs = 45, label = "Starting hive")]
/// async fn start_hive() -> Result<()> {
///     // ... operation ...
///     Ok(())
/// }
/// ```
///
/// ## With countdown
/// ```rust,ignore
/// #[with_timeout(secs = 60, label = "Long operation", countdown = true)]
/// async fn long_operation() -> Result<()> {
///     // ... operation ...
///     Ok(())
/// }
/// ```
///
/// # Requirements
/// - Function must be `async`
/// - Function must return `anyhow::Result<T>` (or compatible Result type)
/// - `timeout-enforcer` must be in scope
///
/// # Context Propagation
/// The macro works with `NarrationContext` - if you wrap the call site in
/// `with_narration_context()`, the timeout narration will include job_id:
///
/// ```rust,ignore
/// let ctx = NarrationContext::new().with_job_id(&job_id);
/// with_narration_context(ctx, async {
///     // Timeout narration automatically includes job_id!
///     my_operation().await?;
///     Ok(())
/// }).await
/// ```
#[proc_macro_attribute]
pub fn with_timeout(args: TokenStream, item: TokenStream) -> TokenStream {
    // Parse args: #[with_timeout(secs = 30, label = "Starting queen-rbee", countdown = true)]
    let parser = Punctuated::<Meta, Token![,]>::parse_terminated;
    let args = match parser.parse(args.clone().into()) {
        Ok(args) => args,
        Err(e) => return e.to_compile_error().into(),
    };

    let mut secs: Option<u64> = None;
    let mut label: Option<String> = None;
    let mut countdown: bool = false;

    for arg in args {
        match arg {
            Meta::NameValue(nv) if nv.path.is_ident("secs") => {
                if let Expr::Lit(ExprLit { lit: Lit::Int(i), .. }) = nv.value {
                    secs = Some(i.base10_parse().unwrap());
                }
            }
            Meta::NameValue(nv) if nv.path.is_ident("label") => {
                if let Expr::Lit(ExprLit { lit: Lit::Str(s), .. }) = nv.value {
                    label = Some(s.value());
                }
            }
            Meta::NameValue(nv) if nv.path.is_ident("countdown") => {
                if let Expr::Lit(ExprLit { lit: Lit::Bool(b), .. }) = nv.value {
                    countdown = b.value;
                }
            }
            _ => {}
        }
    }

    let secs = match secs {
        Some(s) => s,
        None => {
            return syn::Error::new(
                proc_macro2::Span::call_site(),
                "#[with_timeout] requires `secs = <u64>` argument",
            )
            .to_compile_error()
            .into();
        }
    };

    let func = parse_macro_input!(item as ItemFn);

    // Ensure it's async
    if func.sig.asyncness.is_none() {
        return syn::Error::new_spanned(&func.sig.fn_token, "#[with_timeout] requires an async fn")
            .to_compile_error()
            .into();
    }

    // Extract function components
    let vis = &func.vis;
    let attrs = &func.attrs;
    let sig = &func.sig;
    let name = &func.sig.ident;
    let generics = &func.sig.generics;
    let inputs = &func.sig.inputs;
    let output = &func.sig.output;
    let where_clause = &func.sig.generics.where_clause;
    let body = &func.block;

    // Create inner function name
    let inner_name = format_ident!("__{}_inner", name);

    // Extract parameter names for forwarding
    let param_names: Vec<_> = inputs
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(pat_type) = arg {
                if let Pat::Ident(pat_ident) = &*pat_type.pat {
                    return Some(&pat_ident.ident);
                }
            }
            None
        })
        .collect();

    // Build label chain
    let label_chain = if let Some(lbl) = label {
        quote! { .with_label(#lbl) }
    } else {
        quote! {}
    };

    // Build countdown chain
    let countdown_chain = if countdown {
        quote! { .with_countdown() }
    } else {
        quote! {}
    };

    // Generate the wrapped function
    let expanded = quote! {
        // Keep original attributes/visibility/signature
        #(#attrs)*
        #vis #sig {
            // Inner function keeps the original body
            async fn #inner_name #generics ( #inputs ) #output #where_clause {
                #body
            }

            // The wrapper enforces the timeout on the inner call
            {
                use std::time::Duration;
                timeout_enforcer::TimeoutEnforcer::new(Duration::from_secs(#secs))
                    #label_chain
                    #countdown_chain
                    .enforce(#inner_name( #(#param_names),* ))
                    .await
            }
        }
    };

    TokenStream::from(expanded)
}
