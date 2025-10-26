// TEAM-309: Simple actor injection macro
//! Proc macro to inject actor into function scope

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn, LitStr};

/// Attribute macro to set narration actor for entire function.
///
/// All n!() calls inside automatically use this actor.
///
/// # Example
/// ```ignore
/// use narration_macros::with_actor;
///
/// #[with_actor("auto-update")]
/// fn needs_rebuild() -> Result<bool> {
///     n!("check", "Checking..."); // Uses actor="auto-update"
///     n!("found", "Found {} deps", count); // Also uses actor="auto-update"
///     Ok(true)
/// }
/// ```
pub fn with_actor_impl(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let actor = parse_macro_input!(attr as LitStr);
    
    let fn_vis = &input_fn.vis;
    let fn_sig = &input_fn.sig;
    let fn_block = &input_fn.block;
    let fn_attrs = &input_fn.attrs;
    
    let is_async = fn_sig.asyncness.is_some();
    
    let output = if is_async {
        quote! {
            #(#fn_attrs)*
            #fn_vis #fn_sig {
                observability_narration_core::__internal_set_actor(#actor);
                let __result = async #fn_block.await;
                observability_narration_core::__internal_clear_actor();
                __result
            }
        }
    } else {
        quote! {
            #(#fn_attrs)*
            #fn_vis #fn_sig {
                observability_narration_core::__internal_set_actor(#actor);
                let __result = (|| #fn_block)();
                observability_narration_core::__internal_clear_actor();
                __result
            }
        }
    };
    
    TokenStream::from(output)
}
