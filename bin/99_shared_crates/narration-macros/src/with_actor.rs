// TEAM-309: Function name injection for scoped narration
//! Proc macro to inject function name as narration target

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

/// Attribute macro to set function name as narration target.
///
/// All n!() calls inside automatically use function name as target.
/// Actor is still auto-detected from crate name.
///
/// # Example
/// ```ignore
/// use narration_macros::narrate_fn;
///
/// #[narrate_fn]
/// fn rebuild() -> Result<()> {
///     n!("start", "Starting...");   // actor="auto-update", target="rebuild"
///     n!("success", "Done!");        // actor="auto-update", target="rebuild"
///     Ok(())
/// }
/// ```
///
/// Output format: [crate/function] action : message
pub fn narrate_fn_impl(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    
    let fn_vis = &input_fn.vis;
    let fn_sig = &input_fn.sig;
    let fn_block = &input_fn.block;
    let fn_attrs = &input_fn.attrs;
    let fn_name = &fn_sig.ident;
    let fn_name_str = fn_name.to_string();
    
    let is_async = fn_sig.asyncness.is_some();
    
    let output = if is_async {
        quote! {
            #(#fn_attrs)*
            #fn_vis #fn_sig {
                observability_narration_core::__internal_set_target(#fn_name_str);
                let __result = async #fn_block.await;
                observability_narration_core::__internal_clear_target();
                __result
            }
        }
    } else {
        quote! {
            #(#fn_attrs)*
            #fn_vis #fn_sig {
                observability_narration_core::__internal_set_target(#fn_name_str);
                let __result = (|| #fn_block)();
                observability_narration_core::__internal_clear_target();
                __result
            }
        }
    };
    
    TokenStream::from(output)
}
