use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

/// Implementation of #[trace_fn] attribute macro.
/// 
/// Generates entry/exit traces with automatic timing using the narration system.
pub fn trace_fn_impl(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    
    let fn_name = &input_fn.sig.ident;
    let fn_name_str = fn_name.to_string();
    let fn_vis = &input_fn.vis;
    let fn_sig = &input_fn.sig;
    let fn_block = &input_fn.block;
    let fn_attrs = &input_fn.attrs;
    
    // Check if function is async
    let is_async = fn_sig.asyncness.is_some();
    
    // Infer actor from module path
    let actor_inference = quote! {
        {
            let module = module_path!();
            observability_narration_core::extract_service_name(module)
        }
    };
    
    // Generate the instrumented function
    let output = if is_async {
        quote! {
            #(#fn_attrs)*
            #fn_vis #fn_sig {
                #[cfg(feature = "trace-enabled")]
                {
                    use std::time::Instant;
                    let __trace_start = Instant::now();
                    let __actor = #actor_inference;
                    
                    observability_narration_core::trace::trace_enter(
                        __actor,
                        #fn_name_str,
                    );
                    
                    let __result = async move #fn_block.await;
                    
                    let __elapsed_ms = __trace_start.elapsed().as_millis() as u64;
                    observability_narration_core::trace::trace_exit(
                        __actor,
                        #fn_name_str,
                        __elapsed_ms,
                    );
                    
                    __result
                }
                
                #[cfg(not(feature = "trace-enabled"))]
                {
                    async move #fn_block.await
                }
            }
        }
    } else {
        quote! {
            #(#fn_attrs)*
            #fn_vis #fn_sig {
                #[cfg(feature = "trace-enabled")]
                {
                    use std::time::Instant;
                    let __trace_start = Instant::now();
                    let __actor = #actor_inference;
                    
                    observability_narration_core::trace::trace_enter(
                        __actor,
                        #fn_name_str,
                    );
                    
                    let __result = (|| #fn_block)();
                    
                    let __elapsed_ms = __trace_start.elapsed().as_millis() as u64;
                    observability_narration_core::trace::trace_exit(
                        __actor,
                        #fn_name_str,
                        __elapsed_ms,
                    );
                    
                    __result
                }
                
                #[cfg(not(feature = "trace-enabled"))]
                #fn_block
            }
        }
    };
    
    TokenStream::from(output)
}
