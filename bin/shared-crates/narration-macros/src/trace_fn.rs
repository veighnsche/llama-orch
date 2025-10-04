use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

/// Implementation of #[trace_fn] attribute macro.
/// 
/// Generates entry/exit traces with automatic timing.
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
    
    // Generate the instrumented function
    let output = if is_async {
        quote! {
            #(#fn_attrs)*
            #fn_vis #fn_sig {
                #[cfg(feature = "trace-enabled")]
                {
                    use std::time::Instant;
                    let __trace_start = Instant::now();
                    
                    tracing::trace!(
                        action = "enter",
                        target = #fn_name_str,
                        human = concat!("ENTER ", #fn_name_str),
                    );
                    
                    let __result = async move #fn_block.await;
                    
                    let __elapsed_ms = __trace_start.elapsed().as_millis();
                    tracing::trace!(
                        action = "exit",
                        target = #fn_name_str,
                        human = format!("EXIT {} ({}ms)", #fn_name_str, __elapsed_ms),
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
                    
                    tracing::trace!(
                        action = "enter",
                        target = #fn_name_str,
                        human = concat!("ENTER ", #fn_name_str),
                    );
                    
                    let __result = (|| #fn_block)();
                    
                    let __elapsed_ms = __trace_start.elapsed().as_millis();
                    tracing::trace!(
                        action = "exit",
                        target = #fn_name_str,
                        human = format!("EXIT {} ({}ms)", #fn_name_str, __elapsed_ms),
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
