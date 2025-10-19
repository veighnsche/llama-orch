use crate::template::{generate_template_code, validate_template};
use proc_macro::TokenStream;
use quote::quote;
use syn::parse::Parser;
use syn::{parse_macro_input, Expr, ItemFn, Lit, Meta, MetaNameValue};

/// Parse narration attributes from the macro input.
///
/// Expected format:
/// ```ignore
/// #[narrate(
///     action: "dispatch",
///     human: "Dispatched job {job_id} to worker {worker_id}",
///     cute: "Sent job {job_id} off to its new friend {worker_id}! 🎫",
///     story: "'Ready!' said worker {worker_id}"
/// )]
/// ```
struct NarrateAttrs {
    action: Option<syn::LitStr>,
    human: Option<syn::LitStr>,
    cute: Option<syn::LitStr>,
    story: Option<syn::LitStr>,
}

impl NarrateAttrs {
    fn parse(attr: TokenStream) -> syn::Result<Self> {
        let mut attrs = NarrateAttrs { action: None, human: None, cute: None, story: None };

        // Parse as TokenStream2 for better error handling
        let attr2: proc_macro2::TokenStream = attr.into();

        // Parse comma-separated key: value pairs
        let parsed =
            syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated.parse2(attr2)?;

        for meta in parsed {
            if let Meta::NameValue(MetaNameValue { path, value, .. }) = meta {
                let key = path
                    .get_ident()
                    .ok_or_else(|| syn::Error::new_spanned(&path, "expected identifier"))?
                    .to_string();

                if let Expr::Lit(expr_lit) = value {
                    if let Lit::Str(lit_str) = &expr_lit.lit {
                        match key.as_str() {
                            "action" => attrs.action = Some(lit_str.clone()),
                            "human" => attrs.human = Some(lit_str.clone()),
                            "cute" => attrs.cute = Some(lit_str.clone()),
                            "story" => attrs.story = Some(lit_str.clone()),
                            _ => {
                                return Err(syn::Error::new_spanned(
                                    path,
                                    format!("unknown attribute key: {}", key),
                                ))
                            }
                        }
                    } else {
                        return Err(syn::Error::new_spanned(expr_lit, "expected string literal"));
                    }
                } else {
                    return Err(syn::Error::new_spanned(value, "expected string literal"));
                }
            } else {
                return Err(syn::Error::new_spanned(meta, "expected key: value pair"));
            }
        }

        Ok(attrs)
    }
}

/// Implementation of #[narrate(...)] attribute macro.
///
/// Generates narration events with template interpolation.
pub fn narrate_impl(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);

    // Parse attributes
    let attrs = match NarrateAttrs::parse(attr) {
        Ok(attrs) => attrs,
        Err(err) => return err.to_compile_error().into(),
    };

    // Validate required fields
    let action = match attrs.action {
        Some(action) => action,
        None => {
            return syn::Error::new(
                proc_macro2::Span::call_site(),
                "narrate macro requires 'action' attribute",
            )
            .to_compile_error()
            .into();
        }
    };

    let human = match attrs.human {
        Some(human) => human,
        None => {
            return syn::Error::new(
                proc_macro2::Span::call_site(),
                "narrate macro requires 'human' attribute",
            )
            .to_compile_error()
            .into();
        }
    };

    // Validate templates
    if let Err(err) = validate_template(&human.value()) {
        return syn::Error::new(human.span(), format!("invalid human template: {}", err))
            .to_compile_error()
            .into();
    }

    if let Some(ref cute) = attrs.cute {
        if let Err(err) = validate_template(&cute.value()) {
            return syn::Error::new(cute.span(), format!("invalid cute template: {}", err))
                .to_compile_error()
                .into();
        }
    }

    if let Some(ref story) = attrs.story {
        if let Err(err) = validate_template(&story.value()) {
            return syn::Error::new(story.span(), format!("invalid story template: {}", err))
                .to_compile_error()
                .into();
        }
    }

    // Generate template interpolation code
    let human_code = generate_template_code(&human);
    let cute_code = attrs.cute.as_ref().map(generate_template_code);
    let story_code = attrs.story.as_ref().map(generate_template_code);

    // Extract function components
    let fn_vis = &input_fn.vis;
    let fn_sig = &input_fn.sig;
    let fn_block = &input_fn.block;
    let fn_attrs = &input_fn.attrs;

    // Infer actor from module path at compile time
    // We'll use module_path!() which is available at runtime
    let actor_inference = quote! {
        {
            let module = module_path!();
            let actor = observability_narration_core::extract_service_name(module);
            actor
        }
    };

    // Check if function is async
    let is_async = fn_sig.asyncness.is_some();

    // Generate narration call
    let cute_field = if let Some(ref code) = cute_code {
        quote! { cute: Some(#code), }
    } else {
        quote! { cute: None, }
    };

    let story_field = if let Some(ref code) = story_code {
        quote! { story: Some(#code), }
    } else {
        quote! { story: None, }
    };

    let narration_call = quote! {
        {
            let __human = #human_code;

            observability_narration_core::narrate(
                observability_narration_core::NarrationFields {
                    actor: #actor_inference,
                    action: #action,
                    target: String::new(), // Could be inferred from function args
                    human: __human,
                    #cute_field
                    #story_field
                    ..Default::default()
                }
            );
        }
    };

    // Generate the instrumented function
    let output = if is_async {
        quote! {
            #(#fn_attrs)*
            #fn_vis #fn_sig {
                #narration_call
                async move #fn_block.await
            }
        }
    } else {
        quote! {
            #(#fn_attrs)*
            #fn_vis #fn_sig {
                #narration_call
                (|| #fn_block)()
            }
        }
    };

    TokenStream::from(output)
}
