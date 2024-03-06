use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, AttributeArgs, ItemFn};

#[proc_macro_attribute]
pub fn local_alloc_disabled(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input_fn = parse_macro_input!(item as ItemFn);
    let args = parse_macro_input!(attr as AttributeArgs);

    if !args.is_empty() {
        panic!("The `with_local_alloc_disabled` attribute does not take any arguments.");
    }

    let fn_block = &input_fn.block;
    let output_fn = quote! {
        {
            let _guard = crate::vm::LocalAllocGuard::new();
            #fn_block
        }
    };
    input_fn.block = syn::parse2(output_fn).unwrap();
    quote! { #input_fn }.into()
}
