use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(LuaRef, attributes(lua))]
pub fn derive_lua_reference(input: TokenStream) -> TokenStream {
    let krate = if cfg!(feature = "internal") {
        quote!(crate)
    } else {
        quote!(::engine)
    };

    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let ref_name = quote::format_ident!("{}Ref", name);
    let name_str = name.to_string();

    let mut user_data_fields = Vec::new();
    let fields = match &input.data {
        syn::Data::Struct(data_struct) => &data_struct.fields,
        _ => panic!("LuaExport only supports structs"),
    };

    let no_default = input.attrs.iter().any(|attr| {
        if !attr.path().is_ident("lua") {
            return false;
        }
        attr.parse_args::<syn::Ident>()
            .map(|ident| ident == "no_default")
            .unwrap_or(false)
    });

    let default_construct = if no_default {
        quote!(None)
    } else {
        quote!(Some(|| Box::new(<#name>::default()) as Box<dyn #krate::ecs::DynamicComponent>))
    };

    let mut field_info = Vec::new();
    let mut lua_types = Vec::new();

    for field in fields {
        let skip = field.attrs.iter().any(|attr| {
            if !attr.path().is_ident("lua") {
                return false;
            }
            attr.parse_args::<syn::Ident>()
                .map(|ident| ident == "skip")
                .unwrap_or(false)
        });

        if skip {
            continue;
        }

        let field_name = &field.ident;
        let field_type = &field.ty;
        let field_name_str = field_name.as_ref().unwrap().to_string();
        let field_type_str = get_field_type_string_name(field_type);
        field_info.push(quote! {
            (#field_name_str, #field_type_str),
        });
        lua_types.push(quote! {(#field_name_str, #field_type_str),});

        let field_tokens = quote! {
            fields.add_field_method_set(
                #field_name_str,
                |_lua, this, val: ::mlua::Value| -> ::mlua::Result<()> {
                    let result = #krate::lua::LuaSeralize::from_lua(&val)?;
                    unsafe {
                        (*this.ptr).#field_name = result;
                    }
                    Ok(())
                },
            );
            fields.add_field_method_get(#field_name_str, |lua, this| -> ::mlua::Result<::mlua::Value> {
                let val_ref = unsafe { &mut *this.ptr };
                let lua_val = #krate::lua::LuaSeralize::to_lua(&mut val_ref.#field_name, lua);
                lua_val
            });
        };
        user_data_fields.push(field_tokens);
    }

    let result = quote! {
        pub struct #ref_name {
            ptr: *mut #name,
        }

        impl ::mlua::UserData for #ref_name {
            fn add_fields<F: ::mlua::UserDataFields<Self>>(fields: &mut F) {
                #(#user_data_fields)*
            }
        }

        impl #krate::lua::LuaSeralize for #name {
            fn to_lua(&mut self, lua: &::mlua::Lua) -> ::mlua::Result<::mlua::Value> {
                let val_ref = #ref_name {
                    ptr: self as *mut #name,
                };
                let ud = lua.create_userdata(val_ref)?;
                Ok(::mlua::Value::UserData(ud))
            }
            fn from_lua(value: &::mlua::Value) -> ::mlua::Result<Self> {
                match value {
                    ::mlua::Value::UserData(data) => {
                        let data_ref = data.borrow::<#ref_name>()?;
                        let data = unsafe { (*data_ref.ptr).clone() };
                        Ok(data)
                    },
                    _ => Err(::mlua::Error::RuntimeError(
                        format!("expected userdata for {}, found {}", #name_str, value.type_name())
                    ))
                }
            }
        }
        inventory::submit! {
            #krate::lua::LuaTypeDoc{
                name: #name_str,
                fields: &[#(#lua_types)*]
            }
        }

        inventory::submit! {
            #krate::lua::TypeRegistration {
                name: #name_str,
                typeid: ::std::any::TypeId::of::<#name>(),
                lua_from_ptr: |ptr, lua| -> ::mlua::Result<::mlua::Value> {
                    let val = unsafe { &mut *(ptr as *mut #name) };
                    let val_ref = #ref_name {
                        ptr: val as *mut #name,
                    };
                    let ud = lua.create_userdata(val_ref)?;
                    Ok(::mlua::Value::UserData(ud))
                },
                dyn_from_lua: |value| -> ::mlua::Result<Box<dyn #krate::ecs::DynamicComponent>> {
                    match value {
                        ::mlua::Value::UserData(data) => {
                            let data_ref = data.borrow::<#ref_name>()?;
                            let data = unsafe { (*data_ref.ptr).clone() };
                            Ok(Box::new(data) as Box<dyn #krate::ecs::DynamicComponent>)
                        },
                        _ => Err(::mlua::Error::RuntimeError(
                            format!("expected userdata to convert, found {}", value.type_name())
                        )) }
                },
                default_construct: #default_construct,
            }
        }

        const _: fn() = || {
            fn assert_clone<T: Clone>() {}
            assert_clone::<#name>();
        };
    };
    result.into()
}

#[proc_macro_derive(LuaVal, attributes(lua))]
pub fn derive_lua_value(input: TokenStream) -> TokenStream {
    let krate = if cfg!(feature = "internal") {
        quote!(crate)
    } else {
        quote!(::engine)
    };
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let name_str = name.to_string();

    let fields = match &input.data {
        syn::Data::Struct(data_struct) => &data_struct.fields,
        _ => panic!("LuaSeralize only supports structs"),
    };

    let mut struct_fields = Vec::new();
    let mut table_set_fields = Vec::new();

    let mut lua_types = Vec::new();

    for field in fields {
        let skip = field.attrs.iter().any(|attr| {
            if !attr.path().is_ident("lua") {
                return false;
            }
            attr.parse_args::<syn::Ident>()
                .map(|ident| ident == "skip")
                .unwrap_or(false)
        });
        if skip {
            continue;
        }

        let field_name = &field.ident;
        let field_type = &field.ty;
        let field_name_str = field_name.as_ref().unwrap().to_string();
        let field_type_str = get_field_type_string_name(field_type);

        lua_types.push(quote! {
            (#field_name_str, #field_type_str),
        });

        let set_token = quote! {
            {
                let lua_val = #krate::lua::LuaSeralize::to_lua(&mut val.#field_name, lua)?;
                table.set(#field_name_str, lua_val)?;
            }
        };

        let struct_field_token = quote! {
            #field_name: {
                let lua_val = t.get::<::mlua::Value>(#field_name_str)?;
                <#field_type as #krate::lua::LuaSeralize>::from_lua(&lua_val)?
            },
        };

        table_set_fields.push(set_token);
        struct_fields.push(struct_field_token);
    }

    let no_default = input.attrs.iter().any(|attr| {
        if !attr.path().is_ident("lua") {
            return false;
        }
        attr.parse_args::<syn::Ident>()
            .map(|ident| ident == "no_default")
            .unwrap_or(false)
    });

    let default_construct = if no_default {
        quote!(None)
    } else {
        quote!(Some(|| Box::new(<#name>::default()) as Box<dyn #krate::ecs::DynamicComponent>))
    };

    let result = quote! {
        const _: fn() = || {
            fn assert_clone<T: Clone>() {}
            assert_clone::<#name>();
        };

        inventory::submit! {
            #krate::lua::TypeRegistration {
                name: #name_str,
                typeid: ::std::any::TypeId::of::<#name>(),
                lua_from_ptr: |ptr, lua| -> ::mlua::Result<::mlua::Value> {
                    let val = unsafe { &mut *(ptr as *mut #name) };
                    let lua_val = <#name as #krate::lua::LuaSeralize>::to_lua(val, lua)?;
                    Ok(lua_val)
                },
                dyn_from_lua: |value| -> ::mlua::Result<Box<dyn #krate::ecs::DynamicComponent>> {
                    let val = <#name as #krate::lua::LuaSeralize>::from_lua(value)?;
                    Ok(Box::new(val) as Box<dyn #krate::ecs::DynamicComponent>)
                },
                default_construct: #default_construct,
            }
        }
        impl #krate::lua::LuaSeralize for #name {
            fn to_lua(&mut self, lua: &::mlua::Lua) -> ::mlua::Result<::mlua::Value> {
                let val = unsafe { &mut *(self as *mut #name) };
                let table = lua.create_table()?;
                #(#table_set_fields)*
                Ok(::mlua::Value::Table(table))
            }
            fn from_lua(value: &::mlua::Value) -> ::mlua::Result<Self> {
                match value {
                    ::mlua::Value::Table(t) => {
                        Ok(#name {
                            #(#struct_fields)*
                        })
                    },
                    _ => Err(::mlua::Error::RuntimeError(
                        format!("expected table for {}, found {}", #name_str, value.type_name())
                    ))
                }
            }
        }

        inventory::submit! {
            #krate::lua::LuaTypeDoc {
                name: #name_str,
                fields: &[#(#lua_types)*],
            }
        }
    };
    result.into()
}

#[proc_macro_derive(LuaUnion, attributes(lua))]
pub fn derive_lua_union(input: TokenStream) -> TokenStream {
    let krate = if cfg!(feature = "internal") {
        quote!(crate)
    } else {
        quote!(::engine)
    };

    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let name_str = name.to_string();

    let variants = match &input.data {
        syn::Data::Enum(data_enum) => &data_enum.variants,
        _ => panic!("LuaUnion only supports enums"),
    };

    let mut to_lua_arms = Vec::new();
    let mut from_lua_arms = Vec::new();
    let mut variant_names = Vec::new();
    let mut data_types = Vec::new();

    let no_default = input.attrs.iter().any(|attr| {
        if !attr.path().is_ident("lua") {
            return false;
        }
        attr.parse_args::<syn::Ident>()
            .map(|ident| ident == "no_default")
            .unwrap_or(false)
    });

    let default_construct = if no_default {
        quote!(None)
    } else {
        quote!(Some(|| Box::new(<#name>::default()) as Box<dyn #krate::ecs::DynamicComponent>))
    };

    for variant in variants {
        let v_name = &variant.ident;
        let v_str = v_name.to_string();

        let ty = match &variant.fields {
            syn::Fields::Unnamed(fields) if fields.unnamed.len() == 1 => {
                &fields.unnamed.first().unwrap().ty
            }
            _ => panic!("LuaUnion only supports tuple variants with exactly one field"),
        };

        variant_names.push(v_str.clone());
        data_types.push(quote!(#ty));

        to_lua_arms.push(quote! {
            #name::#v_name(inner) => {
                table.set("type".to_string(), #v_str)?;
                let lua_val = #krate::lua::LuaSeralize::to_lua(inner, lua)?;
                table.set("data".to_string(), lua_val)?;
            }
        });

        from_lua_arms.push(quote! {
            #v_str => Ok(#name::#v_name(<#ty as #krate::lua::LuaSeralize>::from_lua(&data)?)),
        });
    }

    // Build "Aabb | Obb" string for docs
    let data_union_str = variant_names
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(" | ");

    let result = quote! {
        impl #krate::lua::LuaSeralize for #name {
            fn to_lua(&mut self, lua: &::mlua::Lua) -> ::mlua::Result<::mlua::Value> {
                let table = lua.create_table()?;

                match self {
                    #(#to_lua_arms)*
                }

                Ok(::mlua::Value::Table(table))
            }

            fn from_lua(value: &::mlua::Value) -> ::mlua::Result<Self> {
                let table = value.as_table().ok_or_else(|| {
                    ::mlua::Error::RuntimeError(format!(
                        "expected table for {}, found {}",
                        #name_str,
                        value.type_name()
                    ))
                })?;

                let ty: String = table.get("type".to_string())?;
                let data = table.get::<::mlua::Value>("data".to_string())?;

                match ty.as_str() {
                    #(#from_lua_arms)*
                    _ => Err(::mlua::Error::RuntimeError(format!(
                        "unknown variant {} for {}",
                        ty,
                        #name_str
                    ))),
                }
            }
        }

        inventory::submit! {
            #krate::lua::TypeRegistration {
                name: #name_str,
                typeid: ::std::any::TypeId::of::<#name>(),
                lua_from_ptr: |ptr, lua| -> ::mlua::Result<::mlua::Value> {
                    let val = unsafe { &mut *(ptr as *mut #name) };
                    let lua_val = <#name as #krate::lua::LuaSeralize>::to_lua(val, lua)?;
                    Ok(lua_val)
                },
                dyn_from_lua: |value| -> ::mlua::Result<Box<dyn #krate::ecs::DynamicComponent>> {
                    let val = <#name as #krate::lua::LuaSeralize>::from_lua(value)?;
                    Ok(Box::new(val) as Box<dyn #krate::ecs::DynamicComponent>)
                },
                default_construct: #default_construct,
            }
        }
        inventory::submit! {
            #krate::lua::LuaTypeDoc {
                name: #name_str,
                fields: &[
                    ("type", "string"),
                    ("data", #data_union_str),
                ],
            }
        }


        const _: fn() = || {
            fn assert_clone<T: Clone>() {}
            assert_clone::<#name>();
        };
    };

    result.into()
}

fn get_field_type_string_name(field_type: &syn::Type) -> String {
    let field_type_str = match field_type {
        syn::Type::Path(type_path) => {
            let last = type_path.path.segments.last().unwrap();
            let name = last.ident.to_string();
            match &last.arguments {
                syn::PathArguments::AngleBracketed(args) => {
                    if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                        let inner_str = match inner {
                            syn::Type::Path(p) => p
                                .path
                                .segments
                                .last()
                                .map(|s| s.ident.to_string())
                                .unwrap_or_else(|| quote!(#inner).to_string()),
                            _ => quote!(#inner).to_string(),
                        };
                        match name.as_str() {
                            "Vec" => format!("{}[]", inner_str),
                            "Option" => format!("{}?", inner_str),
                            _ => name,
                        }
                    } else {
                        name
                    }
                }
                _ => name,
            }
        }
        _ => quote!(#field_type).to_string(),
    };
    field_type_str
}
