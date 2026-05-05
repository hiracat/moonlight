use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{DeriveInput, parse_macro_input};

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

    let fields = match &input.data {
        syn::Data::Struct(data_struct) => &data_struct.fields,
        _ => panic!("LuaRef only supports structs, use LuaUnion"),
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

    let owned_name = quote::format_ident!("{}Owned", name);

    let mut user_data_fields = Vec::new();
    let mut owned_data_fields = Vec::new();
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

        let field_name = field.ident.as_ref().unwrap();
        let field_type = &field.ty;
        let field_name_str = field_name.to_string();
        let field_type_str = get_field_type_string_name(field_type);

        lua_types.push(quote! { (#field_name_str, #field_type_str), });

        user_data_fields.push(quote! {
            fields.add_field_method_get(#field_name_str, |lua, this| {
                let world = unsafe { &mut *this.world };
                let field_val = match this.source {
                    #krate::lua::ComponentSource::Entity(entity) => world
                        .get::<(#name,)>(entity)
                        .map_err(|e| ::mlua::Error::RuntimeError(
                                format!("entity missing component {}: {:?}", #name_str, e)
                        ))?
                        .#field_name
                        .clone(),
                    #krate::lua::ComponentSource::Resource => world
                        .get_resource::<#name>()
                        .ok_or_else(|| ::mlua::Error::RuntimeError(
                                format!("resource {} missing", #name_str)
                        ))?
                        .#field_name
                        .clone(),
                };
                <#field_type as #krate::lua::LuaSeralize>::to_lua(&field_val, lua, world)
            });

            fields.add_field_method_set(#field_name_str, |_lua, this, val: ::mlua::Value| {
                let world = unsafe { &mut *this.world };
                let result = <#field_type as #krate::lua::LuaSeralize>::from_lua(&val, world)?;
                let comp = match this.source {
                    #krate::lua::ComponentSource::Entity(entity) => world
                        .get_mut::<(#name,)>(entity)
                        .map_err(|e| ::mlua::Error::RuntimeError(
                                format!("entity missing component {}: {:?}", #name_str, e)
                        ))?,
                    #krate::lua::ComponentSource::Resource => world
                        .get_mut_resource::<#name>()
                        .ok_or_else(|| ::mlua::Error::RuntimeError(
                                format!("resource {} missing", #name_str)
                        ))?
                };
                comp.#field_name = result;
                Ok(())
            });
        });

        owned_data_fields.push(quote! {
            fields.add_field_method_get(#field_name_str, |lua, this| {
                let world = unsafe { &mut *this.world };
                <#field_type as #krate::lua::LuaSeralize>::to_lua(&this.value.#field_name, lua, world)
            });

            fields.add_field_method_set(#field_name_str, |_lua, this, val: ::mlua::Value| {
                let world = unsafe { &mut *this.world };
                let result = <#field_type as #krate::lua::LuaSeralize>::from_lua(&val, world)?;
                this.value.#field_name = result;
                Ok(())
            });
        });
    }

    quote! {
        pub struct #ref_name {
            world: *mut #krate::ecs::World,
            source: #krate::lua::ComponentSource,
        }
        unsafe impl Send for #ref_name {}

        pub struct #owned_name {
            world: *mut #krate::ecs::World,
            value: #name,
        }
        unsafe impl Send for #owned_name {}

        impl ::mlua::UserData for #ref_name {
            fn add_fields<F: ::mlua::UserDataFields<Self>>(fields: &mut F) {
                #(#user_data_fields)*
            }
        }

        impl ::mlua::UserData for #owned_name {
            fn add_fields<F: ::mlua::UserDataFields<Self>>(fields: &mut F) {
                #(#owned_data_fields)*
            }
        }

        impl #krate::lua::LuaSeralize for #name {
            fn from_lua(value: &::mlua::Value, world: &mut #krate::ecs::World) -> ::mlua::Result<Self> {
                match value {
                    ::mlua::Value::UserData(data) => {
                        if let Ok(data_ref) = data.borrow::<#ref_name>() {
                            Ok(match data_ref.source {
                                #krate::lua::ComponentSource::Entity(entity) => {
                                    world.get::<(#name,)>(entity)
                                        .map(|x| x.clone())
                                        .map_err(|e| ::mlua::Error::RuntimeError(
                                                format!("entity missing component {}: {:?}", #name_str, e)
                                        ))?
                                },
                                #krate::lua::ComponentSource::Resource => {
                                    world.get_resource::<#name>()
                                        .cloned()
                                        .ok_or_else(|| ::mlua::Error::RuntimeError(
                                            format!("resource {} missing", #name_str)
                                        ))?
                                },
                            })
                        } else if let Ok(owned) = data.borrow::<#owned_name>() {
                            Ok(owned.value.clone())
                        } else {
                            Err(::mlua::Error::RuntimeError(
                                format!("expected {} userdata, found {}", #name_str, value.type_name())
                            ))
                        }
                    },
                    _ => Err(::mlua::Error::RuntimeError(
                        format!("expected {} userdata, found {}", #name_str, value.type_name())
                    )),
                }
            }

            fn to_lua(&self, lua: &::mlua::Lua, world: &mut #krate::ecs::World) -> ::mlua::Result<::mlua::Value> {
                let ud = lua.create_userdata(#owned_name {
                    world: world as *mut #krate::ecs::World,
                    value: self.clone(),
                })?;
                Ok(::mlua::Value::UserData(ud))
            }
        }

        inventory::submit! {
            #krate::lua::LuaTypeDoc {
                name: #name_str,
                fields: &[#(#lua_types)*],
            }
        }

        inventory::submit! {
            #krate::lua::TypeRegistration {
                name: #name_str,
                typeid: ::std::any::TypeId::of::<#name>(),
                lua_from_entity: |entity, world, lua| {
                    let ud = lua.create_userdata(#ref_name {
                        world,
                        source: #krate::lua::ComponentSource::Entity(entity),
                    })?;
                    Ok(::mlua::Value::UserData(ud))
                },
                lua_from_resource: |world, lua| {
                    let ud = lua.create_userdata(#ref_name {
                        world,
                        source: #krate::lua::ComponentSource::Resource,
                    })?;
                    Ok(::mlua::Value::UserData(ud))
                },
                dyn_from_lua: |value, world| {
                    let val = <#name as #krate::lua::LuaSeralize>::from_lua(value, world)?;
                    Ok(Box::new(val) as Box<dyn #krate::ecs::DynamicComponent>)
                },
                default_construct: #default_construct,
            }
        }

        const _: fn() = || {
            fn assert_clone<T: Clone>() {}
            assert_clone::<#name>();
        };
    }.into()
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
        _ => panic!("LuaVal only supports structs"),
    };

    let mut struct_fields = Vec::new();
    let mut table_set_fields = Vec::new();
    let mut lua_types = Vec::new();

    for field in fields {
        let field_name = field.ident.as_ref().unwrap();
        let field_type = &field.ty;
        let field_name_str = field_name.to_string();
        let field_type_str = get_field_type_string_name(field_type);

        lua_types.push(quote! { (#field_name_str, #field_type_str), });

        table_set_fields.push(quote! {
            {
                let lua_val = <#field_type as #krate::lua::LuaSeralize>::to_lua(
                    &self.#field_name, lua, world
                )?;
                table.set(#field_name_str, lua_val)?;
            }
        });

        struct_fields.push(quote! {
            #field_name: {
                let lua_val = t.get::<::mlua::Value>(#field_name_str)?;
                <#field_type as #krate::lua::LuaSeralize>::from_lua(&lua_val, world)?
            },
        });
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

    quote! {
        const _: fn() = || {
            fn assert_clone<T: Clone>() {}
            assert_clone::<#name>();
        };

        impl #krate::lua::LuaSeralize for #name {
            fn from_lua(value: &::mlua::Value, world: &mut #krate::ecs::World) -> ::mlua::Result<Self> {
                match value {
                    ::mlua::Value::Table(t) => Ok(#name {
                        #(#struct_fields)*
                    }),
                    _ => Err(::mlua::Error::RuntimeError(
                        format!("expected table for {}, found {}", #name_str, value.type_name())
                    )),
                }
            }

            fn to_lua(&self, lua: &::mlua::Lua, world: &mut #krate::ecs::World) -> ::mlua::Result<::mlua::Value> {
                let table = lua.create_table()?;
                #(#table_set_fields)*
                Ok(::mlua::Value::Table(table))
            }
        }

        inventory::submit! {
            #krate::lua::TypeRegistration {
                name: #name_str,
                typeid: ::std::any::TypeId::of::<#name>(),
                lua_from_entity: |entity, world, lua| {
                    let world_ref = unsafe { &mut *world };
                    let comp = world_ref
                        .get::<(#name,)>(entity)
                        .map_err(|e| ::mlua::Error::RuntimeError(
                                format!("entity missing component {}: {:?}", #name_str, e)
                        ))?
                        .clone();
                    <#name as #krate::lua::LuaSeralize>::to_lua(&comp, lua, world_ref)
                },
                lua_from_resource: |world, lua| {
                    let world_ref = unsafe { &mut *world };
                    let res = world_ref
                        .get_resource::<#name>()
                        .ok_or_else(|| ::mlua::Error::RuntimeError(
                                format!("resource {} missing", #name_str)
                        ))?
                        .clone();
                    <#name as #krate::lua::LuaSeralize>::to_lua(&res, lua, world_ref)
                },
                dyn_from_lua: |value, world| {
                    let val = <#name as #krate::lua::LuaSeralize>::from_lua(value, world)?;
                    Ok(Box::new(val) as Box<dyn #krate::ecs::DynamicComponent>)
                },
                default_construct: #default_construct,
            }
        }

        inventory::submit! {
            #krate::lua::LuaTypeDoc {
                name: #name_str,
                fields: &[#(#lua_types)*],
            }
        }
    }.into()
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

        let fields = match &variant.fields {
            syn::Fields::Unnamed(fields) => Some(&fields.unnamed),
            syn::Fields::Unit => None,
            _ => panic!("LuaUnion only supports tuple + unit variants"),
        };

        if let Some(fields) = fields {
            let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();
            variant_names.push(v_str.clone());

            if field_types.len() == 1 {
                let ty = &field_types[0];
                data_types.push(quote!(#ty).to_string());
            } else {
                let joined = field_types
                    .iter()
                    .map(|ty| quote!(#ty).to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                data_types.push(joined);
            }

            let bindings: Vec<syn::Ident> = (0..field_types.len())
                .map(|i| format_ident!("field_{}", i))
                .collect();

            let to_lua_body = if field_types.len() == 1 {
                let binding = &bindings[0];
                quote! {
                    let lua_val = <_ as #krate::lua::LuaSeralize>::to_lua(#binding, lua, world)?;
                    table.set("data".to_string(), lua_val)?;
                }
            } else {
                let sets = bindings.iter().enumerate().map(|(i, binding)| {
                    let idx = i + 1;
                    quote! {
                        arr.set(#idx, <_ as #krate::lua::LuaSeralize>::to_lua(#binding, lua, world)?)?;
                    }
                });
                quote! {
                    let arr = lua.create_table()?;
                    #(#sets)*
                    table.set("data".to_string(), arr)?;
                }
            };

            to_lua_arms.push(quote! {
                #name::#v_name(#(#bindings),*) => {
                    table.set("type".to_string(), #v_str)?;
                    #to_lua_body
                }
            });

            let from_lua_body = if field_types.len() == 1 {
                let ty = &field_types[0];
                quote! {
                    Ok(#name::#v_name(
                        <#ty as #krate::lua::LuaSeralize>::from_lua(&data, world)?
                    ))
                }
            } else {
                let reads = field_types.iter().enumerate().map(|(i, ty)| {
                    let idx = i + 1;
                    quote! {
                        <#ty as #krate::lua::LuaSeralize>::from_lua(
                            &arr.get::<::mlua::Value>(#idx)?,
                            world,
                        )?
                    }
                });
                quote! {
                    let arr = data.as_table().ok_or_else(|| {
                        ::mlua::Error::RuntimeError(format!(
                            "expected tuple table for variant {}", #v_str
                        ))
                    })?;
                    Ok(#name::#v_name(#(#reads),*))
                }
            };

            from_lua_arms.push(quote! {
                #v_str => { #from_lua_body },
            });
        } else {
            variant_names.push(v_str.clone());
            data_types.push(String::new());

            to_lua_arms.push(quote! {
                #name::#v_name => {
                    table.set("type".to_string(), #v_str)?;
                }
            });

            from_lua_arms.push(quote! {
                #v_str => Ok(#name::#v_name),
            });
        }
    }

    let variant_entries: Vec<_> = variant_names
        .iter()
        .zip(data_types.iter())
        .map(|(name, ty)| quote!((#name, #ty)))
        .collect();

    quote! {
        impl #krate::lua::LuaSeralize for #name {
            fn to_lua(&self, lua: &::mlua::Lua, world: &mut #krate::ecs::World) -> ::mlua::Result<::mlua::Value> {
                let table = lua.create_table()?;
                match self {
                    #(#to_lua_arms)*
                }
                Ok(::mlua::Value::Table(table))
            }

            fn from_lua(value: &::mlua::Value, world: &mut #krate::ecs::World) -> ::mlua::Result<Self> {
                let table = value.as_table().ok_or_else(|| {
                    ::mlua::Error::RuntimeError(format!(
                        "expected table for {}, found {}", #name_str, value.type_name()
                    ))
                })?;
                let ty: String = table.get("type".to_string())?;
                let data = table.get::<::mlua::Value>("data".to_string())?;
                match ty.as_str() {
                    #(#from_lua_arms)*
                    _ => Err(::mlua::Error::RuntimeError(format!(
                        "unknown variant {} for {}", ty, #name_str
                    ))),
                }
            }
        }

        inventory::submit! {
            #krate::lua::TypeRegistration {
                name: #name_str,
                typeid: ::std::any::TypeId::of::<#name>(),
                lua_from_entity: |entity, world, lua| {
                    let world_ref = unsafe { &mut *world };
                    let comp = world_ref
                        .get::<(#name,)>(entity)
                        .map_err(|e| ::mlua::Error::RuntimeError(
                                format!("entity missing component {}: {:?}", #name_str, e)
                        ))?
                        .clone();
                    <#name as #krate::lua::LuaSeralize>::to_lua(&comp, lua, world_ref)
                },
                lua_from_resource: |world, lua| {
                    let world_ref = unsafe { &mut *world };
                    let res = world_ref
                        .get_resource::<#name>()
                        .ok_or_else(|| ::mlua::Error::RuntimeError(
                                format!("resource {} missing", #name_str)
                        ))?
                        .clone();
                    <#name as #krate::lua::LuaSeralize>::to_lua(&res, lua, world_ref)
                },
                dyn_from_lua: |value, world| {
                    let val = <#name as #krate::lua::LuaSeralize>::from_lua(value, world)?;
                    Ok(Box::new(val) as Box<dyn #krate::ecs::DynamicComponent>)
                },
                default_construct: #default_construct,
            }
        }

        inventory::submit! {
            #krate::lua::LuaUnionDoc {
                name: #name_str,
                variants: &[#(#variant_entries),*],
            }
        }

        const _: fn() = || {
            fn assert_clone<T: Clone>() {}
            assert_clone::<#name>();
        };
    }.into()
}

fn generic_arg_to_type_str(arg: &syn::GenericArgument) -> String {
    match arg {
        syn::GenericArgument::Type(ty) => get_field_type_string_name(ty),
        other => panic!("unsupported type arg {}", other.to_token_stream()),
    }
}

fn get_field_type_string_name(field_type: &syn::Type) -> String {
    match field_type {
        syn::Type::Path(type_path) => {
            let last = type_path.path.segments.last().unwrap();
            let name = last.ident.to_string();
            match &last.arguments {
                syn::PathArguments::AngleBracketed(args) => {
                    let mut iter = args.args.iter();
                    match name.as_str() {
                        "Vec" => format!("{}[]", generic_arg_to_type_str(iter.next().unwrap())),
                        "Option" => format!("{}?", generic_arg_to_type_str(iter.next().unwrap())),
                        "Box" => generic_arg_to_type_str(iter.next().unwrap()),
                        "HashMap" => {
                            let key_str =
                                rust_type_to_lua(&generic_arg_to_type_str(iter.next().unwrap()));
                            let val_str =
                                rust_type_to_lua(&generic_arg_to_type_str(iter.next().unwrap()));
                            format!("table<{}, {}>", key_str, val_str)
                        }
                        _ => name,
                    }
                }
                _ => name,
            }
        }
        other => panic!("unsupported type arg {}", other.to_token_stream()),
    }
}

//WARNING: this is duplicated from lua.rs, updates should also go there
fn rust_type_to_lua(rust_type: &str) -> String {
    let split_idx = rust_type
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .unwrap_or(rust_type.len());

    let (base_type, suffix) = rust_type.split_at(split_idx);

    let lua_type = match base_type {
        "f32" | "f64" | "i32" | "i64" | "u32" | "u64" | "isize" | "usize" => "number",
        "String" => "string",
        "bool" => "boolean",
        other => other,
    };

    format!("{lua_type}{suffix}")
}
