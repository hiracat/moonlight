use core::panic;
use std::{
    env,
    ffi::OsStr,
    fs::{self, create_dir, File},
    io::Read,
    path::Path,
};
use walkdir::WalkDir;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let shader_dir = Path::new("src/shaders");
    println!("cargo:rerun-if-changed=src/shaders/");

    let compiler = shaderc::Compiler::new().expect("Failed to create shader compiler");
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_generate_debug_info();
    // options.add_macro_definition("VK_KHR_shader_non_semantic_info", Some("1"));

    for entry in WalkDir::new(shader_dir) {
        let entry = entry.unwrap();
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().unwrap_or(OsStr::new("")) != "glsl" {
            continue;
        }
        println!("cargo:rerun-if-changed={}", path.display());

        let mut file = File::open(path).expect("failed to open file");
        let mut source = String::new();
        file.read_to_string(&mut source)
            .expect(&format! {"invalid file {}", path.to_string_lossy()});

        let shader_kind = shader_kind_from_filename(path);

        let result = match compiler.compile_into_spirv(
            &source,
            shader_kind,
            &path.to_string_lossy(),
            "main",
            Some(&options),
        ) {
            Ok(result) => result,
            Err(err) => {
                eprintln!("{}", err);
                panic!()
            }
        };
        if result.get_num_warnings() > 0 {
            println!("cargo:warning={}", result.get_warning_messages().trim());
        }

        let filename = path.file_stem().unwrap().to_string_lossy();
        let output_dir = Path::new(&out_dir).join("shaders");
        create_dir(&output_dir).unwrap_or(());
        let file_path = output_dir.join(format!("{filename}.spv"));

        fs::write(&file_path, result.as_binary_u8()).expect("Failed to write SPIR-V");
    }
}

fn shader_kind_from_filename(path: &Path) -> shaderc::ShaderKind {
    let name = path.file_name().unwrap().to_string_lossy();
    if name.contains("vert") {
        shaderc::ShaderKind::Vertex
    } else if name.contains("frag") {
        shaderc::ShaderKind::Fragment
    } else if name.contains("comp") {
        shaderc::ShaderKind::Compute
    } else {
        panic!("Unknown shader type for file: {}", name);
    }
}
