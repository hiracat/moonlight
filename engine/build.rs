use std::{env, ffi::OsStr, fs, path::Path, process::Command};

use walkdir::WalkDir;

const ENTRY_POINTS: &[(&str, &str, &str)] = &[
    ("[shader(\"vertex\")]", "vertexMain", "vertex"),
    ("[shader(\"fragment\")]", "fragmentMain", "fragment"),
    ("[shader(\"compute\")]", "computeMain", "compute"),
];

fn main() {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let shader_out_dir = Path::new(&out_dir).join("shaders");
    fs::create_dir_all(&shader_out_dir).expect("failed to create shader output dir");

    let shader_dir = Path::new("src/shaders");
    println!("cargo:rerun-if-changed=src/shaders/");

    for entry in WalkDir::new(shader_dir) {
        let entry = entry.expect("failed to walk src/shaders");
        let path = entry.path();

        if !path.is_file() || path.extension().unwrap_or(OsStr::new("")) != "slang" {
            continue;
        }
        println!("cargo:rerun-if-changed={}", path.display());

        compile_shader(path, &shader_out_dir);
    }
}

fn compile_shader(path: &Path, shader_out_dir: &Path) {
    let source = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));

    let file_stem = path.file_stem().unwrap().to_string_lossy();
    let output_path = shader_out_dir.join(format!("{file_stem}.spv"));
    // TODO: write reflection JSON into OUT_DIR too once i understand the format
    let reflect_output_path = format!("src/shaders/{file_stem}_reflect.json");

    let mut args: Vec<String> = vec![
        path.to_str().unwrap().to_owned(),
        "-o".into(),
        output_path.to_str().unwrap().into(),
        "-reflection-json".into(),
        reflect_output_path,
    ];

    let mut found_entry = false;
    for (marker, entry_name, _stage) in ENTRY_POINTS {
        if source.contains(marker) {
            found_entry = true;
            args.push("-entry".into());
            args.push((*entry_name).into());
        }
    }
    if !found_entry {
        return;
    }

    let status = Command::new("slangc")
        .args(&args)
        .status()
        .expect("failed to run slangc (is it on PATH?)");

    if !status.success() {
        panic!(
            "slangc failed with code {status} for {}\nargs: {args:?}",
            path.display()
        );
    }
}
