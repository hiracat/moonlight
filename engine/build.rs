use std::{
    collections::HashMap,
    env,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use walkdir::WalkDir;

const ENTRY_POINTS: &[(&str, &str, &str)] = &[
    ("[shader(\"vertex\")]", "vertexMain", "vertex"),
    ("[shader(\"fragment\")]", "fragmentMain", "fragment"),
    ("[shader(\"compute\")]", "computeMain", "compute"),
];

fn main() {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let shader_out_dir = PathBuf::from(out_dir).join("shaders");
    clean_shader_out_dir(&shader_out_dir);
    fs::create_dir_all(&shader_out_dir).expect("failed to create shader output dir");

    let shader_dir = PathBuf::from("src/shaders");
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
    rust_type_gen(Path::new(&shader_out_dir));
}
// use this instead of just nuking the directory to prevent accidental nuking of the home directory
fn clean_shader_out_dir(shader_out_dir: &Path) {
    eprintln!("cleaning out directory {}", shader_out_dir.display());
    if !shader_out_dir.exists() {
        eprintln!("directory {} does not exist", shader_out_dir.display());
        return;
    }
    for entry in fs::read_dir(shader_out_dir).expect("failed to read shader output dir") {
        let entry = entry.expect("failed to read dir entry");
        dbg!(&entry);
        let path = entry.path();
        if path.is_file()
            && matches!(
                path.extension().and_then(|e| e.to_str()),
                Some("spv") | Some("json")
            )
        {
            fs::remove_file(&path).expect("failed to remove stale shader artifact");
        }
    }
    let _ = fs::remove_file("tonemap_reflect.json");
    let _ = fs::remove_file("egui_reflect.json");
}

fn compile_shader(path: &Path, shader_out_dir: &Path) {
    let source = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));

    let file_stem = path.file_stem().unwrap().to_string_lossy();
    let output_path = shader_out_dir.join(format!("{file_stem}.spv"));
    // TODO: write reflection JSON into OUT_DIR too once i understand the format
    // let reflect_output_path = shader_out_dir.join(format!("{file_stem}_reflect.json"));
    let reflect_output_path = PathBuf::from(format!("{file_stem}_reflect.json"));
    assert!(
        !output_path.exists(),
        "path already exists: {:?}",
        output_path
    );
    assert!(
        !reflect_output_path.exists(),
        "path already exists: {:?}",
        reflect_output_path
    );

    let mut args: Vec<String> = vec![
        path.to_str().unwrap().to_owned(),
        "-o".into(),
        output_path.to_str().unwrap().into(),
        "-reflection-json".into(),
        reflect_output_path.to_str().unwrap().into(),
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

fn rust_type_gen(out_dir: &Path) {
    let files: Vec<_> = fs::read_dir(out_dir)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .collect();

    let reflect_json_files = files
        .iter()
        .filter(|x| x.to_string_lossy().contains("reflect.json"));
}

struct ShaderUnit {
    entry_points: HashMap<String, EntryPoint>,
}
struct EntryPoint;
