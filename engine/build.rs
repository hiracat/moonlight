use core::panic;
use std::{env, ffi::OsStr, fs::create_dir, path::Path, process::Command};
use walkdir::WalkDir;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let shader_dir = Path::new("src/shaders");
    println!("cargo:rerun-if-changed=src/shaders/");

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

        let stage = shader_stage_from_filename(path);
        println!("Compiling {:?} as {:?}", path, stage);

        let output_dir = Path::new(&out_dir).join("shaders");
        create_dir(&output_dir).unwrap_or(());
        let filename = path.file_stem().unwrap().to_string_lossy();
        let output_path = output_dir.join(format!("{filename}.spv"));

        let reflect_path = output_dir.join(format!("{filename}.reflect.spv"));

        for (extra_flags, out) in [(&["-gVS"][..], &output_path), (&[][..], &reflect_path)] {
            let result = Command::new("glslangValidator")
                .args(extra_flags)
                .args([
                    "-V",
                    "--target-env",
                    "vulkan1.3",
                    "-S",
                    stage,
                    "-o",
                    out.to_str().unwrap(),
                    path.to_str().unwrap(),
                ])
                .output()
                .expect("failed to run glslangValidator — is it installed?");

            let stdout = String::from_utf8_lossy(&result.stdout);
            let stderr = String::from_utf8_lossy(&result.stderr);

            if !result.status.success() {
                eprintln!("{stdout}");
                eprintln!("{stderr}");
                panic!("glslangValidator failed for {}", path.display());
            }

            for line in stdout.lines() {
                if line.contains("WARNING") {
                    println!("cargo:warning={line}");
                }
            }
        }
    }
}

fn shader_stage_from_filename(path: &Path) -> &'static str {
    let name = path.file_name().unwrap().to_string_lossy();
    let mut stage = None;

    if name.contains(".vert") {
        stage = match stage {
            None => Some("vert"),
            Some(_) => panic!("Multiple shader kinds detected in file: {}", name),
        };
    }
    if name.contains(".frag") {
        stage = match stage {
            None => Some("frag"),
            Some(_) => panic!("Multiple shader kinds detected in file: {}", name),
        };
    }
    if name.contains(".comp") {
        stage = match stage {
            None => Some("comp"),
            Some(_) => panic!("Multiple shader kinds detected in file: {}", name),
        };
    }

    stage.unwrap_or_else(|| panic!("Unknown shader type for file: {}", name))
}
