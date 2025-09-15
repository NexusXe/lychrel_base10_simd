#![feature(allocator_api)]

use lychrel_base10_simd::integer_limb::*;
use std::{alloc::Global, path::Path};

fn main() {
    // get file path from args
    let mut args: Vec<String> = std::env::args().collect();

    let mut help: bool = false;
    let mut verify: bool = false;
    let mut display_usage: bool = false;
    let mut unrecognized_args: Vec<&String> = Vec::new();

    if args.len() < 2 {
        display_usage = true;
    }

    let file_path = args.pop();

    if file_path.is_none() {
        display_usage = true;
    }

    for arg in args.iter().skip(1) {
        match arg.as_str() {
            "-h" | "--help" => help = true,
            "-v" | "--verify" => verify = true,
            _ => {
                unrecognized_args.push(arg);
                display_usage = true;
            }
        }
    }

    if display_usage {
        let executed_path = &std::env::current_exe()
            .unwrap_or("read_packed".into())
            .file_name()
            .unwrap_or(std::ffi::OsStr::new("read_packed"))
            .display()
            .to_string();
        if !unrecognized_args.is_empty() {
            eprintln!(
                "\x1b[1;31merror\x1b[0m: unrecognized arguments: {:?}",
                unrecognized_args
            );
        }
        eprintln!("Usage: {} [options] <file_path>", executed_path);
        return;
    }

    if help {
        println!("Usage: {} [options] <file_path>", args[0]);
        println!(
            "Options:
        -h|--help: Display this help
        -v|--verify: Verifies that the read Integer is valid"
        );
        std::process::exit(0);
    }

    let file_path = file_path.unwrap();
    let file_path = Path::new(&file_path);
    // try to read from the file
    let file = match std::fs::read(file_path) {
        Ok(file) => file,
        Err(error) => {
            eprintln!("Unable to read file: {error}");
            return;
        }
    };

    let data: Vec<[LimbVecScalar; LV_LEN]> = match Integer::<Global>::chop(file) {
        None => {
            eprintln!("\x1b[1;31merror\x1b[0m: file length is not a multiple of 64 bytes");
            std::process::exit(1);
        }
        Some(data) => data,
    };

    let global_allocator = Global;

    let integer = Integer::from_bytes(data, global_allocator).unpack(global_allocator);

    if verify && integer.has_carries() {
        eprintln!("\x1b[1;31merror\x1b[0m: unpacked integer has carries");
        std::process::exit(0xA);
    }

    struct IntegerRawDisplay(Integer<Global>);

    impl std::fmt::Display for IntegerRawDisplay {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.display_raw(f)
        }
    }

    println!("{}", IntegerRawDisplay(integer));
}
