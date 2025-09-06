#![feature(portable_simd)]
#![feature(const_from)]
#![feature(const_trait_impl)]
#![feature(likely_unlikely)]
#![feature(cold_path)]
#![feature(slice_from_ptr_range)]
#![feature(int_roundings)]
#![feature(const_cmp)]
#![feature(const_eval_select)]
#![feature(core_intrinsics)]
#![allow(internal_features)]
#![feature(iter_collect_into)]
#![deny(clippy::all)]

mod integer_limb;
use integer_limb::{Checkpoint, Integer, Limb};
use std::hint::{cold_path, likely, unlikely};
use std::path::{Path, PathBuf};
use std::simd::prelude::*;
use std::sync::mpsc;
use std::sync::mpsc::Sender;
use std::thread;
use std::time::Instant;

use windows::Win32::System::Memory::{
    GetLargePageMinimum, MEM_COMMIT, MEM_LARGE_PAGES, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc2,
};
use windows::Win32::System::Threading::GetCurrentProcess;

#[cfg(not(feature = "no-verify"))]
use std::io::Read;

pub struct IterationResult {
    last_iteration: usize,
    start_time: Instant,
    end_integer: Integer,
}

pub struct StatusReport {
    iteration: usize,
    current_value: Option<Integer>,
}

const CHECKPOINT_DIR: &str = "./checkpoints";

const INITIAL_SEED: &str = "196";
const LOG_FREQUENCY_EXP: usize = 14;
const LOG_MASK: usize = 2usize.pow(LOG_FREQUENCY_EXP as u32);

/// Iterates over a given input. If the returned `usize` is less than `range.end`, a palindrome was found.
fn iterate(
    range: std::ops::Range<usize>,
    starting_integer: Integer,
    tx: Option<Sender<StatusReport>>,
) -> IterationResult {
    let mut current_iteration: Integer = starting_integer;

    let mut carried: bool = false;
    let mut i: usize = range.start;

    let start_time = Instant::now();

    while likely(i < range.end) {
        if unlikely(!carried) {
            cold_path();
            let mut reverse: Integer = Integer(Vec::with_capacity(current_iteration.0.len()));
            current_iteration.reverse_into_integer(&mut reverse);
            if current_iteration.0 == reverse.0 {
                cold_path();
                break;
            }
        }
        carried = current_iteration.fused_reverse_add_asm_interleave();
        if unlikely(i.is_multiple_of(LOG_MASK)) {
            let report = StatusReport {
                iteration: i,
                current_value: {
                    if unlikely(i.is_multiple_of(2usize.pow(18))) {
                        cold_path();
                        Some(current_iteration.clone())
                    } else {
                        None
                    }
                },
            };

            if likely(tx.is_some()) {
                if unlikely(
                    unsafe { tx.as_ref().unwrap_unchecked() }
                        .send(report)
                        .is_err(),
                ) {
                    //eprintln!("Main thread has disconnected. Stopping.");
                    break;
                }
            } else {
                cold_path();
                //println!("{i}; {rate:} iter/sec");
            }
        }
        i += 1;
    }
    IterationResult {
        last_iteration: i,
        start_time,
        end_integer: current_iteration,
    }
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    const LIMIT_SHORT: usize = 603_567;

    #[cfg(debug_assertions)]
    {
        let large_page_size = unsafe { GetLargePageMinimum() };
        assert!(integer_limb::HUGE_PAGE_SIZE_BYTES.is_multiple_of(large_page_size));
    }

    let process_handle = unsafe { GetCurrentProcess() };

    let large_page_memory = unsafe {
        VirtualAlloc2(
            Some(process_handle),
            None, // Let the OS determine the address
            integer_limb::HUGE_PAGE_SIZE_BYTES,
            MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, // Request large pages
            PAGE_READWRITE.0,
            None
        )
    };

    let _ = std::hint::black_box(large_page_memory);

    //const LIMIT: usize = 500;
    //const LIMIT: usize = 100_358;
    const LIMIT: usize = usize::MAX;

    let compile_datetime = compile_time::datetime_str!();
    let rustc_version = compile_time::rustc_version_str!();

    println!("lychrel_base10_simd compiled with {rustc_version} on {compile_datetime}");

    #[cfg(not(debug_assertions))]
    let _ = affinity::set_thread_affinity([5]);

    let mut initial_value: Integer = integer!(INITIAL_SEED);
    let mut starting_iteration: usize = 1;

    let args: Vec<String> = std::env::args().collect();

    if !args.contains(&"--no-checkpoint".to_string()) {
        let checkpoint_path = Path::new(CHECKPOINT_DIR);

        match std::fs::read_dir(checkpoint_path) {
            Ok(entries) => {
                eprintln!("Using pre-existing checkpoints folder.");
                // since the folder exists, get the `Path`s of all files inside of it
                // filter out those that are irrelevant to our current seed
                let mut checkpoint_files: Vec<std::path::PathBuf> = entries
                    .filter_map(|entry| entry.ok())
                    .map(|entry| entry.path())
                    .filter(|path| {
                        path.file_name()
                            .and_then(|name| name.to_str())
                            .is_some_and(|s| s.ends_with(&format!("{INITIAL_SEED:}_checkpoint")))
                    })
                    .collect();

                checkpoint_files.sort_unstable_by_key(|path| {
                    path.file_name()
                        .and_then(|s| s.to_str())
                        .and_then(|s| s.split('.').next())
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap()
                });

                let checkpoint_path: Option<PathBuf> = if args.contains(&"--start-at".to_string()) {
                    // get the arg after "--start-at"
                    let start_at_index = match args.iter().position(|arg| arg == "--start-at") {
                        Some(index) => index,
                        None => {
                            eprintln!("Please specify a checkpoint index to start at");
                            std::process::exit(1);
                        }
                    };
                    let start_at_value = match args[start_at_index + 1].parse::<usize>() {
                        Ok(value) => value,
                        Err(_) => {
                            eprintln!("Please specify a valid checkpoint index to start at");
                            std::process::exit(1);
                        }
                    };

                    // find the checkpoint that starts with `start_at_value`
                    Some(
                        match checkpoint_files.into_iter().find(|path| {
                            path.file_name()
                                .and_then(|name| name.to_str())
                                .and_then(|s| s.split('.').next())
                                .and_then(|s| s.parse::<usize>().ok())
                                .is_some_and(|i| i == start_at_value)
                        }) {
                            Some(path) => path,
                            None => {
                                eprintln!("No checkpoint found with index {start_at_value:}");
                                std::process::exit(1);
                            }
                        },
                    )
                } else if checkpoint_files.len() >= 2 {
                    // use the second to last checkpoint
                    Some(checkpoint_files[checkpoint_files.len() - 2].clone())
                } else {
                    None
                };

                match checkpoint_path {
                    Some(used_checkpoint_path) => {
                        let used_checkpoint_iteration = used_checkpoint_path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .and_then(|s| s.split('.').next())
                            .map(|s| s.parse::<usize>())
                            .unwrap()?;
                        let used_checkpoint = std::fs::read(used_checkpoint_path)?;
                        let checkpoint =
                            Checkpoint::new(used_checkpoint_iteration, used_checkpoint);
                        (initial_value, starting_iteration) = Integer::from_checkpoint(checkpoint);
                        println!("Starting from checkpoint at iteration {starting_iteration:}");
                        starting_iteration += 1;
                    }

                    None => {
                        eprintln!("No valid checkpoints to start from found.")
                    }
                }
            }
            Err(_) => {
                std::fs::create_dir(checkpoint_path)?;
                eprintln!("Created new local checkpoints folder in local directory");
            }
        }
    } else {
        eprintln!("Not starting from checkpoint.");
    }

    let limit: usize = if args.contains(&"--short".to_string()) {
        eprintln!("Performing short run.");
        LIMIT_SHORT * 5
    } else if args.contains(&"--bench".to_string()) {
        eprintln!("Performing benchmark run.");
        LIMIT_SHORT
    } else if args.contains(&"--long-bench".to_string()) {
        eprintln!("Performing long benchmark run.");
        LIMIT_SHORT * 2
    } else {
        LIMIT
    };
    println!("----------------------------------------------------------------");

    let (tx, rx) = mpsc::channel::<StatusReport>();

    let iteration_handle = thread::spawn(move || {
        #[cfg(not(debug_assertions))]
        let _ = affinity::set_thread_affinity([4]);

        iterate(starting_iteration..limit, initial_value, Some(tx))
    });

    let mut step_time = Instant::now();

    for status_report in rx {
        let elapsed_time = step_time.elapsed();
        step_time = Instant::now();

        let i = status_report.iteration;
        let current_value = status_report.current_value;

        let log_idx = i.div_floor(LOG_MASK) % 16;

        let rate: f32 =
            unsafe { std::intrinsics::fdiv_fast(LOG_MASK as f32, elapsed_time.as_secs_f32()) };

        println!(
            "{}:{} {i}; {rate:} iter/sec",
            if log_idx == 0 { 16 } else { log_idx },
            if log_idx < 10 { " " } else { "" }
        );

        if current_value.is_some() {
            cold_path();

            let current_value = unsafe { current_value.unwrap_unchecked() };

            let num_limbs = current_value.0.len();

            #[cfg(not(feature = "no-verify"))]
            {
                let checkpoint_path =
                    Path::new(CHECKPOINT_DIR).join(format!("{i}.{INITIAL_SEED:}_checkpoint"));

                println!(
                    "Reached checkpoint: {}",
                    unsafe { checkpoint_path.file_name().unwrap_unchecked() }.display()
                );
                let checkpoint = current_value.into_checkpoint(i);
                cold_path();

                if checkpoint_path.exists() && checkpoint_path.is_file() {
                    print!("Checkpoint already exists; validating... ");
                    // read the file
                    let mut file = match std::fs::File::open(checkpoint_path) {
                        Ok(file) => file,
                        Err(_) => {
                            cold_path();
                            eprintln!("UNABLE TO OPEN FILE\nContinuing anyway...");

                            continue;
                        }
                    };
                    let mut buffer = Vec::new();
                    match file.read_to_end(&mut buffer) {
                        Ok(_) => {
                            let read_checkpoint = Checkpoint::new(i, buffer);
                            if likely(read_checkpoint == checkpoint) {
                                println!("OK");
                            } else {
                                cold_path();
                                println!("FAILED");

                                eprintln!("Checkpoint validation failed at checkpoint {i:}");
                                std::process::exit(1)
                            }
                        }
                        Err(_) => {
                            cold_path();
                            eprintln!("UNABLE TO READ FILE");
                            eprintln!("Continuing anyway...")
                        }
                    }
                } else {
                    print!("Writing checkpoint to {}... ", checkpoint_path.display());
                    let data = checkpoint.data().1;
                    let data_length = data.len();
                    match std::fs::write(checkpoint_path, data) {
                        Ok(_) => {
                            println!("OK");
                            println!("Wrote {:} KiB", data_length / 1024);
                        }
                        Err(e) => {
                            eprintln!("FAILED: {e}");
                            std::process::exit(1);
                        }
                    }
                }
            }
            println!(
                "{:} limbs, approx. {:} digits, {:} KiB of memory",
                num_limbs,
                num_limbs * 64,
                (num_limbs * 64).div_ceil(1024)
            );
        }
    }

    let result = iteration_handle.join();

    if result.is_err() {
        eprintln!("Worker thread died. Exiting.");
        std::process::exit(1);
    }

    let result = result.unwrap();

    let (last_iteration, start_time, end_integer) =
        (result.last_iteration, result.start_time, result.end_integer);

    let found_palindrome: bool = last_iteration < limit;

    let elapsed_time = start_time.elapsed();

    println!(
        "\nIterating took {:.4} seconds\n
        End result has {:} limbs representing {:} digits, occupying {:} KiB of memory\n
        {} find a palindrome after {:} iterations{}",
        elapsed_time.as_secs_f64(),
        end_integer.0.len(),
        end_integer.len(),
        end_integer.0.len() * 64 / 1024,
        if found_palindrome { "Did" } else { "Did not" },
        last_iteration,
        if found_palindrome {
            &format! {": {:}", end_integer}
        } else {
            "."
        }
    );

    if unlikely(found_palindrome) {
        println!("Writing packed found palindrome to \"FOUND_{last_iteration}.txt\"");
        std::fs::write(
            format!("FOUND_{last_iteration}.txt"),
            end_integer.pack().into_bytes(),
        )
        .unwrap();
    }

    Ok(())
}

#[cfg(test)]
mod tests;
