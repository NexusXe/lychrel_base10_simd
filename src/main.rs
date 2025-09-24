#![feature(likely_unlikely)]
#![feature(cold_path)]
#![feature(int_roundings)]
#![feature(core_intrinsics)]
#![allow(internal_features)]
#![feature(allocator_api)]
#![deny(clippy::all)]

#[cfg(all(
    target_pointer_width = "64",
    not(target_family = "wasm"),
    not(feature = "global-alloc")
))]
use lychrel_base10_simd::integer_limb::HugePageAllocator;

use lychrel_base10_simd::integer_limb::{
    Checkpoint, Integer, LV_BYTES, LV_LEN, Limb, LimbVecScalar, WV_LEN, WideVecScalar,
};
use std::alloc::{Allocator, Global};
use std::any::type_name;
use std::hint::{cold_path, likely, unlikely};
use std::path::Path;
use std::sync::mpsc;
use std::sync::mpsc::Sender;
use std::thread;
use std::time::Instant;

#[cfg(not(feature = "no-verify"))]
use std::io::Read;

pub struct IterationResult<T: Allocator + Clone + Copy> {
    last_iteration: usize,
    start_time: Instant,
    end_integer: Integer<T>,
}

pub struct StatusReport {
    iteration: usize,
    current_value: Option<Integer<Global>>,
}

const LOG_FREQUENCY_EXP: usize = 14;
const LOG_MASK: usize = 2usize.pow(LOG_FREQUENCY_EXP as u32);

/// Iterates over a given input. If the returned `usize` is less than `range.end`, a palindrome was found.
fn iterate<T: std::alloc::Allocator + Clone + Copy>(
    range: std::ops::Range<usize>,
    starting_integer: Integer<T>,
    tx: Option<Sender<StatusReport>>,
) -> IterationResult<T> {
    let mut current_iteration = starting_integer;

    current_iteration.0.reserve(2048.min(range.end / 100));

    let mut carried: bool = true; // ignore palindrome check on the first loop
    let mut i: usize = range.start;

    let start_time = Instant::now();

    while likely(i < range.end) {
        if unlikely(!carried) {
            cold_path();
            eprintln!("Checking...");
            let mut reverse = Integer(Vec::with_capacity(current_iteration.0.len()));
            current_iteration.reverse_into_integer(&mut reverse);
            if current_iteration.0 == reverse.0 {
                cold_path();
                break;
            }
        }
        carried = current_iteration.fused_reverse_add_asm_interleave();
        if unlikely(i.is_multiple_of(LOG_MASK)) {
            //eprintln!("{:} limbs, capacity: {:}", current_iteration.0.len(), current_iteration.0.capacity());
            let report = StatusReport {
                iteration: i,
                current_value: {
                    if unlikely(i.is_multiple_of(2usize.pow(18))) {
                        cold_path();
                        // manually clone the current iteration into a new vector using the global allocator
                        let mut output_vec =
                            Vec::with_capacity_in(current_iteration.0.len(), Global);
                        output_vec.extend_from_slice(&current_iteration.0);
                        Some(Integer(output_vec))
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
    const INITIAL_SEED: u128 = 196;
    const LIMIT_SHORT: usize = 603_567;
    const LIMIT_LONG_BENCH: usize = LIMIT_SHORT * 2;
    const LIMIT_PROFILING: usize = LIMIT_SHORT * 5;

    //const LIMIT: usize = 500;
    //const LIMIT: usize = 100_358;
    const LIMIT: usize = usize::MAX;
    println!("
SIMD Lychrel Number Search
    Compile options:
    Overall SIMD Width: {} bits
    Limb Vector Scalar: {}
    Limb Vector Width:  {} ({} bits total)
    Packed Limb Scalar: {}
    Packed Limb Width:  {}  ({} bits total)\n{}\n{}",
        std::mem::size_of::<Limb>() * 8,
        type_name::<LimbVecScalar>(),
        LV_LEN,
        LV_LEN as u32 * LimbVecScalar::BITS,
        type_name::<WideVecScalar>(),
        WV_LEN,
        WV_LEN as u32 * WideVecScalar::BITS,
        if cfg!(debug_assertions) {
            "NOTICE: Debug assertions are enabled. Expect this to cause a significant slowdown!"
        } else {
            ""
        },
        if cfg!(feature = "1g-pages") {
            "1 GiB huge pages are enabled."
        } else {
            ""
        }
    );

    #[cfg(all(
        target_pointer_width = "64",
        not(target_family = "wasm"),
        not(feature = "global-alloc")
    ))]
    let allocator = HugePageAllocator::init()?;

    #[cfg(any(
        not(target_pointer_width = "64"),
        target_family = "wasm",
        feature = "global-alloc"
    ))]
    let allocator = Global;

    let mut starting_iteration: usize = 1;
    let args: Vec<String> = std::env::args().collect();

    const DEFAULT_CHECKPOINT_DIR: &str = "./checkpoints";

    let checkpoint_dir = match std::env::var("LYCHREL_CHECKPOINTS_PATH") {
        Ok(path) => path.trim_end_matches(['/', '\\']).to_string(),
        Err(_) => DEFAULT_CHECKPOINT_DIR.to_string(),
    };

    enum RunType {
        Bench,
        LongBench,
        Short,
        Long,
    }

    let mut help: bool = false;
    let mut short_help: bool = false;
    let mut version: bool = false;
    let mut skip_next_arg: bool = false;

    let mut start_at: Option<usize> = None;

    let mut stop_at: Option<usize> = None;
    let mut checkpoint_path_str = checkpoint_dir.clone();
    let mut seed_number: u128 = INITIAL_SEED;

    let mut no_checkpoint: bool = false;
    let mut write_yield = false;
    let mut run_type = RunType::Long;

    #[cfg(all(
        target_pointer_width = "64",
        not(target_family = "wasm"),
        not(feature = "global-alloc")
    ))]
    let mut initial_value: Integer<HugePageAllocator> = Integer(Vec::new_in(allocator));

    #[cfg(any(
        not(target_pointer_width = "64"),
        target_family = "wasm",
        feature = "global-alloc"
    ))]
    let mut initial_value: Integer<Global> = Integer(Vec::new());


    for (idx, arg) in args.iter().skip(1).enumerate() {
        if skip_next_arg {
            skip_next_arg = false;
            continue;
        }

        match arg.as_str() {
            "--help" | "-h" => {
                help = true;
                break;
            }
            "--version" => version = true,
            "--seed" => {
                skip_next_arg = true;
                seed_number = match args.get(idx + 1) {
                    Some(seed) => match seed.parse::<u128>() {
                        Ok(seed) => seed,
                        Err(_) => {
                            eprintln!("Please specify a valid seed number");
                            std::process::exit(1);
                        }
                    },
                    None => {
                        eprintln!("Please specify a seed number");
                        std::process::exit(1);
                    }
                };
            }
            "--checkpoint-dir" => {
                skip_next_arg = true;
                checkpoint_path_str = match args.get(idx + 1) {
                    Some(path) => path.to_string(),
                    None => {
                        eprintln!("Please specify a checkpoint directory path");
                        std::process::exit(1);
                    }
                };
            }
            "--start-at" => {
                skip_next_arg = true;
                start_at = match args.get(idx + 1) {
                    Some(start_at) => match start_at.parse::<usize>() {
                        Ok(start_at) => Some(start_at),
                        Err(_) => {
                            eprintln!("Please specify a valid start value");
                            std::process::exit(1);
                        }
                    },
                    None => {
                        eprintln!("Please specify a start value");
                        std::process::exit(1);
                    }
                };
            }
            "--stop-at" => {
                skip_next_arg = true;
                stop_at = match args.get(idx + 1) {
                    Some(stop_at_str) => match stop_at_str.parse::<usize>() {
                        Ok(stop_at_val) => Some(if stop_at_val == 0 {
                            LIMIT
                        } else {
                            stop_at_val + 1
                        }),
                        Err(_) => {
                            eprintln!("Please specify a valid stop value");
                            std::process::exit(1);
                        }
                    },
                    None => {
                        eprintln!("Please specify a stop value");
                        std::process::exit(1);
                    }
                };
            }
            "--no-checkpoint" => {
                no_checkpoint = true;
            }
            "--yield" => {
                write_yield = true;
            }
            "--bench" => {
                run_type = RunType::Bench;
            }
            "--short" => {
                run_type = RunType::Short;
            }
            "--long-bench" => {
                run_type = RunType::LongBench;
            }
            "--long" => {
                run_type = RunType::Long;
            }

            _ => {
                eprintln!("unrecognized argument: {arg}");
                short_help = true;
            }
        }
    }

    if help {
        println!("lychrel_base10_simd usage: lychrel_base10_simd [options]
General options:
--help              show this help
--version           show version
--seed <seed>       specify the base number to iterate over (default: {INITIAL_SEED:})
--checkpoint-dir    specify the directory of checkpoint files (default: {DEFAULT_CHECKPOINT_DIR:})
--start-at          specify starting checkpoint iteration number (default: second to last if available)
--stop-at           specify iteration target number (default: 0)
--no-checkpoint     don't start at a checkpoint, start at iteration 1 with seed instead
--yield             write output to file regardless of whether a palindrome was found

Run selection:
--bench             Run short benchmark; alias for `--no-checkpoint --stop-at {LIMIT_SHORT}`
--bench             Run short benchmark; alias for `--no-checkpoint --stop-at {LIMIT_LONG_BENCH}`
--short             Run profiling run; alias for `--stop-at {LIMIT_PROFILING}`
--long              Run long run; alias for `--stop-at 0` (set by default)
");
        std::process::exit(0);
    }

    if short_help {
        println!("Usage: lychrel_base10_simd [options]");
        std::process::exit(0);
    }

    if version {
        std::process::exit(0);
    }

    initial_value.0.push(Limb::new_from_value(seed_number));

    let run_type_stop_at: usize;
    match run_type {
        RunType::Bench => {
            println!("Performing benchmark run.");
            start_at = None;
            no_checkpoint = true;
            run_type_stop_at = LIMIT_SHORT;
        }
        RunType::LongBench => {
            println!("Performing long benchmark run.");
            start_at = None;
            no_checkpoint = true;
            run_type_stop_at = LIMIT_LONG_BENCH;
        }
        RunType::Short => {
            println!("Performing profiling run.");
            if start_at.is_none() {
                no_checkpoint = true;
            }
            run_type_stop_at = LIMIT_PROFILING;
        }
        RunType::Long => {
            no_checkpoint ^= false;
            run_type_stop_at = LIMIT;
        }
    }

    let stop_at: usize = match stop_at {
        Some(stop_at) => stop_at,
        None => run_type_stop_at,
    };

    let checkpoint_path = Path::new(&checkpoint_path_str);
    match std::fs::read_dir(checkpoint_path) {
        Ok(entries) => {
            println!(
                "Using pre-existing checkpoints folder at {}",
                checkpoint_path.display()
            );
            if no_checkpoint {
                println!("Not starting from checkpoint.");
            } else {
                // since the folder exists, get the `Path`s of all files inside of it
                // filter out those that are irrelevant to our current seed
                let mut checkpoint_files: Vec<std::path::PathBuf> = entries
                    .filter_map(|entry| entry.ok())
                    .map(|entry| entry.path())
                    .filter(|path| {
                        path.file_name()
                            .and_then(|name| name.to_str())
                            .is_some_and(|s| s.ends_with(&format!("{seed_number:}_checkpoint")))
                    })
                    .collect();

                checkpoint_files.sort_unstable_by_key(|path| {
                    path.file_name()
                        .and_then(|s| s.to_str())
                        .and_then(|s| s.split('.').next())
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap()
                });

                match start_at {
                    Some(start_at_value) => {
                        let checkpoint_path = match checkpoint_files.into_iter().find(|path| {
                            path.file_name()
                                .and_then(|name| name.to_str())
                                .and_then(|s| s.split('.').next())
                                .and_then(|s| s.parse::<usize>().ok())
                                .is_some_and(|i| i == start_at_value)
                        }) {
                            Some(path) => path,
                            None => {
                                eprintln!(
                                    "No checkpoint found with index {start_at_value:} in {}",
                                    checkpoint_path.canonicalize()?.display()
                                );
                                std::process::exit(1);
                            }
                        };
                        let checkpoint_data = std::fs::read(checkpoint_path)?;
                        let checkpoint = Checkpoint::new(start_at_value, checkpoint_data);
                        (initial_value, _) = Integer::from_checkpoint(checkpoint, allocator);
                        starting_iteration = start_at_value + 1;
                    }

                    None => {
                        if checkpoint_files.len() >= 2 {
                            // use the second to last checkpoint
                            let checkpoint_path = &checkpoint_files[checkpoint_files.len() - 2];
                            let checkpoint_data = std::fs::read(checkpoint_path)?;
                            let checkpoint_iteration = checkpoint_path
                                .file_name()
                                .and_then(|name| name.to_str())
                                .and_then(|s| s.split('.').next())
                                .and_then(|s| s.parse::<usize>().ok())
                                .unwrap();
                            let checkpoint = Checkpoint::new(checkpoint_iteration, checkpoint_data);
                            (initial_value, _) = Integer::from_checkpoint(checkpoint, allocator);
                            starting_iteration = checkpoint_iteration + 1;
                        }
                    }
                }
            }
        }
        Err(_) => {
            match std::fs::create_dir(checkpoint_path) {
                Ok(_) => eprintln!("Created new local checkpoints folder in local directory"),
                Err(err) => {
                    eprintln!("Error creating local checkpoints folder: {err}");
                    std::process::exit(1);
                }
            };
        }
    }

    println!("limit: {stop_at:}");

    println!("{}", "-".repeat(32));

    let (tx, rx) = mpsc::channel::<StatusReport>();

    let iteration_handle = thread::spawn(move || {

        iterate(starting_iteration..stop_at, initial_value, Some(tx))
    });

    let mut step_time = Instant::now();

    for status_report in rx {
        let elapsed_time = step_time.elapsed();
        step_time = Instant::now();

        let i = status_report.iteration;
        let current_value = status_report.current_value;

        let log_idx = i.div_floor(LOG_MASK) % 16;

        let rate: f64 =
            unsafe { std::intrinsics::fdiv_fast(LOG_MASK as f64, elapsed_time.as_secs_f64()) };

        println!(
            "{}:{} {i}; {rate:.2} iter/sec",
            if log_idx == 0 { 16 } else { log_idx },
            if (log_idx < 10) && log_idx > 0 {
                " "
            } else {
                ""
            },
        );

        if current_value.is_some() {
            cold_path();

            let current_value = unsafe { current_value.unwrap_unchecked() };

            let num_limbs = current_value.0.len();

            #[cfg(not(feature = "no-verify"))]
            {
                let checkpoint_path =
                    Path::new(&checkpoint_dir).join(format!("{i}.{INITIAL_SEED:}_checkpoint"));

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
                    let mut buffer = Vec::with_capacity(checkpoint.integer.len());
                    match file.read_to_end(&mut buffer) {
                        Ok(_) => {
                            let read_checkpoint = Checkpoint::new(i, buffer);
                            if likely(read_checkpoint == checkpoint) {
                                println!("OK");
                            } else {
                                cold_path();
                                println!("FAILED");
                                eprintln!("Checkpoint validation failed at checkpoint {i:}");
                                let read_checkpoint_len = read_checkpoint.data().1.len();
                                let read_checkpoint_vector_size: u8 =
                                    if read_checkpoint_len.is_multiple_of(64) {
                                        64
                                    } else if read_checkpoint_len.is_multiple_of(32) {
                                        32
                                    } else if read_checkpoint_len.is_multiple_of(16) {
                                        16
                                    } else if read_checkpoint_len.is_multiple_of(8) {
                                        8
                                    } else if read_checkpoint_len.is_multiple_of(4) {
                                        4
                                    } else if read_checkpoint_len.is_multiple_of(2) {
                                        2
                                    } else {
                                        1
                                    };

                                if !checkpoint
                                    .data()
                                    .1
                                    .len()
                                    .is_multiple_of(read_checkpoint_vector_size as usize)
                                {
                                    cold_path();
                                    eprintln!(
                                        "It is possible that the current machine uses a different word size than the machine that generated this checkpoint.\nRead vector size: {read_checkpoint_vector_size:} bytes\nCurrent vector size: {LV_LEN:} bytes",
                                    )
                                }
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
                            println!("FAILED: {e}");
                            std::process::exit(1);
                        }
                    }
                }
            }
            print!("{:} limbs, approx. {:} digits, ", num_limbs, num_limbs * 64);
            
            #[cfg(target_family = "windows")]
            if let Some(usage) = memory_stats::memory_stats() {
                print!("{:} KiB physical memory, ", usage.physical_mem / 1024);
                println!("{:} KiB virtual memory.", usage.virtual_mem / 1024);
            } else {
                println!("{:} KiB of memory", (num_limbs * LV_BYTES) / 1024);
            }

            #[cfg(not(target_family = "windows"))]
            println!("{:} KiB of memory", (num_limbs * LV_BYTES) / 1024);

            use std::intrinsics::{fdiv_fast, fmul_fast};
            let tetrahexacontabytes_per_second = unsafe { fmul_fast(num_limbs as f64, rate) };
            println!(
                "Current stats:\n{:.2} GiBps\n{:.3} million limbs / sec\n{:.2} billion digits / sec\n",
                unsafe {
                    fdiv_fast(
                        tetrahexacontabytes_per_second,
                        (1073741824 / LV_BYTES) as f64,
                    )
                },
                unsafe { fdiv_fast(tetrahexacontabytes_per_second, 1_000_000f64) },
                unsafe { fdiv_fast(tetrahexacontabytes_per_second, 15625000f64) },
            );
            // current rate = 64(num_limbs) / 1073741824
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

    let found_palindrome: bool = last_iteration < stop_at;

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
            format! {": {:}", &end_integer}
        } else {
            ".".to_string()
        }
    );

    let file_prefix = if found_palindrome { "FOUND_" } else { "yield_" };

    if unlikely(found_palindrome) || write_yield {
        println!("Writing packed found palindrome to \"{file_prefix}{last_iteration}.txt\"");
        std::fs::write(
            format!("{file_prefix}{last_iteration}.txt"),
            end_integer.pack().into_bytes(),
        )
        .unwrap();
    }

    Ok(())
}

#[cfg(test)]
mod tests;
