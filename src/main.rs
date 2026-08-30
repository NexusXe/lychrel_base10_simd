#![feature(likely_unlikely)]
#![feature(int_roundings)]
#![feature(core_intrinsics)]
#![allow(internal_features)]
#![feature(allocator_api)]
#![feature(portable_simd)]
#![feature(const_convert)]
#![feature(const_trait_impl)]
#![feature(const_cmp)]
#![feature(const_eval_select)]
#![feature(const_default)]
#![allow(unused_features)]
#![cfg_attr(
    all(any(target_arch = "x86_64", target_arch = "x86", not(feature = "no-avx"))),
    feature(stdarch_const_x86)
)]
#![deny(clippy::all)]
#![allow(
    clippy::missing_safety_doc,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::inline_always
)]
#![feature(trivial_bounds)]

pub mod integer_limb;

#[cfg(all(
    not(feature = "global-alloc"),
    any(target_family = "windows", target_family = "unix")
))]
use integer_limb::HugePageAllocator;

use integer_limb::{
    Checkpoint, Integer, LV_BYTES, LV_LEN, Limb, LimbVecScalar, WV_LEN, WideVecScalar,
};

use std::alloc::Global;
use std::any::type_name;
use std::hint::{cold_path, unlikely};
use std::intrinsics::{fdiv_fast, fmul_fast};
use std::path::Path;
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

#[cfg(not(feature = "no-verify"))]
use std::io::Read;

mod iterate;

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    const INITIAL_SEED: u128 = 196;
    const LIMIT_SHORT: usize = 603_567;
    const LIMIT_LONG_BENCH: usize = LIMIT_SHORT * 2;
    const LIMIT_PROFILING: usize = LIMIT_SHORT * 5;
    const DEFAULT_CHECKPOINT_DIR: &str = "./checkpoints";

    #[cfg(any(not(target_feature = "avx512bw"), feature = "no-avx"))]
    eprintln!("\x1b[1;31mWarning:\x1b[22m Using portable_simd fallback code. This will be very, very slow.
The portable_simd implementation of this program is mostly for reference, and is not intended for end-user use.
\x1b[1mProceed with caution.\x1b[0m\n");

    //const LIMIT: usize = 500;
    //const LIMIT: usize = 100_358;
    const LIMIT: usize = usize::MAX;

    #[cfg(all(
        not(feature = "global-alloc"),
        any(target_family = "windows", target_family = "unix")
    ))]
    let allocator = HugePageAllocator::init()?;

    #[cfg(not(all(
        not(feature = "global-alloc"),
        any(target_family = "windows", target_family = "unix")
    )))]
    let allocator = Global;

    let mut starting_iteration: usize = 1;
    let args: Vec<String> = std::env::args().collect();

    let checkpoint_dir = std::env::var("LYCHREL_CHECKPOINTS_PATH").map_or_else(
        |_| DEFAULT_CHECKPOINT_DIR.to_string(),
        |path| path.trim_end_matches(['/', '\\']).to_string(),
    );

    #[derive(PartialEq, Eq, Clone, Copy)]
    enum ExecType {
        Run,
        Read,
    }

    enum RunType {
        Bench,
        LongBench,
        LongerBench,
        Short,
        Long,
    }

    let mut help: bool = false;
    let mut short_help: bool = false;
    let mut version: bool = false;
    let mut skip_next_arg: bool = false;
    let mut quiet: bool = false;

    let mut start_at: Option<usize> = None;

    let mut stop_at: Option<usize> = None;
    let mut checkpoint_path_str = checkpoint_dir.clone();
    let mut seed_number: u128 = INITIAL_SEED;

    let mut no_checkpoint: bool = false;
    let mut write_yield = false;
    let mut run_type = RunType::Long;

    let mut read_path: Option<&Path> = None;
    let mut read_verify: bool = false;

    #[cfg(all(
        not(feature = "global-alloc"),
        any(target_family = "windows", target_family = "unix")
    ))]
    let mut initial_value: Integer<HugePageAllocator> = Integer(Vec::new_in(allocator));

    #[cfg(not(all(
        not(feature = "global-alloc"),
        any(target_family = "windows", target_family = "unix")
    )))]
    let mut initial_value: Integer<Global> = Integer(Vec::new());

    let exec_type = args.get(1).map_or_else(
        || {
            eprintln!("Please specify a run type\n");
            None
        },
        |arg| match arg.as_str() {
            "run" => Some(ExecType::Run),
            "read" => Some(ExecType::Read),
            _ => {
                eprintln!("Unexpected run type argument: {arg}\n");
                None
            }
        },
    );

    for (idx, arg) in args.iter().enumerate().skip(2) {
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
            "--quiet" => quiet = true,
            "--seed" => {
                skip_next_arg = true;
                seed_number = args.get(idx + 1).map_or_else(
                    || {
                        eprintln!("Please specify a seed number");
                        std::process::exit(1);
                    },
                    |seed| {
                        seed.parse::<u128>().unwrap_or_else(|_| {
                            eprintln!("Please specify a valid seed number");
                            std::process::exit(1);
                        })
                    },
                );
            }
            "--checkpoint-dir" => {
                skip_next_arg = true;
                checkpoint_path_str = args.get(idx + 1).map_or_else(
                    || {
                        eprintln!("Please specify a checkpoint directory path");
                        std::process::exit(1);
                    },
                    std::clone::Clone::clone,
                );
            }
            "--start-at" => {
                skip_next_arg = true;
                start_at = args.get(idx + 1).map_or_else(
                    || {
                        eprintln!("Please specify a start value");
                        std::process::exit(1);
                    },
                    |start_at_str| {
                        start_at_str.parse::<usize>().map_or_else(
                            |_| {
                                eprintln!("Please specify a valid start value");
                                std::process::exit(1);
                            },
                            Some,
                        )
                    },
                );
            }
            "--stop-at" => {
                skip_next_arg = true;
                stop_at = args.get(idx + 1).map_or_else(
                    || {
                        eprintln!("Please specify a stop value");
                        std::process::exit(1);
                    },
                    |stop_at_str| {
                        stop_at_str.parse::<usize>().map_or_else(
                            |_| {
                                eprintln!("Please specify a valid stop value");
                                std::process::exit(1);
                            },
                            |stop_at_val| {
                                Some(if stop_at_val == 0 {
                                    LIMIT
                                } else {
                                    stop_at_val + 1
                                })
                            },
                        )
                    },
                );
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
            "--longer-bench" => {
                run_type = RunType::LongerBench;
            }
            "--long" => {
                run_type = RunType::Long;
            }
            "--path" => {
                skip_next_arg = true;
                read_path = args.get(idx + 1).map_or_else(
                    || {
                        eprintln!("Please specify a path");
                        std::process::exit(1);
                    },
                    |path| Some(Path::new(path)),
                );
            }
            "--verify" => {
                read_verify = true;
            }

            _ => {
                eprintln!("unrecognized argument: {arg}");
                short_help = true;
            }
        }
    }

    if help {
        println!("lychrel_base10_simd usage:
lychrel_base10_simd run [options]
lychrel_base10_simd read --path [path] [options]

General options:
--help              show this help
--version           show version

Run options:
--seed <seed>       specify the base number to iterate over (default: {INITIAL_SEED:})
--checkpoint-dir    specify the directory of checkpoint files (default: {DEFAULT_CHECKPOINT_DIR:})
--start-at          specify starting checkpoint iteration number (default: second to last if available)
--stop-at           specify iteration target number (default: 0)
--no-checkpoint     don't start at a checkpoint, start at iteration 1 with seed instead
--yield             write output to file regardless of whether a palindrome was found

    Run type selection:
    --bench             Run short benchmark; alias for `--no-checkpoint --stop-at {LIMIT_SHORT}`
    --bench             Run short benchmark; alias for `--no-checkpoint --stop-at {LIMIT_LONG_BENCH}`
    --short             Run profiling run; alias for `--stop-at {LIMIT_PROFILING}`
    --long              Run long run; alias for `--stop-at 0` (set by default)

Read options:
--path              path of checkpoint file to read
--verify            verify that the read value is a valid Integer

Note: Run options used with read / read options used with run will be ignored
");
        std::process::exit(0);
    }

    if short_help
        || exec_type.is_none()
        || (exec_type.unwrap() == ExecType::Read && read_path.is_none())
    {
        println!(
            "Usage:
lychrel_base10_simd run [options]
lychrel_base10_simd read --path [path] [options]
For more information, pass --help"
        );
        std::process::exit(0);
    }

    let exec_type = exec_type.unwrap();

    if version {
        std::process::exit(0);
    }

    match exec_type {
        ExecType::Run => {
            println!(
                "
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
                RunType::LongerBench => {
                    println!("Performing longer benchmark run.");
                    start_at = None;
                    no_checkpoint = true;
                    run_type_stop_at = LIMIT_LONG_BENCH * 2;
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

            let stop_at: usize = stop_at.unwrap_or(run_type_stop_at);

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
                        let mut checkpoint_files: Vec<std::path::PathBuf> =
                            entries
                                .filter_map(std::result::Result::ok)
                                .map(|entry| entry.path())
                                .filter(|path| {
                                    path.file_name().and_then(|name| name.to_str()).is_some_and(
                                        |s| s.ends_with(&format!("{seed_number:}_checkpoint")),
                                    )
                                })
                                .collect();

                        #[cfg(target_family = "unix")]
                        {
                            use std::cmp::Ordering;
                            use std::ffi::{c_int, c_void};
                            use std::path::{Path, PathBuf};
                            #[inline(never)]
                            fn get_key_from_path(path: &Path) -> usize {
                                unsafe {
                                    path.file_name()
                                        .and_then(|s| s.to_str())
                                        .and_then(|s| s.split('.').next())
                                        .map(|s| s.parse::<usize>().unwrap_unchecked())
                                        .unwrap_unchecked()
                                }
                            }

                            #[inline]
                            extern "C" fn compare_paths(
                                a: *const c_void,
                                b: *const c_void,
                            ) -> c_int {
                                let ordering = {
                                    let path_a = unsafe { &*(a as *const PathBuf) };
                                    let path_b = unsafe { &*(b as *const PathBuf) };

                                    let key_a = get_key_from_path(path_a);
                                    let key_b = get_key_from_path(path_b);

                                    key_a.cmp(&key_b)
                                };

                                match ordering {
                                    Ordering::Less => -1,
                                    Ordering::Equal => 0,
                                    Ordering::Greater => 1,
                                }
                            }

                            // Only sort if the vector is not empty
                            if !checkpoint_files.is_empty() {
                                unsafe {
                                    libc::qsort(
                                        checkpoint_files.as_mut_ptr() as *mut c_void,
                                        checkpoint_files.len() as libc::size_t,
                                        std::mem::size_of::<PathBuf>() as libc::size_t,
                                        Some(compare_paths),
                                    );
                                }
                            }
                        }

                        #[cfg(not(target_family = "unix"))]
                        {
                            checkpoint_files.sort_unstable_by_key(|path| {
                                path.file_name()
                                    .and_then(|s| s.to_str())
                                    .and_then(|s| s.split('.').next())
                                    .and_then(|s| s.parse::<usize>().ok())
                                    .unwrap()
                            });
                        }

                        match start_at {
                            Some(start_at_value) => {
                                let Some(checkpoint_path) =
                                    checkpoint_files.into_iter().find(|path| {
                                        path.file_name()
                                            .and_then(|name| name.to_str())
                                            .and_then(|s| s.split('.').next())
                                            .and_then(|s| s.parse::<usize>().ok())
                                            .is_some_and(|i| i == start_at_value)
                                    })
                                else {
                                    eprintln!(
                                        "No checkpoint found with index {start_at_value:} in {}",
                                        checkpoint_path.canonicalize()?.display()
                                    );
                                    std::process::exit(1);
                                };
                                let checkpoint_data = std::fs::read(checkpoint_path)?;
                                let checkpoint = Checkpoint::new(start_at_value, checkpoint_data);
                                (initial_value, _) =
                                    Integer::from_checkpoint(&checkpoint, allocator);
                                starting_iteration = start_at_value + 1;
                            }

                            None => {
                                if checkpoint_files.len() >= 2 {
                                    // use the second to last checkpoint
                                    let checkpoint_path =
                                        &checkpoint_files[checkpoint_files.len() - 2];
                                    let checkpoint_data = std::fs::read(checkpoint_path)?;
                                    let checkpoint_iteration = checkpoint_path
                                        .file_name()
                                        .and_then(|name| name.to_str())
                                        .and_then(|s| s.split('.').next())
                                        .and_then(|s| s.parse::<usize>().ok())
                                        .unwrap();
                                    let checkpoint =
                                        Checkpoint::new(checkpoint_iteration, checkpoint_data);
                                    (initial_value, _) =
                                        Integer::from_checkpoint(&checkpoint, allocator);
                                    starting_iteration = checkpoint_iteration + 1;
                                }
                            }
                        }
                    }
                }
                Err(_) => match std::fs::create_dir(checkpoint_path) {
                    Ok(()) => {
                        eprintln!("Created new local checkpoints folder in local directory");
                    }
                    Err(err) => {
                        eprintln!("Error creating local checkpoints folder: {err}");
                        std::process::exit(1);
                    }
                },
            }

            if starting_iteration > 1 {
                println!("Starting at iteration {starting_iteration:}");
            }

            println!("limit: {stop_at:}");

            let (tx, rx) = mpsc::channel::<iterate::StatusReport>();

            let iteration_handle = {
                println!("{}", "-".repeat(32));
                thread::spawn(move || {
                    iterate::iterate(starting_iteration..stop_at, initial_value, Some(&tx))
                })
            };

            let mut step_time = Instant::now();

            for status_report in rx {
                let elapsed_time = step_time.elapsed();
                step_time = Instant::now();

                let i = status_report.iteration;
                let current_value = status_report.current_value;

                let rate: f64 =
                    unsafe { fdiv_fast(iterate::LOG_MASK as f64, elapsed_time.as_secs_f64()) };

                if !quiet {
                    let log_idx = i.div_floor(iterate::LOG_MASK) % 16;
                    println!(
                        "{}:{} {i}; {rate:.2} iter/sec",
                        if log_idx == 0 { 16 } else { log_idx },
                        if (log_idx < 10) && log_idx > 0 {
                            " "
                        } else {
                            ""
                        },
                    );
                }

                if current_value.is_some() {
                    cold_path();

                    let current_value = unsafe { current_value.unwrap_unchecked() };

                    let num_limbs = current_value.0.len();

                    #[cfg(not(feature = "no-verify"))]
                    {
                        let checkpoint_path = Path::new(&checkpoint_path_str)
                            .join(format!("{i}.{seed_number:}_checkpoint"));

                        println!(
                            "Reached checkpoint: {}",
                            unsafe { checkpoint_path.file_name().unwrap_unchecked() }.display()
                        );
                        let checkpoint = current_value.into_checkpoint(i);
                        cold_path();

                        if checkpoint_path.exists() && checkpoint_path.is_file() {
                            print!("Checkpoint already exists; validating... ");
                            // read the file
                            let Ok(mut file) = std::fs::File::open(checkpoint_path) else {
                                cold_path();
                                eprintln!("UNABLE TO OPEN FILE\nContinuing anyway...");

                                continue;
                            };
                            let mut buffer = Vec::with_capacity(checkpoint.integer.len());
                            if file.read_to_end(&mut buffer).is_ok() {
                                use std::hint::likely;

                                let read_checkpoint = Checkpoint::new(i, buffer);
                                if likely(read_checkpoint == checkpoint) {
                                    println!("OK");
                                } else {
                                    cold_path();
                                    println!("FAILED");
                                    eprintln!("Checkpoint validation failed at checkpoint {i:}");
                                    let read_checkpoint_len = read_checkpoint.data().1.len();
                                    let read_checkpoint_vector_size: u8 =
                                        1u8 << read_checkpoint_len.trailing_zeros().min(6);

                                    if !checkpoint
                                        .data()
                                        .1
                                        .len()
                                        .is_multiple_of(read_checkpoint_vector_size as usize)
                                    {
                                        cold_path();
                                        eprintln!(
                                            "It is possible that the current machine uses a different word size than the machine that generated this checkpoint.\nRead vector size: {read_checkpoint_vector_size:} bytes\nCurrent vector size: {LV_LEN:} bytes",
                                        );
                                    }
                                    std::process::exit(1)
                                }
                            } else {
                                cold_path();
                                eprintln!("UNABLE TO READ FILE");
                                eprintln!("Continuing anyway...");
                            }
                        } else {
                            print!("Writing checkpoint to {}... ", checkpoint_path.display());
                            let data = checkpoint.data().1;
                            let data_length = data.len();
                            match std::fs::write(checkpoint_path, data) {
                                Ok(()) => {
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

                    let limbs_per_second = unsafe { fmul_fast(num_limbs as f64, rate) };
                    let bytes_per_second = unsafe { fmul_fast(limbs_per_second, LV_BYTES as f64) };
                    let gibibytes_per_second =
                        unsafe { fmul_fast(bytes_per_second, 9.313_225_746_154_785e-10) }; // 1 / 1024^3
                    let gigabits_per_second = unsafe { fmul_fast(bytes_per_second, 8.0e-9) }; // 8 / 1_000_000_000
                    println!(
                        "Current stats:\n{:.2} GiBps ({:.2} Gbps)\n{:.3} million limbs / sec\n{:.2} billion digits / sec\n",
                        gibibytes_per_second,
                        gigabits_per_second,
                        unsafe { fmul_fast(limbs_per_second, 1.0e-6) },
                        unsafe { fmul_fast(bytes_per_second, 1.0e-9) },
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
                    format!(": {end_integer:}")
                } else {
                    ".".to_string()
                }
            );

            let file_prefix = if found_palindrome { "FOUND_" } else { "yield_" };

            if unlikely(found_palindrome) || write_yield {
                println!(
                    "Writing packed found palindrome to \"{file_prefix}{last_iteration}.txt\""
                );
                std::fs::write(
                    format!("{file_prefix}{last_iteration}.txt"),
                    end_integer.pack().into_bytes(),
                )
                .unwrap();
            }

            Ok(())
        }

        ExecType::Read => {
            let file_path = read_path.unwrap();
            // try to read from the file
            let file = std::fs::read(file_path)?;

            let data: Vec<[LimbVecScalar; LV_LEN]> =
                Integer::<Global>::chop(&file).unwrap_or_else(|| {
                    eprintln!("\x1b[1;31merror\x1b[0m: file length is not a multiple of 64 bytes");
                    std::process::exit(1);
                });

            let global_allocator = Global;

            let integer = Integer::from_bytes(&data, global_allocator).unpack(global_allocator);

            if read_verify && integer.has_carries() {
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

            Ok(())
        }
    }
}

#[cfg(test)]
mod tests;
