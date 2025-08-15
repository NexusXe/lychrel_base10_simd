#![feature(portable_simd)]
#![feature(const_from)]
#![feature(const_trait_impl)]
#![feature(likely_unlikely)]
#![feature(cold_path)]

mod integer_limb;
use integer_limb::{Checkpoint, Integer, Limb};
use std::hint::{cold_path, likely, unlikely};
use std::io::Read;
use std::path::Path;
use std::simd::prelude::*;
use std::thread;
use std::time::Instant;

pub struct IterationResult {
    last_iteration: usize,
    start_time: Instant,
    end_integer: Integer,
}

const CHECKPOINT_DIR: &str = "./checkpoints";
const INITIAL_SEED: &str = "196";

/// Iterates over a given input. If the returned `usize` is less than `range.end`, a palindrome was found.
pub(crate) fn iterate(range: std::ops::Range<usize>, starting_integer: Integer) -> IterationResult {
    let checkpoint_suffix = format!("{INITIAL_SEED:}_checkpoint");

    let mut current_iteration: Integer = starting_integer;
    let mut reverse: Integer = Integer(Vec::<Limb>::new());

    reverse.0.reserve(current_iteration.0.len());
    current_iteration.reverse_into_integer(&mut reverse);

    let mut carried: bool = false;
    let mut i: usize = range.start;
    let mut acc: u8 = 0;

    let checkpoint_path = Path::new(CHECKPOINT_DIR);

    let start_time = Instant::now();
    let mut step_time = Instant::now();

    while likely(i < range.end) {
        if unlikely(unlikely(!carried) && unlikely(current_iteration.0 == reverse.0)) {
            cold_path();
            break;
        }
        let reverse_scrap = reverse.clone();
        carried = current_iteration.add_into_self(reverse_scrap);
        const STEP_SIZE: usize = 2usize.pow(14);
        const ACC_LIMIT: u8 = 16;
        if unlikely(i.is_multiple_of(STEP_SIZE)) {
            acc += 1;

            let elapsed_time = step_time.elapsed();
            step_time = Instant::now();

            let rate: f32 = STEP_SIZE as f32 / elapsed_time.as_secs_f32();

            println!(
                "{i}; {:} until checkpoint; {rate:} iter/sec",
                ACC_LIMIT - acc
            );
            if unlikely(acc == ACC_LIMIT) {
                cold_path();
                acc = 0;

                let checkpoint_path = checkpoint_path.join(format!("{i}.{}", &checkpoint_suffix));
                let current_iteration_cloned = current_iteration.clone();
                thread::spawn(move || {
                    let checkpoint = current_iteration_cloned.into_checkpoint(i);
                    if checkpoint_path.exists() && checkpoint_path.is_file() {
                        print!("Checkpoint already exists; validating... ");
                        // read the file
                        let mut file = std::fs::File::open(checkpoint_path).unwrap();
                        let mut buffer = Vec::new();
                        file.read_to_end(&mut buffer).unwrap();
                        let read_checkpoint = Checkpoint::new(i, buffer);
                        if read_checkpoint == checkpoint {
                            println!("OK");
                        } else {
                            println!("FAILED");

                            eprintln!("Checkpoint validation failed at checkpoint {i:}");
                            std::process::exit(1)
                        }
                    } else {
                        print!("Writing checkpoint to {}... ", checkpoint_path.display());
                        let data = checkpoint.data().1;
                        let data_length = data.len();
                        std::fs::write(checkpoint_path, data).unwrap();
                        println!("OK");
                        println!("Wrote {:} KiB", data_length / 1024);
                    }
                });

                let num_limbs = current_iteration.0.len();

                println!(
                    "{:} limbs, approx. {:} digits, {:} KiB of memory",
                    num_limbs,
                    num_limbs * 64,
                    (num_limbs * 64).div_ceil(1024)
                );
            }
        }

        current_iteration.reverse_into_integer(&mut reverse);

        i += 1;
    }
    IterationResult {
        last_iteration: i,
        start_time,
        end_integer: current_iteration,
    }
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    //const LIMIT: usize = 603_567;
    //const LIMIT: usize = 500;
    //const LIMIT: usize = 100_358;
    const LIMIT: usize = u32::MAX as usize;

    let mut initial_value: Integer = integer!(INITIAL_SEED);
    let mut starting_iteration: usize = 1;

    let args: Vec<String> = std::env::args().collect();

    if !args.contains(&"--no-checkpoint".to_string()) {
        let checkpoint_path = Path::new(CHECKPOINT_DIR);

        match std::fs::read_dir(checkpoint_path) {
            Ok(entries) => {
                println!("Using pre-existing checkpoints folder.");
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
                        .and_then(|s| s.split('.').next()) // Get the first part before '.'
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap()
                });

                if checkpoint_files.len() >= 2 {
                    // use the second to last checkpoint
                    let used_checkpoint_path = &checkpoint_files[checkpoint_files.len() - 2];
                    let used_checkpoint_iteration = used_checkpoint_path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .and_then(|s| s.split('.').next())
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap();
                    let used_checkpoint = std::fs::read(used_checkpoint_path)?;
                    let checkpoint = Checkpoint::new(used_checkpoint_iteration, used_checkpoint);
                    (initial_value, starting_iteration) = Integer::from_checkpoint(checkpoint);
                    println!("Starting from checkpoint at iteration {starting_iteration:}");
                    starting_iteration += 1;
                }
            }
            Err(_) => {
                std::fs::create_dir(checkpoint_path)?;
                println!(
                    "Created new local checkpoints folder at {}",
                    checkpoint_path.canonicalize()?.display()
                );
            }
        }
    } else {
        println!("Not starting from checkpoint.");
    }

    let result = iterate(starting_iteration..LIMIT, initial_value);
    let (last_iteration, start_time, end_integer) =
        (result.last_iteration, result.start_time, result.end_integer);

    let found_palindrome: bool = last_iteration < LIMIT;

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

    if found_palindrome {
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
