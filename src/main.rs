#![feature(portable_simd)]
#![feature(const_from)]
#![feature(const_trait_impl)]
#![feature(likely_unlikely)]

mod integer_limb;
use integer_limb::{Integer, Limb};
use std::simd::prelude::*;

pub fn main() {
    const LIMIT: usize = 603_567;
    //const LIMIT: usize = 500;
    //const LIMIT: usize = 100_358;
    //const LIMIT: usize = usize::MAX;

    let mut current_iteration: Integer = integer!("196");
    let mut reverse: Integer = Integer(Vec::<Limb>::new());
    reverse.0.reserve(current_iteration.0.len());
    current_iteration.reverse_into_integer(&mut reverse);
    let mut found_palindrome: bool = false;
    for i in 1..=LIMIT {
        if &current_iteration.0 == &reverse.0 {
            found_palindrome = true;
            break;
        }
        current_iteration = current_iteration + reverse.clone();
        if std::hint::unlikely(i % 8384 == 0) {
            println!("{i}");
            if std::hint::unlikely(i % 134144 == 0) {
                println!(": {:} digits", current_iteration.len());
                // checkpoint into file
                let checkpoint_data = current_iteration.clone().pack();
                let file_path = format!("checkpoint_{}.bin", i);
                // let _ = std::thread::spawn(move || {
                //     match std::fs::File::create(&file_path) {
                //         Ok(mut file) => {
                //             let serialized: Vec<[u8; 64]> = checkpoint_data.0.iter().map(|limb| limb.0.into()).collect();
                //             let serialized: Vec<u8> = serialized.into_iter().flatten().collect();
                //             if let Err(e) = std::io::Write::write_all(&mut file, &serialized) {
                //                 eprintln!("Error writing checkpoint file {}: {}", file_path, e);
                //             } else {
                //                 println!(" (Checkpoint saved)");
                //             }
                //         }
                //         Err(e) => eprintln!("Error creating checkpoint file {}: {}", file_path, e),
                //     }
                // });

            }
        }

        current_iteration.reverse_into_integer(&mut reverse);
        //println!("{i:}: {current_iteration:}");
    }

    println!("End result has {} limbs", current_iteration.0.len());
    println!(
        "{} find a palindrome.",
        if found_palindrome { "Did" } else { "Did not" }
    );
}
