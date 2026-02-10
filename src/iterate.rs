use super::integer_limb::Integer;
use std::alloc::{Allocator, Global};
use std::hint::{cold_path, likely, unlikely};
use std::sync::mpsc::Sender;
use std::time::Instant;

pub struct IterationResult<T: Allocator + Clone + Copy> {
    pub(crate) last_iteration: usize,
    pub(crate) start_time: Instant,
    pub(crate) end_integer: Integer<T>,
}

pub struct StatusReport {
    pub(crate) iteration: usize,
    pub(crate) current_value: Option<Integer<Global>>,
}

pub const LOG_FREQUENCY_EXP: usize = 14;

pub const LOG_MASK: usize = 2usize.pow(LOG_FREQUENCY_EXP as u32);

/// Iterates over a given input. If the returned `usize` is less than `range.end`, a palindrome was found.
#[inline]
pub fn iterate<T: std::alloc::Allocator + Clone + Copy>(
    range: std::ops::Range<usize>,
    starting_integer: Integer<T>,
    tx: Option<&Sender<StatusReport>>,
) -> IterationResult<T> {
    let mut current_iteration = starting_integer;

    current_iteration.0.reserve(2048.min(range.end / 100));

    #[allow(unused_variables)]
    let mut carried: bool = true; // ignore palindrome check on the first loop
    let mut i: usize = range.start;

    let start_time = Instant::now();

    #[allow(unused_assignments)]
    while likely(i < range.end) {
        #[cfg(not(feature = "no-verify"))]
        if unlikely(!carried) {
            cold_path();
            //eprintln!("Checking...");
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
