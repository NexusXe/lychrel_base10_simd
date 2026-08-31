//! The single-threaded reference implementation: the original fused
//! reverse-and-add kernel and its iteration loop. The parallel engine is
//! validated against it by the parity tests in src/parallel/tests.rs, and a
//! binary built with the `reference-impl` feature runs it for `--threads 1`,
//! which keeps old benchmark numbers directly comparable.

use crate::impossible;
use crate::integer_limb::{
    Integer, LV_LEN, Limb, LimbVec, LimbVecScalar, WV_LEN, WideVec, WideVecScalar, add_resolve_limb,
};
use crate::parallel::{IterationResult, LOG_MASK, StatusReport};
use std::alloc::{Allocator, Global};
use std::hint::{cold_path, likely, unlikely};
use std::sync::mpsc::Sender;
use std::time::Instant;

impl<T: Allocator + Clone + Copy> Integer<T> {
    #[inline(always)]
    pub fn fused_reverse_add_asm_interleave(&mut self) -> bool {
        use std::ptr::read_unaligned;

        if self.0.is_empty() {
            impossible!("Tried to reverse and add empty integer");
        }

        let total_limbs = self.num_limbs();
        if total_limbs > 2usize.pow(26) {
            impossible!("Tried to iterate over an integer with more than 2^26 limbs");
        }

        self.0.push(Limb::new()); // padding

        let skip_len = LV_LEN as u8 - unsafe { self.0.get_unchecked(total_limbs - 1).len() };

        if skip_len >= LV_LEN as u8 {
            impossible!("skip_len out of bounds");
        }

        let limbs_ptr = self.0.as_mut_ptr().cast::<LimbVec>();
        let rev_ptr = unsafe { limbs_ptr.add(total_limbs - 1) };
        if !std::ptr::eq(rev_ptr, unsafe {
            &raw const self.0.get_unchecked_mut(total_limbs.unchecked_sub(1)).0
        }) {
            impossible!("Incoherent rev_ptr");
        }

        Limb::zip_halves(limbs_ptr, total_limbs);

        #[cfg(all(target_arch = "x86_64", not(feature = "no-prefetch")))]
        #[allow(clippy::pointers_in_nomem_asm_block)] // ptr is being used for prefetch
        unsafe {
            std::arch::asm!(r#"
            # implicit xor {i:e}, {i:e}; 2 bytes, 1 uop
            2:
            prefetchw byte ptr [{limbs_ptr:r} + {i:r} * 8]      # 4 bytes, 1 uop
            add {i:l}, 8                                        # 3 or 4 bytes; fuses with conditional jump for 1 uop for both
            jns 2b                                              # 2 bytes; shares uop with add instruction
            "#,
            limbs_ptr = in(reg) limbs_ptr,
            i = inout(reg) 0 => _,
            options(nostack, nomem));
            // in addition to the first limbs, the last ones are also accessed first
            // however, they are likely still in cache
        }

        let mut overflowed = false;
        let mut ever_carried = false;

        // addition process
        // the reversed data is offset, but is guaranteed to be fully contained between the current and next cache line
        // conveniently, the next cache line is going to be needed soon anyway so this is fine
        for (_, limb) in self
            .0
            .iter_mut()
            .enumerate()
            .take_while(|(idx, _)| idx < &total_limbs)
        {
            unsafe {
                let limb_vec_ptr = &raw const limb.0;

                let reversed_limb: LimbVec =
                    read_unaligned(limb_vec_ptr.byte_add(skip_len as usize)) >> 4;

                overflowed = add_resolve_limb(limb, reversed_limb, overflowed, &mut ever_carried);
            }
        }

        let pad_ptr = unsafe { rev_ptr.add(1).cast::<WideVec>() };

        if unsafe { *pad_ptr != std::mem::zeroed() } {
            impossible!("Dirty padding data!");
        }

        if likely(overflowed) {
            //#[cfg(all(target_feature = "avx512f", feature = "stream"))]
            unsafe {
                // by writing the entire 64-byte cache line again, this memory doesn't have to be read at all to set the overflow
                const ONE_LV: LimbVec = LimbVec::from_array({
                    let mut arr = [0 as LimbVecScalar; LV_LEN];
                    arr[0] = 1;
                    arr
                });
                const ONE_WV: WideVec = WideVec::from_array({
                    let mut arr = [0 as WideVecScalar; WV_LEN];
                    arr[0] = 1;
                    arr
                });
                debug_assert_eq!(ONE_LV, std::mem::transmute::<WideVec, LimbVec>(ONE_WV));
                *pad_ptr = const {
                    let mut wide: WideVec = std::mem::zeroed();
                    wide.as_mut_array()[0] = 1;
                    wide
                };
            }

            // #[cfg(not(all(target_feature = "avx512f", feature = "stream")))]
            // unsafe {
            //     *((rev_ptr as usize).unchecked_add(LV_LEN) as *mut u8) = 1;
            // }
        } else {
            self.0.pop();
        }

        #[cfg(all(target_arch = "x86_64", not(feature = "no-prefetch")))]
        #[allow(clippy::pointers_in_nomem_asm_block)] // ptr is being used for prefetch
        unsafe {
            // prefetch the first 16 limbs since, for integers larger than L3$, they've probably been evicted by now
            // prefetching in asm because I don't want this loop unrolled
            // while it probably doesn't matter, less pollution in the L1i$ and the L1$ overall is good
            // asm version uses 12 bytes overall, whereas unrolled version was 7 * 16 = 112 bytes
            std::arch::asm!(r#"
            # implicit xor {i:e}, {i:e}; 2 bytes, 1 uop
            2:
            prefetchw byte ptr [{limbs_ptr:r} + {i:r} * 8]      # 4 bytes, 1 uop
            neg {i:r}                                           # 3 bytes, 1 uop
            prefetchw byte ptr [{rev_ptr:r} + 0 + {i:r} * 8]    # 4 bytes, 1 uop
            neg {i:r}                                           # 3 bytes, 1 uop
            add {i:l}, 8                                        # 3 or 4 bytes; fuses with conditional jump for 1 uop for both
            jns 2b                                              # 2 bytes; shares uop with add instruction
            "#,
            limbs_ptr = in(reg) limbs_ptr,
            rev_ptr = in(reg) rev_ptr,
            i = inout(reg) 0 => _,
            options(nostack, nomem));
            // in addition to the first limbs, the last ones are also accessed first
            // however, they are likely still in cache
        }

        likely(ever_carried)
    }
}

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
            let mut reverse = Integer(Vec::with_capacity(current_iteration.0.len()));
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
                    break;
                }
            } else {
                cold_path();
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
