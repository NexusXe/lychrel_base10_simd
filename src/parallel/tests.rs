use super::ParallelEngine;
use crate::integer_limb::{Integer, LV_LEN, Limb, LimbVec};
use rand::prelude::*;
use rand::rngs::SmallRng;
use std::alloc::Global;

/// A random valid integer: every digit 0..=9, top limb non-zero.
fn random_integer(num_limbs: usize, rng: &mut SmallRng) -> Integer<Global> {
    let mut limbs: Vec<Limb> = Vec::with_capacity(num_limbs);
    for _ in 0..num_limbs {
        let mut limb = Limb::default();
        for digit in limb.0.as_mut_array() {
            *digit = rng.random_range(0..=9);
        }
        limbs.push(limb);
    }
    let last = limbs.last_mut().unwrap();
    let top_digit = rng.random_range(0..LV_LEN);
    last.0.as_mut_array()[top_digit] = rng.random_range(1..=9);
    for digit in &mut last.0.as_mut_array()[top_digit + 1..] {
        *digit = 0;
    }
    Integer(limbs)
}

/// The engine must agree with the serial kernel step for step, carry flag
/// included, across sizes that produce empty blocks, odd middles, and growth.
#[test]
fn test_engine_matches_serial_kernel() {
    let mut rng = SmallRng::seed_from_u64(0x196);

    for num_threads in [1, 2, 3, 8] {
        let engine = ParallelEngine::new(num_threads);
        for num_limbs in [1usize, 2, 3, 4, 5, 7, 16, 33, 64, 100] {
            let serial = random_integer(num_limbs, &mut rng);
            let mut parallel = Integer(serial.0.clone());
            let mut serial = serial;

            for step in 0..50 {
                let carried_serial = serial.fused_reverse_add_asm_interleave();
                let carried_parallel = engine.step(&mut parallel);
                assert_eq!(
                    carried_serial, carried_parallel,
                    "carry flag diverged: {num_threads} threads, {num_limbs} limbs, step {step}"
                );
                assert_eq!(
                    serial.0.len(),
                    parallel.0.len(),
                    "length diverged: {num_threads} threads, {num_limbs} limbs, step {step}"
                );
                assert!(
                    serial == parallel,
                    "value diverged: {num_threads} threads, {num_limbs} limbs, step {step}"
                );
            }
        }
    }
}

/// All-nines inputs exercise the cross-block carry fixup: the increment has
/// to ripple through whole limbs of nines and grow the integer.
#[test]
fn test_engine_all_nines_fixup() {
    for num_threads in [2, 5] {
        let engine = ParallelEngine::new(num_threads);
        for num_limbs in [1usize, 3, 8, 21] {
            let limbs = vec![Limb(LimbVec::splat(9)); num_limbs];
            let mut serial = Integer(limbs.clone());
            let mut parallel = Integer(limbs);

            for step in 0..4 {
                let carried_serial = serial.fused_reverse_add_asm_interleave();
                let carried_parallel = engine.step(&mut parallel);
                assert_eq!(carried_serial, carried_parallel);
                assert_eq!(serial.0.len(), parallel.0.len());
                assert!(
                    serial == parallel,
                    "value diverged: {num_threads} threads, {num_limbs} limbs, step {step}"
                );
            }
        }
    }
}

/// The engine-driven trajectory from 196 must reach the same value as the
/// serial kernel after 500 iterations (the fixture in src/tests.rs).
#[test]
fn test_engine_196_trajectory() {
    let engine = ParallelEngine::new(4);
    let mut via_engine = crate::integer!("196");
    let mut via_kernel = crate::integer!("196");
    for _ in 0..500 {
        engine.step(&mut via_engine);
        via_kernel.fused_reverse_add_asm_interleave();
    }
    assert_eq!(via_engine.0.len(), via_kernel.0.len());
    assert!(via_engine == via_kernel);
}
