use super::{PackedEngine, PackedInt, digit_at};
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

/// Every slot at and above `digits` in the packed buffer is zero.
fn assert_clean_padding(x: &PackedInt<Global>, context: &str) {
    for d in x.digits..x.a_cur().len() * super::DPL {
        assert_eq!(digit_at(x.a_cur(), d), 0, "dirty padding: {context}, slot {d}");
    }
}

/// Round trip through the packed representation.
#[test]
fn test_packed_round_trip() {
    let mut rng = SmallRng::seed_from_u64(0x196);
    for num_limbs in [1usize, 2, 3, 5, 8, 17, 33] {
        let integer = random_integer(num_limbs, &mut rng);
        let packed = PackedInt::from_integer(&integer, Global);
        assert!(packed.to_integer(Global) == integer, "{num_limbs} limbs");
        assert_clean_padding(&packed, "from_integer");
    }
}

/// The packed step must agree with the serial kernel step for step, carry
/// flag included, across sizes with partial lines, odd middles, and growth.
#[test]
fn test_packed_matches_serial_kernel() {
    let mut rng = SmallRng::seed_from_u64(0x196);

    for num_limbs in [1usize, 2, 3, 4, 5, 7, 16, 33, 64, 100] {
        let mut serial = random_integer(num_limbs, &mut rng);
        let mut packed = PackedInt::from_integer(&serial, Global);

        for step in 0..50 {
            let carried_serial = serial.fused_reverse_add_asm_interleave();
            let carried_packed = packed.step();
            assert_eq!(
                carried_serial, carried_packed,
                "carry flag diverged: {num_limbs} limbs, step {step}"
            );
            assert!(
                packed.to_integer(Global) == serial,
                "value diverged: {num_limbs} limbs, step {step}"
            );
            assert_eq!(packed.digits, serial.len() as usize);
        }
    }
}

/// All-nines inputs: the carry ripples through the whole number and grows it.
#[test]
fn test_packed_all_nines() {
    for num_limbs in [1usize, 2, 3, 8, 21] {
        let limbs = vec![Limb(LimbVec::splat(9)); num_limbs];
        let mut serial = Integer(limbs);
        let mut packed = PackedInt::from_integer(&serial, Global);

        for step in 0..4 {
            let carried_serial = serial.fused_reverse_add_asm_interleave();
            let carried_packed = packed.step();
            assert_eq!(carried_serial, carried_packed);
            assert!(
                packed.to_integer(Global) == serial,
                "value diverged: {num_limbs} limbs, step {step}"
            );
        }
    }
}

/// The packed trajectory from 196 must match the serial kernel's.
#[test]
fn test_packed_196_trajectory() {
    let mut serial = crate::integer!("196");
    let mut packed = PackedInt::from_integer(&serial, Global);
    for step in 0..1000 {
        serial.fused_reverse_add_asm_interleave();
        packed.step();
        assert!(
            packed.to_integer(Global) == serial,
            "value diverged at step {step}"
        );
    }
}

/// The engine must agree with the serial packed step (itself pinned to the
/// serial kernel above, funnel machinery included) step for step, across
/// thread counts and sizes that produce empty blocks, single-line blocks,
/// and growth.
#[test]
fn test_packed_engine_matches_serial() {
    let mut rng = SmallRng::seed_from_u64(0x196);

    for num_threads in [1, 2, 3, 8] {
        let mut engine = PackedEngine::new(num_threads);
        for num_limbs in [1usize, 2, 3, 4, 5, 7, 16, 33, 64, 100] {
            let start = random_integer(num_limbs, &mut rng);
            let mut serial = PackedInt::from_integer(&start, Global);
            let mut threaded = PackedInt::from_integer(&start, Global);

            for step in 0..50 {
                let carried_serial = serial.step();
                let carried_threaded = engine.step(&mut threaded);
                assert_eq!(
                    carried_serial, carried_threaded,
                    "carry flag diverged: {num_threads} threads, {num_limbs} limbs, step {step}"
                );
                assert_eq!(serial.digits, threaded.digits);
                assert!(
                    threaded.to_integer(Global) == serial.to_integer(Global),
                    "value diverged: {num_threads} threads, {num_limbs} limbs, step {step}"
                );
                assert_clean_padding(
                    &threaded,
                    &format!("{num_threads} threads, {num_limbs} limbs, step {step}"),
                );
            }
        }
    }
}

/// All-nines inputs exercise the cross-block increment fixup.
#[test]
fn test_packed_engine_all_nines() {
    for num_threads in [2, 5] {
        let mut engine = PackedEngine::new(num_threads);
        for num_limbs in [1usize, 3, 8, 21] {
            let limbs = vec![Limb(LimbVec::splat(9)); num_limbs];
            let mut serial = PackedInt::from_integer(&Integer(limbs.clone()), Global);
            let mut threaded = PackedInt::from_integer(&Integer(limbs), Global);

            for step in 0..4 {
                let carried_serial = serial.step();
                let carried_threaded = engine.step(&mut threaded);
                assert_eq!(carried_serial, carried_threaded);
                assert!(
                    threaded.to_integer(Global) == serial.to_integer(Global),
                    "value diverged: {num_threads} threads, {num_limbs} limbs, step {step}"
                );
                assert_clean_padding(
                    &threaded,
                    &format!("all-nines, {num_threads} threads, {num_limbs} limbs, step {step}"),
                );
            }
        }
    }
}

/// The engine trajectory from 196 must match the serial kernel's.
#[test]
fn test_packed_engine_196_trajectory() {
    let mut engine = PackedEngine::new(4);
    let mut serial = crate::integer!("196");
    let mut threaded = PackedInt::from_integer(&serial, Global);
    for step in 0..1000 {
        serial.fused_reverse_add_asm_interleave();
        engine.step(&mut threaded);
        assert!(
            threaded.to_integer(Global) == serial,
            "value diverged at step {step}"
        );
    }
}

/// Palindromes are detected exactly when every digit equals its mirror.
#[test]
fn test_packed_palindrome() {
    for digits in ["5", "44", "121", "123454321", "1230321"] {
        let integer: Integer<Global> = crate::integer!(digits);
        let packed = PackedInt::from_integer(&integer, Global);
        let expected = {
            let fwd: Vec<char> = digits.chars().collect();
            let rev: Vec<char> = digits.chars().rev().collect();
            fwd == rev
        };
        assert_eq!(packed.is_palindrome(), expected, "{digits}");
    }
}
