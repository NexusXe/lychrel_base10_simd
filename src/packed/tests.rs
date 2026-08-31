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

/// The fused double step must agree with two serial packed steps -- value,
/// digit count, second-step carry flag, and the mid-value palindrome flag
/// -- across thread counts and sizes that produce empty blocks, multi-chunk
/// blocks, and growth.
#[test]
fn test_packed_step2_matches_serial() {
    let mut rng = SmallRng::seed_from_u64(0x196);

    for num_threads in [1, 2, 3, 8] {
        let mut engine = PackedEngine::new(num_threads);
        for num_limbs in [1usize, 2, 3, 4, 5, 7, 16, 33, 64, 100] {
            let start = random_integer(num_limbs, &mut rng);
            let mut serial = PackedInt::from_integer(&start, Global);
            let mut fused = PackedInt::from_integer(&start, Global);

            for step in 0..30 {
                let carried_mid = serial.step();
                let mid_pal = !carried_mid && serial.is_palindrome();
                let carried_serial = serial.step();
                let r = engine.step2(&mut fused);
                let context =
                    format!("{num_threads} threads, {num_limbs} limbs, double step {step}");
                assert_eq!(carried_serial, r.carried, "carry flag diverged: {context}");
                assert_eq!(mid_pal, r.palindrome_mid, "mid palindrome diverged: {context}");
                assert_eq!(serial.digits, fused.digits, "digit count diverged: {context}");
                assert!(
                    fused.to_integer(Global) == serial.to_integer(Global),
                    "value diverged: {context}"
                );
                assert_clean_padding(&fused, &context);
            }
        }
    }
}

/// All-nines inputs push a carry through every line of both fused steps.
#[test]
fn test_packed_step2_all_nines() {
    for num_threads in [2, 5] {
        let mut engine = PackedEngine::new(num_threads);
        for num_limbs in [1usize, 3, 8, 21] {
            let limbs = vec![Limb(LimbVec::splat(9)); num_limbs];
            let mut serial = PackedInt::from_integer(&Integer(limbs.clone()), Global);
            let mut fused = PackedInt::from_integer(&Integer(limbs), Global);

            for step in 0..4 {
                serial.step();
                let carried_serial = serial.step();
                let r = engine.step2(&mut fused);
                let context = format!("all-nines, {num_threads} threads, {num_limbs} limbs, double step {step}");
                assert_eq!(carried_serial, r.carried, "{context}");
                assert!(
                    fused.to_integer(Global) == serial.to_integer(Global),
                    "value diverged: {context}"
                );
                assert_clean_padding(&fused, &context);
            }
        }
    }
}

/// The fused trajectory from 196 must match the serial kernel's.
#[test]
fn test_packed_step2_196_trajectory() {
    let mut engine = PackedEngine::new(4);
    let mut serial = crate::integer!("196");
    let mut fused = PackedInt::from_integer(&serial, Global);
    for step in 0..500 {
        serial.fused_reverse_add_asm_interleave();
        serial.fused_reverse_add_asm_interleave();
        engine.step2(&mut fused);
        assert!(
            fused.to_integer(Global) == serial,
            "value diverged at double step {step}"
        );
    }
}

/// A fused step whose intermediate value is a palindrome must say so: 12
/// reverse-adds to 33 with no carry.
#[test]
fn test_packed_step2_mid_palindrome() {
    let mut engine = PackedEngine::new(2);
    for (seed, expect) in [("12", true), ("10", true), ("196", false)] {
        let integer: Integer<Global> = crate::integer!(seed);
        let mut packed = PackedInt::from_integer(&integer, Global);
        let r = engine.step2(&mut packed);
        assert_eq!(r.palindrome_mid, expect, "{seed}");
    }
}

/// The fused triple step must agree with three serial packed steps --
/// value, digit count, third-step carry flag, and both mid-value
/// palindrome flags -- across thread counts and sizes that produce empty
/// blocks, multi-chunk blocks, and growth.
#[test]
fn test_packed_step3_matches_serial() {
    let mut rng = SmallRng::seed_from_u64(0x196);

    for num_threads in [1, 2, 3, 8] {
        let mut engine = PackedEngine::new(num_threads);
        for num_limbs in [1usize, 2, 3, 4, 5, 7, 16, 33, 64, 100] {
            let start = random_integer(num_limbs, &mut rng);
            let mut serial = PackedInt::from_integer(&start, Global);
            let mut fused = PackedInt::from_integer(&start, Global);

            for step in 0..30 {
                let carried_mid1 = serial.step();
                let mid1_pal = !carried_mid1 && serial.is_palindrome();
                let carried_mid2 = serial.step();
                let mid2_pal = !carried_mid2 && serial.is_palindrome();
                let carried_serial = serial.step();
                let r = engine.step3(&mut fused);
                let context =
                    format!("{num_threads} threads, {num_limbs} limbs, triple step {step}");
                assert_eq!(carried_serial, r.carried, "carry flag diverged: {context}");
                assert_eq!(mid1_pal, r.palindrome_mid1, "mid1 palindrome diverged: {context}");
                assert_eq!(mid2_pal, r.palindrome_mid2, "mid2 palindrome diverged: {context}");
                assert_eq!(serial.digits, fused.digits, "digit count diverged: {context}");
                assert!(
                    fused.to_integer(Global) == serial.to_integer(Global),
                    "value diverged: {context}"
                );
                assert_clean_padding(&fused, &context);
            }
        }
    }
}

/// All-nines inputs push a carry through every line of all three fused
/// steps.
#[test]
fn test_packed_step3_all_nines() {
    for num_threads in [2, 5] {
        let mut engine = PackedEngine::new(num_threads);
        for num_limbs in [1usize, 3, 8, 21] {
            let limbs = vec![Limb(LimbVec::splat(9)); num_limbs];
            let mut serial = PackedInt::from_integer(&Integer(limbs.clone()), Global);
            let mut fused = PackedInt::from_integer(&Integer(limbs), Global);

            for step in 0..4 {
                serial.step();
                serial.step();
                let carried_serial = serial.step();
                let r = engine.step3(&mut fused);
                let context = format!("all-nines, {num_threads} threads, {num_limbs} limbs, triple step {step}");
                assert_eq!(carried_serial, r.carried, "{context}");
                assert!(
                    fused.to_integer(Global) == serial.to_integer(Global),
                    "value diverged: {context}"
                );
                assert_clean_padding(&fused, &context);
            }
        }
    }
}

/// The fused triple trajectory from 196 must match the serial kernel's.
#[test]
fn test_packed_step3_196_trajectory() {
    let mut engine = PackedEngine::new(4);
    let mut serial = crate::integer!("196");
    let mut fused = PackedInt::from_integer(&serial, Global);
    for step in 0..333 {
        serial.fused_reverse_add_asm_interleave();
        serial.fused_reverse_add_asm_interleave();
        serial.fused_reverse_add_asm_interleave();
        engine.step3(&mut fused);
        assert!(
            fused.to_integer(Global) == serial,
            "value diverged at triple step {step}"
        );
    }
}

/// A triple step whose intermediate values are palindromes must say so:
/// 12 reverse-adds to the palindrome 33 in one carry-free step, and 5
/// reaches the palindrome 11 on its second step (10 + 01, carry-free)
/// after a first step that carried.
#[test]
fn test_packed_step3_mid_palindromes() {
    let mut engine = PackedEngine::new(2);
    for (seed, expect1, expect2) in [
        ("12", true, true),
        ("5", false, true),
        ("196", false, false),
    ] {
        let integer: Integer<Global> = crate::integer!(seed);
        let mut packed = PackedInt::from_integer(&integer, Global);
        let r = engine.step3(&mut packed);
        assert_eq!(r.palindrome_mid1, expect1, "{seed} mid1");
        assert_eq!(r.palindrome_mid2, expect2, "{seed} mid2");
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
