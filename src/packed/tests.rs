use super::{PackedInt, digit_at};
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

/// Round trip and the mirror invariant rev[d] == a[digits - 1 - d].
#[test]
fn test_packed_round_trip_and_mirror() {
    let mut rng = SmallRng::seed_from_u64(0x196);
    for num_limbs in [1usize, 2, 3, 5, 8, 17, 33] {
        let integer = random_integer(num_limbs, &mut rng);
        let packed = PackedInt::from_integer(&integer, Global);
        assert!(packed.to_integer(Global) == integer, "{num_limbs} limbs");

        let digits = packed.digits;
        for d in 0..digits {
            assert_eq!(
                digit_at(packed.rev_cur(), d),
                digit_at(&packed.a, digits - 1 - d),
                "mirror broken at slot {d} of {digits}"
            );
        }
        for d in digits..packed.rev_cur().len() * super::DPL {
            assert_eq!(digit_at(packed.rev_cur(), d), 0, "dirty rev padding");
        }
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

/// Palindromes are detected exactly when the copies coincide.
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
