//! The dual nibble-packed representation: the number and its digit-reversal
//! are both kept packed (two digits per byte, the checkpoint format), which
//! halves the DRAM traffic of an iteration. The reverse-and-add pass reads
//! the two copies slot-aligned with no skip offset, resolves carries in one
//! ascending sweep, and produces both next-iteration copies.
//!
//! Packed line layout (one 64-byte `Limb` holding 128 digits): digit `p` of
//! the line lives in byte `p` low nibble for `p < 64`, byte `p - 64` high
//! nibble for `p >= 64`. Lines are LSD-first like the byte-per-digit form,
//! and a packed line is exactly `Limb::pack` of two adjacent unpacked limbs.

use crate::impossible;
use crate::integer_limb::{Integer, LV_LEN, Limb, LimbVec, resolve_digits};
use std::alloc::Allocator;
use std::hint::likely;
use std::simd::prelude::*;

/// Digits per packed line.
pub(crate) const DPL: usize = 2 * LV_LEN;

const LO_MASK: LimbVec = LimbVec::splat(0x0F);

/// The two 64-digit halves of a packed line, low digits first.
#[inline(always)]
fn unpack_line(v: LimbVec) -> (LimbVec, LimbVec) {
    (v & LO_MASK, v >> 4)
}

/// Packs two vectors of clean digits (0..=9) into one line.
#[inline(always)]
fn pack_line(lo: LimbVec, hi: LimbVec) -> LimbVec {
    lo | (hi << 4)
}

/// Adds two packed lines digit-wise and resolves every decimal carry inside
/// the line, with `carry` into the line's lowest digit. Returns the packed
/// result, the carry out of the top digit, and whether any digit carried.
#[inline(always)]
fn add_resolve_line(a: LimbVec, r: LimbVec, carry: bool) -> (LimbVec, bool, bool) {
    let (a_lo, a_hi) = unpack_line(a);
    let (r_lo, r_hi) = unpack_line(r);

    let (out_lo, carry_mid, carried_lo) = resolve_digits(a_lo + r_lo, carry);
    let (out_hi, carry_out, carried_hi) = resolve_digits(a_hi + r_hi, carry_mid);

    (
        pack_line(out_lo, out_hi),
        carry_out,
        carried_lo || carried_hi,
    )
}

/// The digit at slot `d` of a packed line slice.
#[inline]
pub(crate) fn digit_at(lines: &[Limb], d: usize) -> u8 {
    let line = lines[d / DPL].0;
    let p = d % DPL;
    if p < LV_LEN {
        line[p] & 0x0F
    } else {
        line[p - LV_LEN] >> 4
    }
}

/// Overwrites the digit at slot `d` of a packed line slice.
#[inline]
pub(crate) fn set_digit(lines: &mut [Limb], d: usize, digit: u8) {
    debug_assert!(digit <= 9);
    let line = &mut lines[d / DPL].0;
    let p = d % DPL;
    if p < LV_LEN {
        line[p] = (line[p] & 0xF0) | digit;
    } else {
        line[p - LV_LEN] = (line[p - LV_LEN] & 0x0F) | (digit << 4);
    }
}

/// The 128 digits of packed line `m` as ascending bytes; lines outside
/// `0..lines.len()` read as zeros (the virtual padding below digit 0 and
/// above the top line).
#[inline(always)]
fn line_digit_bytes(lines: &[Limb], m: isize) -> [u8; DPL] {
    let mut out = [0u8; DPL];
    if m >= 0 && (m as usize) < lines.len() {
        let (lo, hi) = unpack_line(lines[m as usize].0);
        out[..LV_LEN].copy_from_slice(lo.as_array());
        out[LV_LEN..].copy_from_slice(hi.as_array());
    }
    out
}

/// Rebuilds `dst` as the digit-reversal of `lines` (`digits` significant
/// digits): dst slot `s` holds digit `digits - 1 - s`. Slots at and above
/// `digits` in the top line are zero.
pub(crate) fn mirror_into<T: Allocator + Clone + Copy>(
    lines: &[Limb],
    digits: usize,
    dst: &mut Vec<Limb, T>,
) {
    let lines_out = digits.div_ceil(DPL);
    dst.clear();
    dst.reserve(lines_out);

    // dst line l covers slots [DPL*l, DPL*(l+1)) = source digits
    // [digits - DPL*(l+1), digits - DPL*l) in descending order, so the source
    // window walks the array downward at the fixed intra-line offset
    // digits % DPL. The window's two source lines roll: the lower line of
    // step l is the upper line of step l + 1.
    let mut upper_digits = line_digit_bytes(lines, (digits as isize - 1).div_euclid(DPL as isize));
    for l in 0..lines_out {
        let w = digits as isize - (DPL * (l + 1)) as isize;
        let m1 = w.div_euclid(DPL as isize);
        let phi = w.rem_euclid(DPL as isize) as usize;

        let lower_digits = line_digit_bytes(lines, m1);
        let mut buf = [0u8; 2 * DPL];
        buf[..DPL].copy_from_slice(&lower_digits);
        buf[DPL..].copy_from_slice(&upper_digits);

        // window = source digits [w, w + DPL) ascending; the dst line wants
        // them descending, so its low half is the reversed upper window half.
        let window = &buf[phi..phi + DPL];
        let lo = LimbVec::from_slice(&window[LV_LEN..]).reverse();
        let hi = LimbVec::from_slice(&window[..LV_LEN]).reverse();
        dst.push(Limb(pack_line(lo, hi)));

        upper_digits = lower_digits;
    }
}

/// A number held as two packed copies: `a` is the value LSD-first, and
/// `rev[cur]` is its digit-reversal, slot-aligned so that
/// `rev[d] == a[digits - 1 - d]`. The other rev buffer is the write target
/// of the next iteration.
pub struct PackedInt<T: Allocator + Clone + Copy> {
    pub(crate) a: Vec<Limb, T>,
    rev: [Vec<Limb, T>; 2],
    cur: usize,
    pub(crate) digits: usize,
}

impl<T: Allocator + Clone + Copy> PackedInt<T> {
    /// Builds the dual representation from a byte-per-digit integer.
    pub fn from_integer(integer: &Integer<T>, allocator: T) -> Self {
        if integer.0.is_empty() {
            impossible!("Tried to pack an empty integer");
        }
        let digits = integer.len() as usize;
        let mut a = Vec::with_capacity_in(integer.0.len().div_ceil(2), allocator);
        for pair in integer.0.chunks(2) {
            let hi = if pair.len() == 2 { pair[1].0 } else { LimbVec::splat(0) };
            a.push(Limb(pack_line(pair[0].0, hi)));
        }
        let mut rev0 = Vec::new_in(allocator);
        mirror_into(&a, digits, &mut rev0);
        let rev1 = rev0.clone();
        Self {
            a,
            rev: [rev0, rev1],
            cur: 0,
            digits,
        }
    }

    /// The value as a byte-per-digit integer (for reports and checkpoints).
    pub fn to_integer(&self, allocator: T) -> Integer<T> {
        let mut out = Vec::with_capacity_in(self.a.len() * 2, allocator);
        let limbs = self.digits.div_ceil(LV_LEN);
        for line in &self.a {
            let (lo, hi) = unpack_line(line.0);
            out.push(Limb(lo));
            out.push(Limb(hi));
        }
        out.truncate(limbs);
        Integer(out)
    }

    #[inline]
    pub(crate) fn rev_cur(&self) -> &[Limb] {
        &self.rev[self.cur]
    }

    /// Whether the value is a palindrome: the two copies are slot-identical.
    #[inline]
    pub fn is_palindrome(&self) -> bool {
        let lines = self.digits.div_ceil(DPL);
        self.a[..lines] == self.rev[self.cur][..lines]
    }

    /// Whether the next reverse-and-add gains a digit, decided exactly before
    /// the pass: descending from the top, skip digit sums equal to nine (they
    /// propagate whatever comes from below); the first other sum decides.
    /// All-nines sums generate no carry at all and do not grow.
    pub(crate) fn prescan_grow(&self) -> bool {
        let l = self.digits;
        for d in (0..l).rev() {
            let s = digit_at(&self.a, d) + digit_at(&self.a, l - 1 - d);
            if s != 9 {
                return s > 9;
            }
        }
        false
    }

    /// One reverse-and-add step, single-threaded: an ascending fused add over
    /// the two copies with an exact running carry, then the mirror rebuild
    /// into the spare rev buffer. Returns whether any digit carried.
    pub fn step(&mut self) -> bool {
        let l = self.digits;
        let grew = self.prescan_grow();
        let lp = l + grew as usize;
        let lines = l.div_ceil(DPL);

        let mut carry = false;
        let mut any_carried = false;
        for k in 0..lines {
            let (out, c_out, c_any) =
                add_resolve_line(self.a[k].0, self.rev[self.cur][k].0, carry);
            self.a[k] = Limb(out);
            carry = c_out;
            any_carried |= c_any;
        }
        if carry {
            // only reachable when the top line was full to the brim
            let mut top = LimbVec::splat(0);
            top[0] = 1;
            self.a.push(Limb(top));
        }

        debug_assert_eq!(lp.div_ceil(DPL), self.a.len().min(lp.div_ceil(DPL)));
        debug_assert!(digit_at(&self.a, lp - 1) != 0, "prescan missed growth");
        debug_assert!(
            lp % DPL == 0 || self.a.len() * DPL <= lp + DPL,
            "prescan over-grew"
        );
        if lp < self.a.len() * DPL {
            debug_assert_eq!(
                (lp..(lines.max(lp.div_ceil(DPL)) * DPL).min(self.a.len() * DPL))
                    .map(|d| digit_at(&self.a, d))
                    .max()
                    .unwrap_or(0),
                0,
                "dirty padding above the top digit"
            );
        }

        self.digits = lp;
        let (a, rev) = (&self.a, &mut self.rev);
        let next = 1 - self.cur;
        mirror_into(a, lp, &mut rev[next]);
        self.cur = next;

        likely(any_carried)
    }
}

#[cfg(test)]
mod tests;
