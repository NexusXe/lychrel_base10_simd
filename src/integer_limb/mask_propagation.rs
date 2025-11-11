// ===============================
// Closure table computation in Rust
// ===============================
//
// Computes the steady-state of a |= b & (a << 1)
// for all 8-bit combinations of a,b at compile time.
//
// Each table entry stores:
//  - out0, out1 : final byte if incoming==0 or 1
//  - cout0, cout1 : carry-out bit (bool) for each incoming

#[derive(Clone, Copy)]
struct Entry {
    out0: u8,
    out1: u8,
    cout0: bool,
    cout1: bool,
}

/// Compute one step of the propagation for 8-bit a,b
#[inline(always)]
const fn step(a: u8, b: u8) -> u8 {
    a | (b & (a << 1))
}

/// Compute the steady-state closure for 8-bit a,b and an incoming carry bit.
/// This runs ≤8 iterations until stable.
#[inline(always)]
const fn closure8(mut a: u8, b: u8, carry_in: bool) -> (u8, bool) {
    if carry_in {
        // if incoming 1 from the right, treat a[-1] = 1 => shift in an extra 1 at bit 0
        a |= b & 0x01; // only bit0 can be affected by external carry
    }

    let mut prev = 0u8;
    let mut i = 0;
    while i < 8 && a != prev {
        prev = a;
        a = step(a, b);
        i += 1;
    }

    // carry_out if leftmost bit becomes 1 and b's leftmost bit = 1
    let carry_out = ((a & 0x80) != 0) && ((b & 0x80) != 0);
    (a, carry_out)
}

/// Build the table entry for one pair (a,b)
#[inline(always)]
const fn make_entry(a: u8, b: u8) -> Entry {
    let (out0, cout0) = closure8(a, b, false);
    let (out1, cout1) = closure8(a, b, true);
    Entry { out0, out1, cout0, cout1 }
}

/// Build the entire 65536-entry table at compile time.
const fn build_table() -> [Entry; 65536] {
    let mut table = [Entry { out0: 0, out1: 0, cout0: false, cout1: false }; 65536];
    let mut i = 0;
    while i < 256 {
        let mut j = 0;
        while j < 256 {
            let idx = ((j as usize) << 8) | (i as usize);
            table[idx] = make_entry(i as u8, j as u8);
            j += 1;
        }
        i += 1;
    }
    table
}

const T: [Entry; 65536] = build_table();

#[inline(always)]
pub const fn closure64(a: u64, b: u64) -> u64 {
    // Split into 8 bytes, look up table
    let mut out: [u8; 8] = [0; 8];
    let mut carry_in = false;
    let mut i: usize = 0;
    while i < 8 {
        let ai = ((a >> (8 * i)) & 0xFF) as u8;
        let bi = ((b >> (8 * i)) & 0xFF) as u8;
        let entry = T[((bi as usize) << 8) | (ai as usize)];
        let (out_byte, carry_out) = if carry_in {
            (entry.out1, entry.cout1)
        } else {
            (entry.out0, entry.cout0)
        };
        out[i] = out_byte;
        carry_in = carry_out;
        i += 1;
    }
    // Combine bytes back into u64
    let mut r = 0u64;
    let mut i: usize = 0;
    while i < 8 {
        r |= (out[i] as u64) << (8 * i);
        i += 1;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_behavior() {
        let a = 0b0000_0100u64;
        let b = 0b0011_1000u64;
        let res = closure64(a, b);
        assert_eq!(res, 0b0011_1100);
    }

    #[test]
    fn stability() {
        for a in 0u64..=0xFF {
            for b in 0u64..=0xFF {
                let mut x = a;
                for _ in 0..64 {
                    x |= b & (x << 1);
                }
                assert_eq!(x, closure64(a, b));
            }
        }
    }
}
