use super::*;
use rand::prelude::*;

#[test]
fn test_len_empty() {
    let limb = Limb::new();
    assert_eq!(limb.len(), 0);
}

#[test]
fn test_len_one_digit() {
    let limb = Limb::new_from_value(5);
    assert_eq!(limb.len(), 1);
}

#[test]
fn test_len_multiple_digits() {
    let limb = Limb::new_from_value(12345);
    assert_eq!(limb.len(), 5);
}

#[test]
fn test_len_full() {
    let limb = Limb::new_from_value(u128::from(u64::MAX));
    assert_eq!(limb.len(), 20);

    let limb = Limb::new_from_value(u128::MAX);
    assert_eq!(limb.len(), 39);
}

#[test]
fn test_reverse() {
    let limb1 = Limb(u8x64::from([
        9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
        9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
        9, 9, 9, 9,
    ]));
    let limb2 = Limb(u8x64::from([
        1, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0,
        0, 6, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0,
        0, 0, 0, 0,
    ]));

    let integer: Integer = Integer(vec![limb1, limb2]);
    assert_eq!(
        format!("{integer}"),
        "900000008000000070000000600000005000000040000000300020019999999999999999999999999999999999999999999999999999999999999999"
    );
    let mut reversed_integer = Integer(Vec::new());
    integer.reverse_into_integer(&mut reversed_integer);
    assert_eq!(
        format!("{reversed_integer}"),
        "999999999999999999999999999999999999999999999999999999999999999910020003000000040000000500000006000000070000000800000009"
    );

    let limb1 = Limb(u8x64::from([
        0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6,
        7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4,
        5, 6, 7, 8,
    ]));

    let limb2 = Limb(limb1.0.reverse());
    let integer: Integer = Integer(vec![limb1, limb2]);
    assert_eq!(
        format!("{integer}"),
        "123456780000000012345678000000001234567800000000123456788765432100000000876543210000000087654321000000008765432100000000"
    );
    let mut reversed_integer = Integer(Vec::new());
    integer.reverse_into_integer(&mut reversed_integer);
    assert_eq!(
        format!("{reversed_integer}"),
        "1234567800000000123456780000000012345678000000001234567887654321000000008765432100000000876543210000000087654321"
    );

    let val = "12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890";
    let val_reversed = val.chars().rev().collect::<String>();

    let integer = integer!(val);
    let expected_reverse = integer!(val_reversed.as_str());
    let mut reversed_integer = Integer(Vec::new());
    integer.reverse_into_integer(&mut reversed_integer);
    assert_eq!(reversed_integer, expected_reverse);

    let val = "1234567812345678123456781234567812345678123456781234567812345678";
    let val_reversed = val.chars().rev().collect::<String>();

    let integer = integer!(val);
    let expected_reverse = integer!(val_reversed.as_str());
    let mut reversed_integer = Integer(Vec::new());
    integer.reverse_into_integer(&mut reversed_integer);
    assert_eq!(reversed_integer, expected_reverse);

    let val = "12345678123456781234567812345678123456781234567812345678123456781";
    let val_reversed = val.chars().rev().collect::<String>();

    let integer = integer!(val);
    let expected_reverse = integer!(val_reversed.as_str());
    let mut reversed_integer = Integer(Vec::new());
    integer.reverse_into_integer(&mut reversed_integer);
    assert_eq!(reversed_integer, expected_reverse);
}

#[test]
fn test_integer_add() {
    let integer1: Integer = Integer(vec![Limb(u8x64::from([
        9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
        9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
        9, 9, 9, 9,
    ]))]);
    let integer2: Integer = Integer(vec![Limb(u8x64::from([
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ]))]);

    let result = integer1 + integer2;
    assert_eq!(
        result,
        (
            Integer(vec![
                Limb(u8x64::splat(0)),
                Limb(u8x64::from([
                    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ]))
            ]),
            true
        )
    );

    let limb1 = integer!("7");
    let limb2 = integer!("1");
    let result = limb1 + limb2;
    assert_eq!(result, (integer!("8"), false));
}

#[test]
fn test_limb_len() {
    let limb = Limb::new();
    assert_eq!(limb.len(), 0);

    let limb = Limb::new_from_value(1);
    assert_eq!(limb.len(), 1);

    let limb = Limb::new_from_value(12345);
    assert_eq!(limb.len(), 5);

    let limb = Limb::new_from_value(100_000_000_000_000_000_000_000_000_000_000_000_000); // 39 digits
    assert_eq!(limb.len(), 39);

    let limb = Limb::new_from_value(340_282_366_920_938_463_463_374_607_431_768_211_455); // 39 digits
    assert_eq!(limb.len(), 39);

    for i in 1..=9 {
        let limb = Limb(u8x64::splat(i));
        assert_eq!(limb.len(), 64);
    }
}

#[test]
fn test_integer_macro() {
    let integer1 = integer!("1234567890123456789012345678901234567890123456789012345678901234");
    assert_eq!(
        format!("{integer1}"),
        "1234567890123456789012345678901234567890123456789012345678901234"
    );
    let integer2 =
        integer!("1234567890123456789012345678901234567890123456789012345678901234567800000000");
    assert_eq!(
        format!("{integer2}"),
        "1234567890123456789012345678901234567890123456789012345678901234567800000000"
    );
}

#[test]
fn test_integer_len() {
    let integer1 = integer!("12345678");
    assert_eq!(integer1.len(), 8);
    let integer2 = integer!("1234567890123456789012345678901234567890123456789012345678901234");
    assert_eq!(integer2.len(), 64);
    let integer3 =
        integer!("1234567890123456789012345678901234567890123456789012345678901234567800000000");
    assert_eq!(integer3.len(), 76);
}

#[test]
fn test_pack_unpack_limb() {
    let limb1: Limb = u8x64::splat(9).into();
    let limb2: Limb = u8x64::splat(1).into();
    let packed = limb1.pack(limb2);
    assert_eq!(packed, u8x64::splat(0x19).into());

    let packed: Limb = u8x64::splat(0x19).into();
    let (limb1, limb2) = packed.unpack();
    assert_eq!(limb1, u8x64::splat(9).into());
    assert_eq!(limb2, u8x64::splat(1).into());
}

#[test]
fn test_pack_unpack_limb_random() {
    fn test_with_seed(seed: u64) {
        // get two deterministically random limbs from a SmallRng with a constant seed
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut arr1: [u8; 64] = [0; 64];
        let mut arr2: [u8; 64] = [0; 64];

        for (item1, item2) in arr1.iter_mut().zip(arr2.iter_mut()) {
            *item1 = rng.random_range(0..10);
            *item2 = rng.random_range(0..10);
        }
        let limb1 = Limb(u8x64::from(arr1));
        let limb2 = Limb(u8x64::from(arr2));

        let integer = Integer(vec![limb1, limb2]);
        let packed = integer.clone().pack();
        assert_eq!(packed.clone().unpack(), integer);

        let bytes: Vec<u8> = packed.clone().into_bytes();
        let organized_bytes: Vec<[u8; 64]> = bytes
            .chunks(64)
            .map(|chunk| chunk.try_into().unwrap())
            .collect();

        let reconstructed = Integer::from_bytes(organized_bytes);
        assert_eq!(reconstructed, packed);
        assert_eq!(reconstructed.unpack(), integer);
    }

    test_with_seed(0xdead_beef);
    test_with_seed(0xbaad_f00d);
}

#[test]
fn test_pack_unpack_integer() {
    let limb1: Limb = u8x64::splat(9).into();
    let limb2: Limb = u8x64::splat(1).into();
    let integer: Integer = Integer(vec![limb1, limb2]);
    let packed = integer.pack();
    assert_eq!(packed, Integer(vec![u8x64::splat(0x19).into()]));

    let packed = Integer(vec![u8x64::splat(0x19).into()]);
    let unpacked = packed.unpack();
    assert_eq!(unpacked, Integer(vec![limb1, limb2]));

    let packed = Integer(vec![u8x64::splat(0x19).into(); 4]);
    let unpacked = packed.unpack();
    assert_eq!(
        unpacked,
        Integer(vec![limb1, limb2, limb1, limb2, limb1, limb2, limb1, limb2])
    );
}

#[test]
fn test_write_and_read_checkpoint() {
    use std::io::Write;

    const LEN: usize = 13;
    let mut rng = SmallRng::seed_from_u64(0xABCD_EF01_2345_6789);
    let mut arrs: Vec<[u8; 64]> = Vec::with_capacity(LEN);
    for _ in 0..LEN {
        let mut arr: [u8; 64] = [0; 64];
        for item in &mut arr {
            *item = rng.random_range(0..10);
        }
        arrs.push(arr);
    }

    let last_arr = arrs.last_mut().unwrap();

    for item in last_arr.iter_mut().take(64).skip(32) {
        *item = 0;
    }

    let mut limbs: Vec<Limb> = Vec::with_capacity(LEN);
    for arr in arrs {
        limbs.push(Limb(u8x64::from(arr)));
    }

    let integer = Integer(limbs);
    assert!(!integer.has_carries());
    assert_eq!(integer, integer.clone().pack().unpack());

    // instead of writing to an actual file, write to a buffer
    let mut buffer = Vec::new();

    let data_to_write = integer.clone().pack().into_bytes();

    buffer.write_all(&data_to_write).unwrap();

    // now read it as if it was from a file

    let mut checkpoint_read: Vec<u8> = Vec::new();
    checkpoint_read.write_all(&buffer).unwrap();
    let organized_bytes: Vec<[u8; 64]> = checkpoint_read
        .chunks(64)
        .map(|chunk| chunk.try_into().unwrap())
        .collect();
    let integer_read = Integer::from_bytes(organized_bytes);

    //assert_eq!(integer, integer_read.clone());
    assert_eq!(integer, integer_read.clone().unpack());
    assert!(!integer_read.unpack().has_carries());
}

#[test]
fn test_packedlimb_packing() {
    let limb1: Limb = u8x64::splat(1).into();
    let limb2: Limb = u8x64::splat(2).into();

    let packed_limb: PackedLimb = PackedLimb::from((limb1, limb2));
    assert_eq!(packed_limb.0, u8x64::splat(0x21));
}

#[test]
fn test_packedlimb_unpacking() {
    let packed_limb: PackedLimb = u8x64::splat(0x21).into();
    let (limb1, limb2): LimbPair = packed_limb.into();
    assert_eq!(limb1, u8x64::splat(1).into());
    assert_eq!(limb2, u8x64::splat(2).into());
}

#[test]
fn test_packedlimb_len() {
    use std::num::NonZeroU32;
    let packed_limb: PackedLimb = u8x64::splat(0x21).into();
    assert_eq!(packed_limb.len(), NonZeroU32::new(64).unwrap());

    let integer = integer!("123456789012345678901234567890");
    let packed_integer: PackedInteger = integer.into();

    let last_limb_len = packed_integer.0.last().unwrap().len();
    assert_eq!(last_limb_len, NonZeroU32::new(30).unwrap());

    let integer = integer!("1234567890123456789012345678901212345678901234567890123456789012");
    let packed_integer: PackedInteger = integer.into();
    assert_eq!(
        packed_integer.0.last().unwrap().len(),
        NonZeroU32::new(64).unwrap()
    );

    let integer = integer!(
        "12345678901234567890123456789012123456789012345678901234567890121234567890123456789012345678901212345678901234567890123456789012"
    );
    let packed_integer: PackedInteger = integer.into();
    assert_eq!(
        packed_integer.0.last().unwrap().len(),
        NonZeroU32::new(64).unwrap()
    );

    let integer = integer!(
        "1234567890123456789012345678901212345678901234567890123456789012123456789012345678901234567890121234567890123456789012345678901"
    );
    let packed_integer: PackedInteger = integer.into();
    assert_eq!(
        packed_integer.0.last().unwrap().len(),
        NonZeroU32::new(63).unwrap()
    );

    let integer = integer!("12345678901234567890123456789012123456789012345678901234567890123");
    let packed_integer: PackedInteger = integer.into();
    assert_eq!(
        packed_integer.0.last().unwrap().len(),
        NonZeroU32::new(1).unwrap()
    );
}

#[test]
fn test_reverse_interleave_limbs() {
    let mut a: u8x64 = [1u8; 64].into();
    let mut b: u8x64 = [2u8; 64].into();

    Integer::reverse_interleave_x2(&mut a, &mut b);

    assert_eq!(a, [0x21u8; 64].into());
    assert_eq!(b, [0x12u8; 64].into());

    let mut a: u8x64 = [9u8; 64].into();
    let mut b: u8x64 = [0u8; 64].into();

    Integer::reverse_interleave_x2(&mut a, &mut b);

    assert_eq!(a, [0x09u8; 64].into());
    assert_eq!(b, [0x90u8; 64].into());
}

// #[test]
// fn test_reverse_add() {
//     let mut integer: Integer = Integer(vec![Limb(u8x64::splat(1)), Limb(u8x64::splat(2))]);

//     integer.fused_reverse_add_packed_asm();
// }

#[test]
fn test_fused_reverse_add_asm_simple_interleave() {
    let mut integer = integer!("196");
    let ever_carried = integer.fused_reverse_add_asm_interleave();
    assert_eq!(integer, integer!("887"));
    assert!(ever_carried);
}

#[test]
fn test_fused_reverse_add_asm_simple_interleave_2limb_partial() {
    let mut integer =
        integer!("1960000000000000000000000000000000000000000000000000000000000000196");
    let ever_carried = integer.fused_reverse_add_asm_interleave();
    assert_eq!(
        integer,
        integer!("8870000000000000000000000000000000000000000000000000000000000000887")
    );
    assert!(ever_carried);
}

#[test]
fn test_fused_reverse_add_asm_simple_interleave_2limb_full() {
    let mut integer = integer!("1212121212121212121212121212121212121212121212121212121212121212");
    let ever_carried = integer.fused_reverse_add_asm_interleave();
    assert_eq!(
        integer,
        integer!("3333333333333333333333333333333333333333333333333333333333333333")
    );
    assert!(!ever_carried);
}

#[test]
fn test_fused_reverse_add_asm_simple_interleave_wide() {
    let mut integer = integer!("196000000000000000000000000000000000000000000000000000196");
    let ever_carried = integer.fused_reverse_add_asm_interleave();
    assert_eq!(
        integer,
        integer!("887000000000000000000000000000000000000000000000000000887")
    );
    assert!(ever_carried);
}

#[test]
fn test_fused_reverse_add_asm_simple_interleave_2() {
    let mut integer = Integer(vec![Limb({
        let mut arr = [0u8; 64];
        arr[0] = 5;
        arr[63] = 6;
        u8x64::from(arr)
    })]);

    let ever_carried = integer.fused_reverse_add_asm_interleave();
    assert_eq!(
        integer,
        integer!("11000000000000000000000000000000000000000000000000000000000000011")
    );
    assert!(ever_carried);
}

#[test]
fn test_fused_reverse_add_asm_interleave() {
    let mut integer1 = integer!("12345678");
    let ever_carried = integer1.fused_reverse_add_asm_interleave();
    assert_eq!(integer1, integer!("99999999"));
    assert!(!ever_carried);

    let mut integer1 = integer!("1234567812345678");
    let ever_carried = integer1.fused_reverse_add_asm_interleave();
    assert_eq!(integer1, integer!("9999999999999999"));
    assert!(!ever_carried);

    let mut integer = integer!("99999999");
    let ever_carried = integer.fused_reverse_add_asm_interleave();
    assert_eq!(integer, integer!("199999998"));
    assert!(ever_carried);

    let mut integer = integer!("12121212");
    let ever_carried = integer.fused_reverse_add_asm_interleave();
    assert_eq!(integer, integer!("33333333"));
    assert!(!ever_carried);

    let mut integer = integer!("1212121212121212");
    let ever_carried = integer.fused_reverse_add_asm_interleave();
    assert_eq!(integer, integer!("3333333333333333"));
    assert!(!ever_carried);

    let mut integer = integer!("12121212121212121212121212121212");
    let ever_carried = integer.fused_reverse_add_asm_interleave();
    assert_eq!(integer, integer!("33333333333333333333333333333333"));
    assert!(!ever_carried);

    let mut integer: Integer = Integer(vec![Limb(u8x64::splat(9))]);
    let ever_carried = integer.fused_reverse_add_asm_interleave();
    assert_eq!(
        integer,
        integer!("19999999999999999999999999999999999999999999999999999999999999998")
    );
    assert!(ever_carried);

    let mut integer: Integer = integer!(
        "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111"
    );
    let ever_carried = integer.fused_reverse_add_asm_interleave();
    assert_eq!(
        integer,
        integer!(
            "2222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222"
        )
    );
    assert!(ever_carried);
}

#[test]
fn test_asm_bug_interleave() {
    let mut integer: Integer = integer!(
        "73482637274560972997068201320438586559407609388647198764094354927116546966373665141586612278184850999294014223899962612538861846880665005592035821032128969264317433200141231772473459431601149579858274630230771087005620368093635487002423793478228705697976867255505359764767301851874010697230735809434080225388354718502970643860097"
    );
    let ever_carried = integer.fused_reverse_add_asm_interleave();
    let expected = integer!(
        "152489471882481554742456553528482077413110888989695014574471101722467102243241644792368899717917271077747653310202612690556565050527950903186146434527566396977531533433612578069455582444454179129914883495047654608632620200334684786908271980699897219854614234220066532710116348641048699087901231378017482535674434646409517917488534"
    );
    assert_eq!(
        integer,
        expected,
        "error: lhs != rhs:\n{}",
        integer.show_differences(&expected)
    );
    assert!(ever_carried);
}

// #[test]
// fn test_asm_random_interleave() {
//     fn test_with_rng(rng: &mut SmallRng, boundary: bool) {
//         let mut length = rng.random_range(1_000usize..=1_000_000usize);

//         if boundary {
//             const MASK: usize = !63;
//             length &= MASK;
//         }

//         // generate a random string with `length` random digits
//         let mut random_string = String::with_capacity(length);
//         for _ in 0..length {
//             let random_digit_char: char = rng.random_range('0'..='9');
//             random_string.push(random_digit_char);
//         }

//         // if the first character in the string is 0, regenerate just that digit
//         if random_string.starts_with('0') {
//             let random_digit_char_nonzero: char = rng.random_range('1'..='9');
//             random_string.remove(0);
//             random_string.insert(0, random_digit_char_nonzero);
//         }

//         let mut integer1 = integer!(&random_string);
//         let mut reversed_integer_stable = Integer(vec![Limb::new(); integer1.0.len()]);
//         integer1.reverse_into_integer(&mut reversed_integer_stable);

//         let (known_correct_result, known_correct_carried): (Integer, bool) =
//             integer1.clone() + reversed_integer_stable;
//         let asm_carried = integer1.fused_reverse_add_asm_interleave();

//         assert_eq!(integer1, known_correct_result);
//         assert_eq!(asm_carried, known_correct_carried);
//     }

//     let mut rng = SmallRng::seed_from_u64(0xdead_beef);
//     for _ in 0..256 {
//         test_with_rng(&mut rng, false);
//         test_with_rng(&mut rng, true);
//     }
// }

#[test]
fn test_limb_ror4_galois() {
    use std::mem::transmute;

    const TEST: u8x64 = unsafe {
        transmute::<std::simd::Simd<u64, 8>, std::simd::Simd<u8, 64>>(u64x8::from_array([
            0x0103070f1f3f7fff,
            0xfffefcf8f0e0c080,
            0x0103070f1f3f7fff,
            0xfffefcf8f0e0c080,
            0x0103070f1f3f7fff,
            0xfffefcf8f0e0c080,
            0x0103070f1f3f7fff,
            0xfffefcf8f0e0c080,
        ]))
    };
    //const TEST_512: u8x64 = // TEST 4 times in a row

    const EXPECTED_OUTPUT: u8x64 = unsafe {
        transmute(u64x8::from_array([
            0x103070f0f1f3f7ff,
            0xffefcf8f0f0e0c08,
            0x103070f0f1f3f7ff,
            0xffefcf8f0f0e0c08,
            0x103070f0f1f3f7ff,
            0xffefcf8f0f0e0c08,
            0x103070f0f1f3f7ff,
            0xffefcf8f0f0e0c08,
        ]))
    };
    //const EXPECTED_OUTPUT_512: u8x64 = // EXPECTED_OUTPUT 4 times in a row

    let output = Limb::ror4_galois(TEST);
    assert_eq!(output, EXPECTED_OUTPUT);
}

#[test]
fn test_evil_shift() {
    let integer = integer!("196");
    let skip_len = integer.skip_len();
    let evil_results = integer.0[0].evil_shift(skip_len); // awesome variable name
    dbg!(evil_results);
    assert_eq!(evil_results[0], integer!("196000").0[0].0);
    assert_eq!(evil_results[1], u8x64::splat(0));

    let integer = Integer(vec![Limb(u8x64::splat(1))]);
    let skip_len = integer.skip_len();
    let evil_results = integer.0[0].evil_shift(skip_len);

    assert_eq!(evil_results[0], u8x64::splat(0));
    assert_eq!(evil_results[1], u8x64::splat(1));
}
