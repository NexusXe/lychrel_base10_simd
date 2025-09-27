use super::*;
use rand::prelude::*;

#[test]
fn test_len_empty() {
    let limb = Limb::new();
    assert_eq!(limb.len(), 0);
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
    let limb1: Limb = LimbVec::splat(9).into();
    let limb2: Limb = LimbVec::splat(1).into();
    let packed = limb1.pack(limb2);
    assert_eq!(packed, LimbVec::splat(0x19).into());

    let packed: Limb = LimbVec::splat(0x19).into();
    let (limb1, limb2) = packed.unpack();
    assert_eq!(limb1, LimbVec::splat(9).into());
    assert_eq!(limb2, LimbVec::splat(1).into());
}

#[test]
fn test_pack_unpack_limb_random() {
    fn test_with_seed(seed: u64) {
        // get two deterministically random limbs from a SmallRng with a constant seed
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut arr1: [LimbVecScalar; LV_LEN] = [0; LV_LEN];
        let mut arr2: [LimbVecScalar; LV_LEN] = [0; LV_LEN];

        for (item1, item2) in arr1.iter_mut().zip(arr2.iter_mut()) {
            *item1 = rng.random_range(0..10);
            *item2 = rng.random_range(0..10);
        }
        let limb1 = Limb(LimbVec::from(arr1));
        let limb2 = Limb(LimbVec::from(arr2));

        let integer = Integer(vec![limb1, limb2]);
        let packed = integer.clone().pack();
        assert_eq!(packed.clone().unpack(GlobalAllocator), integer);

        let bytes: Vec<LimbVecScalar> = packed.clone().into_bytes();
        let organized_bytes: Vec<[LimbVecScalar; LV_LEN]> = bytes
            .chunks(LV_LEN)
            .map(|chunk| chunk.try_into().unwrap())
            .collect();

        let reconstructed = Integer::from_bytes(&organized_bytes, GlobalAllocator);
        assert_eq!(reconstructed, packed);
        assert_eq!(reconstructed.unpack(GlobalAllocator), integer);
    }

    test_with_seed(0xdead_beef);
    test_with_seed(0xbaad_f00d);
}

#[test]
fn test_pack_unpack_integer() {
    let limb1: Limb = LimbVec::splat(9).into();
    let limb2: Limb = LimbVec::splat(1).into();
    let integer = Integer(vec![limb1, limb2]);
    let packed = integer.pack();
    assert_eq!(packed, Integer(vec![LimbVec::splat(0x19).into()]));

    let packed = Integer(vec![LimbVec::splat(0x19).into()]);
    let unpacked = packed.unpack(GlobalAllocator);
    assert_eq!(unpacked, Integer(vec![limb1, limb2]));

    let packed = Integer(vec![LimbVec::splat(0x19).into(); 4]);
    let unpacked = packed.unpack(GlobalAllocator);
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
    let mut arrs: Vec<[LimbVecScalar; LV_LEN]> = Vec::with_capacity(LEN);
    for _ in 0..LEN {
        let mut arr: [LimbVecScalar; LV_LEN] = [0; LV_LEN];
        for item in &mut arr {
            *item = rng.random_range(0..10);
        }
        arrs.push(arr);
    }

    let last_arr = arrs.last_mut().unwrap();

    for item in last_arr.iter_mut().take(LV_LEN).skip(32) {
        *item = 0;
    }

    let mut limbs: Vec<Limb> = Vec::with_capacity(LEN);
    for arr in arrs {
        limbs.push(Limb(LimbVec::from(arr)));
    }

    let integer = Integer(limbs);
    assert!(!integer.has_carries());
    assert_eq!(integer, integer.clone().pack().unpack(GlobalAllocator));

    // instead of writing to an actual file, write to a buffer
    let mut buffer = Vec::new();

    let data_to_write = integer.clone().pack().into_bytes();

    buffer.write_all(&data_to_write).unwrap();

    // now read it as if it was from a file

    let mut checkpoint_read: Vec<LimbVecScalar> = Vec::new();
    checkpoint_read.write_all(&buffer).unwrap();
    let organized_bytes: Vec<[LimbVecScalar; LV_LEN]> = checkpoint_read
        .chunks(LV_LEN)
        .map(|chunk| chunk.try_into().unwrap())
        .collect();
    let integer_read = Integer::from_bytes(&organized_bytes, GlobalAllocator);

    //assert_eq!(integer, integer_read.clone());
    assert_eq!(integer, integer_read.clone().unpack(GlobalAllocator));
    assert!(!integer_read.unpack(GlobalAllocator).has_carries());
}

#[test]
fn test_fused_reverse_add_asm_simple_interleave() {
    let mut integer = integer!("196");
    let ever_carried = integer.fused_reverse_add_asm_interleave();
    assert_eq!(integer, integer!("887"));
    assert!(ever_carried);
}

#[test]
fn test_fused_reverse_add_asm_simple_interleave_2() {
    let mut integer = Integer(vec![Limb({
        let mut arr: [LimbVecScalar; LV_LEN] = [0; LV_LEN];
        arr[0] = 5;
        arr[LV_LEN - 1] = 6;
        LimbVec::from(arr)
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

    let mut integer2 = integer!("99999999");
    let ever_carried = integer2.fused_reverse_add_asm_interleave();
    assert_eq!(integer2, integer!("199999998"));
    assert!(ever_carried);

    let mut integer3 = integer!("11111111111111111111111111111111");
    let ever_carried = integer3.fused_reverse_add_asm_interleave();
    assert_eq!(integer3, integer!("22222222222222222222222222222222"));
    assert!(!ever_carried);

    let mut integer4 = Integer(vec![Limb(LimbVec::splat(9))]);
    let ever_carried = integer4.fused_reverse_add_asm_interleave();
    assert_eq!(
        integer4,
        integer!("19999999999999999999999999999999999999999999999999999999999999998")
    );
    assert!(ever_carried);
}

#[test]
fn test_asm_bug_interleave() {
    let mut integer = integer!(
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
