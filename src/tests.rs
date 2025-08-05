use super::*;

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
    let limb = Limb::new_from_value(u64::MAX as u128);
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
    let reversed_integer = integer.reverse();
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
    let reversed_integer = integer.reverse();
    assert_eq!(
        format!("{reversed_integer}"),
        "1234567800000000123456780000000012345678000000001234567887654321000000008765432100000000876543210000000087654321"
    );
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
        Integer(vec![
            Limb(u8x64::splat(0)),
            Limb(u8x64::from([
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
            ]))
        ])
    );
}

#[test]
fn test_limb_len() {
    let limb = Limb::new();
    assert_eq!(limb.len(), 0);

    let limb = Limb::new_from_value(1);
    assert_eq!(limb.len(), 1);

    let limb = Limb::new_from_value(12345);
    assert_eq!(limb.len(), 5);

    let limb = Limb::new_from_value(100000000000000000000000000000000000000); // 39 digits
    assert_eq!(limb.len(), 39);

    let limb = Limb::new_from_value(340282366920938463463374607431768211455); // 39 digits
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
