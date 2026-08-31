use super::*;

#[test]
fn test_iterate() {
    let starting_integer = integer!("196");

    let limit = 2;
    let result = crate::reference::iterate(1..limit, starting_integer.clone(), None);
    assert_eq!(result.last_iteration, limit);
    assert_eq!(result.end_integer, integer!("887"));

    let limit = 501;

    let result = crate::reference::iterate(1..limit, starting_integer, None);
    assert_eq!(result.last_iteration, limit);
    assert_eq!(
        result.end_integer,
        integer!(
            "931423633824289735373068912980022207417878996006463505496394221534435934470613658741952102416192959097792354454298681058301624201150137856317064440634435122492605494375600699878703812220089208969284526982429436324128"
        )
    );
    assert_eq!(
        result.end_integer.clone().pack().unpack(std::alloc::Global),
        result.end_integer
    );

    let starting_integer = integer!("197");

    let result = crate::reference::iterate(1..limit, starting_integer, None);

    assert_eq!(result.last_iteration, 8);
    assert_eq!(result.end_integer, integer!("881188"));
}

#[cfg(target_pointer_width = "64")]
#[test]
fn test_iterate_huge() {
    let allocator = HugePageAllocator::init().unwrap();
    let starting_limb = Limb::new_from_value(196);
    let mut internal_vec: Vec<Limb, HugePageAllocator> = Vec::new_in(allocator);
    internal_vec.push(starting_limb);
    let starting_integer = Integer(internal_vec);
    let limit = 501;

    let expected_global_integer = integer!(
        "931423633824289735373068912980022207417878996006463505496394221534435934470613658741952102416192959097792354454298681058301624201150137856317064440634435122492605494375600699878703812220089208969284526982429436324128"
    );

    let mut expected_huge_vec: Vec<Limb, HugePageAllocator> =
        Vec::with_capacity_in(expected_global_integer.0.len(), allocator);
    for limb in expected_global_integer.0 {
        expected_huge_vec.push(limb);
    }

    let expected_integer = Integer(expected_huge_vec);

    let result = crate::reference::iterate(1..limit, starting_integer, None);

    assert_eq!(result.last_iteration, limit);
    assert_eq!(result.end_integer, expected_integer);
    // assert_eq!(
    //     result.end_integer,
    //     result.end_integer.clone().pack().unpack(HugePageAllocator)
    // );
}
