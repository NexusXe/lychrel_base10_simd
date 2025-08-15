use super::*;

#[test]
fn test_iterate() {
    let starting_integer = integer!("196");
    let limit: usize = 501;

    let result = crate::iterate(1..limit, starting_integer);
    assert_eq!(result.last_iteration, limit);
    assert_eq!(
        result.end_integer,
        integer!(
            "931423633824289735373068912980022207417878996006463505496394221534435934470613658741952102416192959097792354454298681058301624201150137856317064440634435122492605494375600699878703812220089208969284526982429436324128"
        )
    );
    assert_eq!(
        result.end_integer.clone().pack().unpack(),
        result.end_integer
    );

    let starting_integer = integer!("197");

    let result = crate::iterate(1..limit, starting_integer);

    assert_eq!(result.last_iteration, 8);
    assert_eq!(result.end_integer, integer!("881188"));
}
