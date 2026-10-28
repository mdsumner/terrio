use terrio_crs::{SpatialRef, Transformer};

#[test]
fn pinned_oblique_mollweide_fixture() {
    let data: serde_json::Value = serde_json::from_str(include_str!("data/ob_tran_benchmark.json")).unwrap();
    assert_eq!(data["environment"]["proj_version"], "9.9.0");
    for case in data["fixtures"].as_array().unwrap() {
        assert_eq!(case["id"], "seed-ob_tran-moll");
        let source = SpatialRef::parse(case["src"].as_str().unwrap()).unwrap();
        let target = SpatialRef::parse(case["dst"].as_str().unwrap()).unwrap();
        assert!(!target.is_rotated_pole());
        let (x, y) = (case["point"][0].as_f64().unwrap(), case["point"][1].as_f64().unwrap());
        let actual = Transformer::new(&source, &target).unwrap().transform(x, y).unwrap();
        assert!((actual.0 - case["oracle"]["fwd"][0].as_f64().unwrap()).abs() < 0.001);
        assert!((actual.1 - case["oracle"]["fwd"][1].as_f64().unwrap()).abs() < 0.001);
        let inverse = Transformer::new(&target, &source).unwrap().transform(actual.0, actual.1).unwrap();
        assert!((inverse.0 - case["oracle"]["inv"][0].as_f64().unwrap()).abs() < 1e-8);
        assert!((inverse.1 - case["oracle"]["inv"][1].as_f64().unwrap()).abs() < 1e-8);
        let wkt1 = case["oracle"]["dst_export"]["wkt1_gdal_bound"].as_str().unwrap();
        let written = target.to_wkt().unwrap();
        assert!(written.contains("+proj=ob_tran +o_proj=moll"), "{written}");
        for wkt in [wkt1, written.as_str()] {
            let reopened = SpatialRef::parse(wkt).unwrap();
            let again = Transformer::new(&source, &reopened).unwrap().transform(x, y).unwrap();
            assert!((again.0 - actual.0).abs() < 0.001 && (again.1 - actual.1).abs() < 0.001, "{wkt}");
        }
    }
}

fn angular_error(actual: (f64, f64), expected: &serde_json::Value) -> f64 {
    let lon = expected[0].as_f64().unwrap();
    let lat = expected[1].as_f64().unwrap();
    let dlon = (actual.0 - lon + 180.0).rem_euclid(360.0) - 180.0;
    (dlon * lat.to_radians().cos()).abs().max((actual.1 - lat).abs())
}

#[test]
fn rotated_projections_pole_forms_figures_and_units_match_frozen_proj() {
    let data: serde_json::Value = serde_json::from_str(include_str!("data/ob_tran_reference.json")).unwrap();
    assert_eq!(data["environment"]["proj"], "9.8.1");
    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let source = SpatialRef::parse(case["source"].as_str().unwrap()).unwrap();
        let factor = case["metres_per_unit"].as_f64().unwrap();
        for encoding in ["target", "wkt1", "wkt2"] {
            let target = SpatialRef::parse(case[encoding].as_str().unwrap()).unwrap_or_else(|e| panic!("{name} {encoding}: {e}"));
            let forward = Transformer::new(&source, &target).unwrap();
            let inverse = Transformer::new(&target, &source).unwrap();
            for sample in case["samples"].as_array().unwrap() {
                let point = (sample[0].as_f64().unwrap(), sample[1].as_f64().unwrap());
                let actual = forward.transform(point.0, point.1);
                if sample[2].is_null() {
                    assert!(actual.is_err(), "{name} {encoding}: forward must refuse {point:?}, as PROJ does: {actual:?}");
                    continue;
                }
                let expected = (sample[2][0].as_f64().unwrap(), sample[2][1].as_f64().unwrap());
                let actual = actual.unwrap_or_else(|e| panic!("{name} {encoding} {point:?}: {e}"));
                let error = (actual.0 - expected.0).abs().max((actual.1 - expected.1).abs()) * factor;
                assert!(error <= 0.001, "{name} {encoding} {point:?}: forward error {error} m");
                let actual = inverse.transform(expected.0, expected.1);
                if sample[3].is_null() {
                    assert!(actual.is_err(), "{name} {encoding}: inverse must refuse {expected:?}, as PROJ does: {actual:?}");
                } else {
                    let actual = actual.unwrap_or_else(|e| panic!("{name} {encoding} inverse {point:?}: {e}"));
                    let error = angular_error(actual, &sample[3]);
                    assert!(error <= 1e-8, "{name} {encoding} {point:?}: inverse error {error} degrees");
                }
            }
            for sample in case["inverse_samples"].as_array().unwrap() {
                let point = (sample[0][0].as_f64().unwrap(), sample[0][1].as_f64().unwrap());
                let actual = inverse.transform(point.0, point.1);
                // PROJ's spherical sinusoidal inverse returns a latitude past a pole (it does
                // without the rotation too), which the rotation then turns into a point; terrio's
                // sinusoidal refuses it, rotated or not.
                let northing =
                    (point.1 * factor - case["false_northing_metres"].as_f64().unwrap()) / case["radius_metres"].as_f64().unwrap();
                let past_pole = name.starts_with("ob_tran sinu +R=") && northing.abs() > std::f64::consts::FRAC_PI_2;
                if sample[1].is_null() || past_pole {
                    assert!(actual.is_err(), "{name} {encoding}: inverse domain {point:?}: {actual:?}");
                } else {
                    let actual = actual.unwrap_or_else(|e| panic!("{name} {encoding} inverse domain {point:?}: {e}"));
                    assert!(angular_error(actual, &sample[1]) <= 1e-8, "{name} {encoding}: inverse {point:?}");
                }
            }
        }
    }
}

#[test]
fn definitions_proj_refuses_are_refused() {
    let data: serde_json::Value = serde_json::from_str(include_str!("data/ob_tran_reference.json")).unwrap();
    for definition in data["refused"].as_array().unwrap() {
        let definition = definition.as_str().unwrap();
        let error = SpatialRef::parse(definition).err().unwrap_or_else(|| panic!("{definition} must be refused"));
        assert!(error.to_string().contains("invalid +proj=ob_tran"), "{definition}: {error}");
    }
}
