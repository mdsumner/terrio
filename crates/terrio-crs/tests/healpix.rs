use terrio_crs::{SpatialRef, Transformer};

#[test]
fn pinned_healpix_and_rhealpix_fixtures() {
    let data: serde_json::Value = serde_json::from_str(include_str!("data/healpix_benchmark.json")).unwrap();
    assert_eq!(data["environment"]["proj_version"], "9.9.0");
    let fixtures = data["fixtures"].as_array().unwrap();
    assert_eq!(fixtures.len(), 2);
    for case in fixtures {
        let id = case["id"].as_str().unwrap();
        let source = SpatialRef::parse(case["src"].as_str().unwrap()).unwrap();
        let target = SpatialRef::parse(case["dst"].as_str().unwrap()).unwrap();
        let (x, y) = (case["point"][0].as_f64().unwrap(), case["point"][1].as_f64().unwrap());
        let actual = Transformer::new(&source, &target).unwrap().transform(x, y).unwrap();
        assert!((actual.0 - case["oracle"]["fwd"][0].as_f64().unwrap()).abs() < 0.001, "{id} {actual:?}");
        assert!((actual.1 - case["oracle"]["fwd"][1].as_f64().unwrap()).abs() < 0.001, "{id} {actual:?}");
        let inverse = Transformer::new(&target, &source).unwrap().transform(actual.0, actual.1).unwrap();
        assert!((inverse.0 - case["oracle"]["inv"][0].as_f64().unwrap()).abs() < 1e-8, "{id} {inverse:?}");
        assert!((inverse.1 - case["oracle"]["inv"][1].as_f64().unwrap()).abs() < 1e-8, "{id} {inverse:?}");
        let wkt1 = case["oracle"]["dst_export"]["wkt1_gdal_bound"].as_str().unwrap();
        let written = target.to_wkt().unwrap();
        // WKT 1 has no method for them: GDAL's custom_proj4, with the PROJ string.
        assert!(written.contains("PROJECTION[\"custom_proj4\"]"), "{written}");
        for wkt in [wkt1, written.as_str()] {
            let reopened = SpatialRef::parse(wkt).unwrap();
            let again = Transformer::new(&source, &reopened).unwrap().transform(x, y).unwrap();
            assert!((again.0 - actual.0).abs() < 0.001 && (again.1 - actual.1).abs() < 0.001, "{wkt}");
        }
        // PROJ has no EPSG method for them, so PROJJSON export refuses them.
        assert!(target.to_projjson().is_err(), "{id}");
    }
}

fn angular_error(actual: (f64, f64), expected: &serde_json::Value) -> f64 {
    let lon = expected[0].as_f64().unwrap();
    let lat = expected[1].as_f64().unwrap();
    let dlon = (actual.0 - lon + 180.0).rem_euclid(360.0) - 180.0;
    (dlon * lat.to_radians().cos()).abs().max((actual.1 - lat).abs())
}

#[test]
fn every_configuration_figure_and_unit_matches_frozen_proj() {
    let data: serde_json::Value = serde_json::from_str(include_str!("data/healpix_reference.json")).unwrap();
    assert_eq!(data["environment"]["proj"], "9.8.1");
    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let source = SpatialRef::parse(case["source"].as_str().unwrap()).unwrap();
        let factor = case["metres_per_unit"].as_f64().unwrap();
        // terrio's own WKT1, written from the PROJ string, is checked as the others.
        let written = SpatialRef::parse(case["target"].as_str().unwrap()).unwrap().to_wkt().unwrap();
        for encoding in ["target", "wkt1", "wkt1_esri", "wkt2", "written"] {
            let text = if encoding == "written" { written.as_str() } else { case[encoding].as_str().unwrap() };
            let target = SpatialRef::parse(text).unwrap_or_else(|e| panic!("{name} {encoding}: {e}"));
            let forward = Transformer::new(&source, &target).unwrap_or_else(|e| panic!("{name} {encoding}: {e}"));
            let inverse = Transformer::new(&target, &source).unwrap_or_else(|e| panic!("{name} {encoding}: {e}"));
            let edge: Vec<usize> = case["edge_samples"].as_array().unwrap().iter().map(|i| i.as_u64().unwrap() as usize).collect();
            for (i, sample) in case["samples"].as_array().unwrap().iter().enumerate() {
                let point = (sample[0].as_f64().unwrap(), sample[1].as_f64().unwrap());
                let expected = (sample[2][0].as_f64().unwrap(), sample[2][1].as_f64().unwrap());
                let actual = forward.transform(point.0, point.1).unwrap_or_else(|e| panic!("{name} {encoding} {point:?}: {e}"));
                let error = (actual.0 - expected.0).abs().max((actual.1 - expected.1).abs()) * factor;
                assert!(error <= 0.001, "{name} {encoding} {point:?}: forward error {error} m");
                let actual = inverse.transform(expected.0, expected.1);
                // On the edge of the image (a pole, the antimeridian, a polar cap's edge), the
                // last bit of the coordinates moves the point in or out of the image: the
                // platform's libm, and the US survey foot (PROJ 9.8.1 takes it in these methods'
                // own +units as 0.304800609601219, terrio, as PROJ's unitconvert for other
                // methods, as 1200/3937), decide it. There either is right, and a point terrio
                // unprojects must be the projected one.
                if edge.contains(&i) {
                    if let Ok(actual) = actual {
                        let error = angular_error(actual, &serde_json::json!([point.0, point.1]));
                        assert!(error <= 1e-8, "{name} {encoding} {point:?}: edge inverse {actual:?}, {error} degrees");
                    }
                } else if sample[3].is_null() {
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
                if sample[1].is_null() {
                    assert!(actual.is_err(), "{name} {encoding}: inverse must refuse {point:?}, as PROJ does: {actual:?}");
                } else {
                    let actual = actual.unwrap_or_else(|e| panic!("{name} {encoding} inverse {point:?}: {e}"));
                    let error = angular_error(actual, &sample[1]);
                    assert!(error <= 1e-8, "{name} {encoding}: inverse {point:?} error {error} degrees");
                }
            }
        }
    }
}

#[test]
fn polar_squares_outside_zero_to_three_are_refused() {
    for bad in ["+north_square=4", "+south_square=-1", "+north_square=7 +south_square=1"] {
        let crs = format!("+proj=rhealpix +R=1 {bad}");
        assert!(SpatialRef::parse(&crs).and_then(|t| Transformer::new(&SpatialRef::parse("+proj=longlat +R=1")?, &t)).is_err(), "{crs}");
    }
}
