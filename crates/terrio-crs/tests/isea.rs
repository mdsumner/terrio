use terrio_crs::{SpatialRef, Transformer};

fn angular_error(actual: (f64, f64), expected: &serde_json::Value) -> f64 {
    let lon = expected[0].as_f64().unwrap();
    let lat = expected[1].as_f64().unwrap();
    let dlon = (actual.0 - lon + 180.0).rem_euclid(360.0) - 180.0;
    (dlon * lat.to_radians().cos()).abs().max((actual.1 - lat).abs())
}

#[test]
fn every_configuration_figure_and_unit_matches_frozen_proj() {
    let data: serde_json::Value = serde_json::from_str(include_str!("data/isea_reference.json")).unwrap();
    assert_eq!(data["environment"]["proj"], "9.8.1");
    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let source = SpatialRef::parse(case["source"].as_str().unwrap()).unwrap();
        let factor = case["metres_per_unit"].as_f64().unwrap();
        // terrio's own WKT1, written from the PROJ string, is checked as the others.
        let written = SpatialRef::parse(case["target"].as_str().unwrap()).unwrap().to_wkt().unwrap();
        assert!(written.contains("PROJECTION[\"custom_proj4\"]"), "{written}");
        for encoding in ["target", "wkt1", "wkt1_esri", "wkt2", "written"] {
            let text = if encoding == "written" { written.as_str() } else { case[encoding].as_str().unwrap() };
            let target = SpatialRef::parse(text).unwrap_or_else(|e| panic!("{name} {encoding}: {e}"));
            let forward = Transformer::new(&source, &target).unwrap_or_else(|e| panic!("{name} {encoding}: {e}"));
            let inverse = Transformer::new(&target, &source).unwrap_or_else(|e| panic!("{name} {encoding}: {e}"));
            for sample in case["samples"].as_array().unwrap() {
                let point = (sample[0].as_f64().unwrap(), sample[1].as_f64().unwrap());
                let expected = (sample[2][0].as_f64().unwrap(), sample[2][1].as_f64().unwrap());
                let actual = forward.transform(point.0, point.1).unwrap_or_else(|e| panic!("{name} {encoding} {point:?}: {e}"));
                let error = (actual.0 - expected.0).abs().max((actual.1 - expected.1).abs()) * factor;
                assert!(error <= 0.001, "{name} {encoding} {point:?}: forward error {error} m");
                // PROJ's inverse, which is not the forward's on an ellipsoid and is refused
                // outside its default grid.
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
                if sample[1].is_null() {
                    assert!(actual.is_err(), "{name} {encoding}: inverse must refuse {point:?}, as PROJ does: {actual:?}");
                } else {
                    let actual = actual.unwrap_or_else(|e| panic!("{name} {encoding} inverse {point:?}: {e}"));
                    let error = angular_error(actual, &sample[1]);
                    assert!(error <= 1e-8, "{name} {encoding}: inverse {point:?} error {error} degrees");
                }
            }
        }
        // PROJ has no EPSG method for it, so PROJJSON export refuses it.
        assert!(SpatialRef::parse(case["target"].as_str().unwrap()).unwrap().to_projjson().is_err(), "{name}");
    }
}

#[test]
fn invalid_and_grid_address_options_are_refused() {
    let lonlat = SpatialRef::parse("+proj=longlat +R=1").unwrap();
    // PROJ 9.8.1 refuses the first seven. The grid addresses of +mode=di, dd and hex are not
    // coordinates, and terrio does not compute them.
    for bad in [
        "+orient=foo",
        "+orient",
        "+mode=foo",
        "+mode",
        "+resolution=4.0",
        "+aperture=-3",
        "+resolution=99999999999999999999x",
        "+mode=di",
        "+mode=dd",
        "+mode=hex",
    ] {
        let crs = format!("+proj=isea +R=1 {bad}");
        assert!(SpatialRef::parse(&crs).and_then(|t| Transformer::new(&lonlat, &t)).is_err(), "{crs}");
    }
    for good in ["+orient=isea", "+orient=pole", "+mode=plane", "+resolution=4 +aperture=3", "+resolution"] {
        let crs = format!("+proj=isea +R=1 {good}");
        assert!(SpatialRef::parse(&crs).and_then(|t| Transformer::new(&lonlat, &t)).is_ok(), "{crs}");
    }
}
