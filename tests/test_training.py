from paint_inspection.training import calibrate_threshold


def test_good_only_threshold_calibration():
    report = calibrate_threshold([.1, .12, .11, .13])
    assert .13 <= report["recommended_threshold"] < .2


def test_labelled_threshold_calibration():
    report = calibrate_threshold([.1,.2,.15], [.75,.8,.9])
    assert report["f1"] == 1 and report["fp"] == 0 and report["fn"] == 0

