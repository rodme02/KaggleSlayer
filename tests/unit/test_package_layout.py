"""Sanity checks that the new package layout is importable."""

def test_top_level_package_importable():
    import kaggle_slayer
    assert kaggle_slayer.__name__ == "kaggle_slayer"


def test_harness_subpackage_importable():
    import kaggle_slayer.harness
    assert kaggle_slayer.harness.__name__ == "kaggle_slayer.harness"


def test_harness_registry_importable():
    import kaggle_slayer.harness.registry
    assert kaggle_slayer.harness.registry.__name__ == "kaggle_slayer.harness.registry"
