import part2pop

import pop2obs


def test_package_and_dependency_import():
    assert pop2obs.__version__
    assert part2pop.__version__
