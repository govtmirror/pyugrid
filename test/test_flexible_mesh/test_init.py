import pyugrid


def test_imports_required():
    """
    Test importing flexible mesh API elements. These API elements requires the installation of optional dependencies.
    """

    assert hasattr(pyugrid, 'FlexibleMesh')


