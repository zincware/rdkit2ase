import sys

import pytest

from rdkit2ase.utils import find_connected_components


@pytest.mark.parametrize("networkx", [True, False])
def test_find_connected_components_networkx(monkeypatch, networkx):
    if not networkx:
        monkeypatch.setitem(sys.modules, "networkx", None)

    connectivity = [
        (0, 1, 1.0),
        (0, 1, 1.0), # duplicate edge
        (1, 2, 1.0),
        (4, 3, 1.0),
        (4, 5, 1.0),
        (6, 7, 1.0),
    ]

    components = list(find_connected_components(connectivity))
    assert len(components) == 3
    assert set(components[0]) == {0, 1, 2}
    assert set(components[1]) == {3, 4, 5}
    assert set(components[2]) == {6, 7}
