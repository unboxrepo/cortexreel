import pytest
import yaml

from cortexreel import safe_yaml

TRIBE_SNIPPET = """
excludes: !!python/tuple
- __pycache__
- .git
folder: !!python/object/apply:pathlib.PosixPath
- /
- checkpoint
- cache
plain: [1, 2]
"""


def test_accepts_tribe_tags():
    data = safe_yaml.load(TRIBE_SNIPPET)
    assert data["excludes"] == ("__pycache__", ".git")
    assert isinstance(data["excludes"], tuple)
    assert data["folder"] == "/checkpoint/cache"
    assert data["plain"] == [1, 2]


def test_relative_posix_path():
    data = safe_yaml.load("p: !!python/object/apply:pathlib.PosixPath [a, b]")
    assert data["p"] == "a/b"


@pytest.mark.parametrize(
    "doc",
    [
        "x: !!python/object/apply:os.system ['echo pwned']",
        "x: !!python/object/apply:subprocess.Popen [['calc']]",
        "x: !!python/object/new:pathlib.PosixPath ['/tmp']",
        "x: !!python/object:collections.OrderedDict {}",
        "x: !!python/name:os.system",
        "x: !!python/module:os",
        "x: !!python/list [1]",
        "x: !!python/dict {a: 1}",
        "x: !!python/object/apply:pathlib.WindowsPath ['C:/']",
        "x: !!python/object/apply:builtins.eval ['1+1']",
    ],
)
def test_rejects_other_python_tags(doc):
    with pytest.raises(yaml.constructor.ConstructorError):
        safe_yaml.load(doc)


@pytest.mark.parametrize(
    "doc",
    [
        "x: !!python/object/apply:pathlib.PosixPath {args: [a]}",
        "x: !!python/object/apply:pathlib.PosixPath [[nested]]",
        "x: !!python/object/apply:pathlib.PosixPath []",
    ],
)
def test_rejects_malformed_posix_path(doc):
    with pytest.raises(yaml.constructor.ConstructorError):
        safe_yaml.load(doc)


def test_shim_ignores_requested_unsafe_loader():
    with pytest.raises(yaml.constructor.ConstructorError):
        safe_yaml.yaml_shim.load("x: !!python/name:os.system", Loader=yaml.UnsafeLoader)
    assert safe_yaml.yaml_shim.dump is yaml.dump


def test_base_safe_loader_untouched():
    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.safe_load("x: !!python/tuple [1]")
