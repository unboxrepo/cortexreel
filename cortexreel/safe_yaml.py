"""A YAML loader for TRIBE's config.yaml that does not execute arbitrary Python.

TRIBE's published config uses exactly two Python-specific tags:

* ``!!python/tuple``                               -> ``tuple``
* ``!!python/object/apply:pathlib.PosixPath``      -> ``str`` (POSIX-joined path)

``PosixPath`` cannot even be instantiated on Windows, and the values are only
paths on Meta's cluster that tribev2 later overrides, so a plain string is the
right representation. Every other ``!!python/...`` tag raises a ConstructorError,
exactly like ``yaml.SafeLoader``.
"""

from __future__ import annotations

import posixpath

import yaml


class TribeConfigLoader(yaml.SafeLoader):
    pass


def _construct_tuple(loader: yaml.SafeLoader, node: yaml.Node) -> tuple:
    return tuple(loader.construct_sequence(node, deep=True))


def _construct_posix_path(loader: yaml.SafeLoader, node: yaml.Node) -> str:
    if not isinstance(node, yaml.SequenceNode):
        raise yaml.constructor.ConstructorError(
            None, None, "PosixPath arguments must be a plain list", node.start_mark
        )
    parts = loader.construct_sequence(node)
    if not parts or not all(isinstance(p, str) for p in parts):
        raise yaml.constructor.ConstructorError(
            None, None, "PosixPath arguments must be strings", node.start_mark
        )
    return posixpath.join(*parts)


TribeConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_tuple)
TribeConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath", _construct_posix_path
)


def load(stream) -> object:
    return yaml.load(stream, Loader=TribeConfigLoader)  # noqa: S506 - restricted loader


class _YamlShim:
    """Stand-in for the ``yaml`` module inside tribev2.demo_utils.

    Whatever ``Loader=`` tribev2 asks for, parsing goes through
    ``TribeConfigLoader``. Every other attribute falls through to PyYAML.
    """

    def load(self, stream, Loader=None):  # noqa: N803 - mirrors yaml.load
        return load(stream)

    def __getattr__(self, name):
        return getattr(yaml, name)


yaml_shim = _YamlShim()
