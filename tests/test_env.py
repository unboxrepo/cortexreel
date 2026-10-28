import sys
from pathlib import Path

import pytest

from cortexreel import env

ALL_VARS = [*env.PRIVACY_DEFAULTS, *env.OFFLINE_VARS, "HF_HOME", "HF_TOKEN_PATH", "CORTEXREEL_HOME"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for key in ALL_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CORTEXREEL_HOME", str(tmp_path / "home"))


def test_privacy_vars_set():
    import os

    env.configure()
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert os.environ["DO_NOT_TRACK"] == "1"
    assert os.environ["WANDB_MODE"] == "disabled"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    assert not env.is_offline()


def test_user_override_kept(monkeypatch):
    import os

    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("HF_HOME", "/custom/hf")
    env.configure()
    assert os.environ["WANDB_MODE"] == "offline"
    assert os.environ["HF_HOME"] == "/custom/hf"


def test_cache_root_and_hf_home(tmp_path):
    import os

    root = env.configure()
    assert root == tmp_path / "home"
    assert Path(os.environ["HF_HOME"]) == root / "huggingface"


def test_default_cache_root(monkeypatch, tmp_path):
    monkeypatch.delenv("CORTEXREEL_HOME")
    if sys.platform == "win32":
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        assert env.cache_root() == tmp_path / "cortexreel"
    else:
        assert env.cache_root() == Path.home() / ".cache" / "cortexreel"


def test_offline_flag():
    import os

    env.configure(offline=True)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert env.is_offline()


def test_offline_flag_overrides_user_value(monkeypatch):
    import os

    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    env.configure(offline=True)
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_cli_offline_parsing():
    from cortexreel.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["--offline", "doctor"]).offline is True
    assert parser.parse_args(["predict", "a.mp4", "--offline"]).offline is True
    assert parser.parse_args(["predict", "a.mp4"]).offline is False


def test_cli_does_not_import_heavy_modules_early():
    import subprocess

    code = (
        "import sys, cortexreel.cli; "
        "heavy = [m for m in ('torch', 'transformers', 'huggingface_hub') if m in sys.modules]; "
        "print(heavy); sys.exit(1 if heavy else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
