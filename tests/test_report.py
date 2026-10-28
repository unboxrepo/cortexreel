import json
import re
from pathlib import Path

import numpy as np

from cortexreel import moments as mom
from cortexreel import report
from cortexreel.regions import load_config
from cortexreel.score import compute_scores

EVIL = "<img src=x onerror=alert(1)>.mp4"


def _fixture(tmp_path):
    config = load_config()
    groups = list(config["groups"])
    n_tr, per = 30, 3
    masks = {}
    for i, name in enumerate(groups):
        m = np.zeros(per * len(groups), dtype=bool)
        m[i * per : (i + 1) * per] = True
        masks[name] = m
    preds = np.full((n_tr, per * len(groups)), 0.05, np.float32)
    preds[12:16] = -0.5
    scores = compute_scores(preds, np.arange(n_tr, dtype=float), 1.0, masks, config)
    found = mom.find_moments(scores)
    for m in found:
        m.words = "<script>alert('w')</script>"
    video = tmp_path / EVIL  # never created: no player, no thumbnails
    metrics = {"use_text": True, "video_sha256": "0" * 64}
    analysis = report.build_analysis(video, metrics, scores, found, config, masks)
    return analysis, scores, found, video


def test_analysis_json_shape(tmp_path):
    analysis, scores, found, _ = _fixture(tmp_path)
    assert analysis["schema"] == 1
    assert set(analysis["summary"]) == {"overall", "hook", "hold", "ending"}
    assert len(analysis["curves"]["engagement"]) == len(scores.times)
    assert any(m["kind"] == "dip" for m in analysis["moments"])
    json.dumps(analysis)  # serialisable
    assert "curves" not in report.summary_json(analysis)


def test_html_is_offline_and_escaped(tmp_path):
    analysis, scores, found, video = _fixture(tmp_path)
    page = report.render_html(analysis, scores, found, tmp_path, video)

    assert not re.search(r"""(src|href)\s*=\s*["']?\s*(https?:)?//""", page.replace(
        f'href="{report.LICENSE_URL}"', ""
    ))
    assert "<link" not in page
    assert not re.search(r"<script[^>]*\bsrc", page)
    assert "@import" not in page and "url(" not in page
    # The only URL is the license link in the footer.
    assert re.findall(r"https?://[^\s\"'<>]+", page) == [report.LICENSE_URL]

    assert EVIL not in page
    assert "&lt;img src=x onerror=alert(1)&gt;.mp4" in page
    assert "<script>alert(" not in page
    assert page.count("<script") == 1
    assert "<video" not in page  # video file does not exist


def test_write_report_files(tmp_path):
    analysis, scores, found, video = _fixture(tmp_path)
    out = tmp_path / "out" / "x"
    json_path, html_path = report.write_report(out, video, analysis, scores, found)
    assert json.loads(Path(json_path).read_text(encoding="utf-8"))["schema"] == 1
    assert Path(html_path).read_text(encoding="utf-8").startswith("<!doctype html>")
