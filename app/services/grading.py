from html import escape

import cv2
import numpy as np

from app.schemas import LeafResult, Scan

PALETTE = [
    (46, 204, 113),
    (52, 152, 219),
    (155, 89, 182),
    (241, 196, 15),
    (230, 126, 34),
    (231, 76, 60),
    (26, 188, 156),
]


def _label_origin(mask: np.ndarray, label: str, font_scale: float) -> tuple[int, int]:
    """Centre the label on the largest visible piece of the leaf (occluded leaves come in pieces)."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    largest = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax()) if count > 1 else 0
    cx, cy = centroids[largest]
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 4)
    height, width = mask.shape
    return int(min(max(cx - w / 2, 0), width - w)), int(min(max(cy + h / 2, h), height - 2))


def render_overlay(rgb: np.ndarray, masks: list[np.ndarray], leaves: list[LeafResult]) -> np.ndarray:
    out = rgb.copy()
    font_scale = max(0.5, max(rgb.shape[:2]) / 1600)
    for leaf in leaves:
        mask = masks[leaf.index]
        color = PALETTE[leaf.index % len(PALETTE)]
        out[mask] = (out[mask] * 0.6 + np.array(color) * 0.4).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)
        label = f"#{leaf.index} {leaf.variegation_pct:.0f}%"
        origin = _label_origin(mask, label, font_scale)
        cv2.putText(out, label, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, label, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


VERDICT_LABELS = {
    "consistent": "Consistent with seller's scan",
    "minor_changes": "Minor changes since shipping",
    "significant_changes": "Significant changes since shipping",
}


def _arrival_html(scan: Scan, overlay_url: str, baseline_overlay_url: str | None) -> str:
    a = scan.arrival
    if a is None:
        return ""
    findings = "".join(f"<li>{escape(f)}</li>" for f in a.findings)
    visual = f"<p>{escape(a.visual.summary)}</p>" if a.visual else ""
    before = (
        f'<figure><img src="{escape(baseline_overlay_url)}" alt="Seller photo before shipping">'
        "<figcaption>Before shipping</figcaption></figure>"
        if baseline_overlay_url
        else ""
    )
    return f"""<h2>Arrival Check</h2>
<div class="verdict {escape(a.verdict)}">{VERDICT_LABELS[a.verdict]}</div>
<div class="meta">Compared with seller scan {escape(a.baseline_id[:8])} &middot; {a.days_in_transit:.1f} days in transit</div>
{visual}<ul>{findings}</ul>
<div class="pair">{before}<figure><img src="{escape(overlay_url)}" alt="Buyer photo on arrival"><figcaption>On arrival</figcaption></figure></div>"""


TRACKING_LABELS = {
    "sam-3.1-video": "every leaf tracked across the video by SAM 3.1",
    "local-keyframe": "leaves counted on the best keyframe (no tracking)",
}


def _video_html(scan: Scan, video_url: str | None) -> str:
    v = scan.video
    if v is None or video_url is None:
        return ""
    tracking = TRACKING_LABELS.get(v.tracking, v.tracking)
    return f"""<h2>Proof of Plant</h2>
<div>Recorded video &middot; {v.duration_s:.1f} s &middot; <b>{v.leaves_tracked} leaves</b>, {escape(tracking)}</div>
<video src="{escape(video_url)}" controls muted playsinline preload="metadata"></video>
<div class="meta">SHA-256 {escape(v.sha256[:16])}&hellip; &middot; <a href="{escape(video_url)}">original recording</a></div>"""


def share_summary(scan: Scan) -> tuple[str, str]:
    """Title and one-line description for link previews (WhatsApp, X, Telegram)."""
    species = scan.species_hint or "Aroid"
    if scan.arrival:
        title = f"{species}: {VERDICT_LABELS[scan.arrival.verdict]}"
    elif scan.variegation:
        title = f"{species}: {scan.variegation.weighted_pct:.0f}% variegation ({scan.variegation.grade.replace('_', ' ')})"
    else:
        title = f"{species}: Kanopi Lens Grading Card"
    parts = [f"{len(scan.leaves)} leaves"]
    if scan.video:
        parts.append(f"verified by a {scan.video.duration_s:.0f} s video")
    if scan.health.score is not None:
        parts.append(f"health {scan.health.score}/100")
    return title, " · ".join(parts) + ". Graded by Kanopi Lens with Meta SAM 3.1."


def render_card_html(
    scan: Scan, overlay_url: str, baseline_overlay_url: str | None = None, video_url: str | None = None
) -> str:
    v = scan.variegation
    variegation = (
        f"<b>{v.weighted_pct:.1f}%</b> &middot; grade <b>{escape(v.grade)}</b> "
        f"(range {v.min_pct:.0f}&ndash;{v.max_pct:.0f}%)"
        if v
        else "No leaves detected"
    )
    health = scan.health
    score = f"<b>{health.score}</b>/100" if health.score is not None else "not available"
    issues = "".join(
        f"<li><b>{escape(i.name)}</b> ({escape(i.severity)}, {i.confidence:.0%})"
        f"{f' on leaf #{i.leaf_index}' if i.leaf_index is not None else ''}: "
        f"{escape(i.recommendation)}</li>"
        for i in health.issues
    ) or ("<li>No visible issues</li>" if health.score is not None else "")
    species = escape(scan.species_hint) if scan.species_hint else "Unspecified aroid"
    share_title, share_text = (escape(t) for t in share_summary(scan))
    arrival = _arrival_html(scan, overlay_url, baseline_overlay_url)
    proof = _video_html(scan, video_url)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{species} · Kanopi Lens Grading Card</title>
<meta name="description" content="{share_text}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Kanopi Lens">
<meta property="og:title" content="{share_title}">
<meta property="og:description" content="{share_text}">
<meta property="og:image" content="{escape(overlay_url)}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{share_title}">
<meta name="twitter:description" content="{share_text}">
<meta name="twitter:image" content="{escape(overlay_url)}">
<style>
:root {{ --bg:#f6f7f4; --card:#fff; --fg:#1d2a1f; --muted:#5f6b61; --accent:#2e7d4f; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#111612; --card:#1a211c; --fg:#e8efe9; --muted:#9aa89d; --accent:#6fcf97; }} }}
body {{ margin:0; padding:16px; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; }}
.card {{ max-width:640px; margin:0 auto; background:var(--card); border-radius:16px; overflow:hidden; box-shadow:0 2px 12px rgba(0,0,0,.08); }}
img {{ width:100%; display:block; }}
.body {{ padding:20px; }}
h1 {{ margin:0 0 4px; font-size:1.3rem; }}
.meta, .disclaimer {{ color:var(--muted); font-size:.85rem; }}
h2 {{ font-size:1rem; color:var(--accent); margin:18px 0 6px; }}
ul {{ padding-left:20px; margin:0; }}
.verdict {{ display:inline-block; font-weight:600; padding:4px 12px; border-radius:999px; margin-bottom:6px; }}
.verdict.consistent {{ background:#dcebdf; color:#2e7d4f; }}
.verdict.minor_changes {{ background:#f5ebcf; color:#8a6508; }}
.verdict.significant_changes {{ background:#f6ddd6; color:#a33b26; }}
.pair {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:12px; }}
figure {{ margin:0; }}
video {{ width:100%; border-radius:10px; margin-top:8px; background:#000; }}
a {{ color:var(--accent); }}
figure img {{ border-radius:10px; }}
figcaption {{ color:var(--muted); font-size:.8rem; margin-top:4px; }}
</style></head>
<body><div class="card">
<img src="{escape(overlay_url)}" alt="Segmented leaves with variegation labels">
<div class="body">
<h1>{species}</h1>
<div class="meta">Scan {escape(scan.id[:8])} &middot; {scan.created_at:%Y-%m-%d} &middot; {len(scan.leaves)} leaves</div>
{arrival}
{proof}
<h2>Variegation</h2><div>{variegation}</div>
<h2>Health</h2><div>Score {score}</div><ul>{issues}</ul>
<p class="disclaimer">{escape(scan.disclaimer)}</p>
</div></div></body></html>"""
