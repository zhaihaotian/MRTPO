"""Render a replay record (written by replay.py) as a self-contained HTML page, and optionally as PNG figures.

Every response token is shaded by q = sqrt(JS / ln 2) of one reward. One colour scale is shared by all rows of a page,
from 0 (white) to the largest q on the page (full colour), so rows and responses can be compared.
"""

from __future__ import annotations

import html
import warnings
from pathlib import Path

TEAL = (87, 152, 137)


def _page_max(record):
    values = [v for r in record["responses"] for row in r["rewards"] if row["q"] for v in row["q"]]
    return max(values, default=0.0) or 1.0


def _token(token, q, w, js, index, scale):
    shown = html.escape(token).replace("\n", "↵\n")
    if q is None:
        return f'<span title="token {index} · not replayed · w 1">{shown}</span>'
    alpha = min(q / scale, 1.0)
    return (f'<span style="background:rgba({TEAL[0]},{TEAL[1]},{TEAL[2]},{alpha:.3f})" '
            f'title="token {index} · q {q:.4f} · w {w:.3f} · JS {js:.5f}">{shown}</span>')


def _row(tokens, row, scale):
    name = html.escape(row["name"])
    if not row["replayed"]:
        strip = "".join(_token(t, None, 1.0, 0.0, i, scale) for i, t in enumerate(tokens))
        return (f'<div class="row"><div class="head"><b>{name}</b> <span class="stat">empty block: not replayed, '
                f'uniform weights</span></div><div class="strip">{strip}</div></div>')
    q, w, js = row["q"], row["w"], row["js"]
    stat = f"mean q {sum(q) / len(q):.4f} · max q {max(q):.4f} · max w {max(w):.3f}"
    strip = "".join(_token(t, q[i], w[i], js[i], i, scale) for i, t in enumerate(tokens))
    detail = (f"<details><summary>block</summary><pre>{html.escape(row['block'])}</pre></details>" if row["block"]
              else "")
    detail += f"<details><summary>P_minus_k (rendered)</summary><pre>{html.escape(row['p_minus'])}</pre></details>"
    return (f'<div class="row"><div class="head"><b>{name}</b> <span class="stat">{stat}</span></div>'
            f'{detail}<div class="strip">{strip}</div></div>')


def render_html(record):
    scale = _page_max(record)
    sections = []
    for i, response in enumerate(record["responses"], start=1):
        rows = "".join(_row(response["tokens"], row, scale) for row in response["rewards"])
        sections.append(
            f'<section><h2>Response {i} <small>({response["source"]}, {len(response["tokens"])} tokens)</small></h2>'
            f'<details><summary>P_all (rendered)</summary><pre>{html.escape(response["p_all"])}</pre></details>'
            f"{rows}</section>"
        )
    bar = "".join(f'<span style="background:rgba({TEAL[0]},{TEAL[1]},{TEAL[2]},{a / 20:.2f})"></span>'
                  for a in range(21))
    title = html.escape(record["name"] or "MRTPO replay")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root{{color-scheme:light}}
body{{font:14px/1.5 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif;max-width:1100px;margin:24px auto;
padding:0 16px;color:#202633;background:#fff}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 6px}} small,.stat,.meta{{color:#687483}}
section{{border-top:1px solid #d4dbe3}}
pre{{white-space:pre-wrap;word-break:break-word;background:#f7f8fa;border:1px solid #e6eaef;padding:8px;
font-size:12px;margin:4px 0}}
summary{{cursor:pointer;color:#687483;font-size:12.5px}}
.row{{margin:14px 0}} .head{{margin-bottom:4px}}
.strip{{font:12.5px/1.7 ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap;word-break:break-word;
border:1px solid #e6eaef;padding:8px;border-radius:4px}}
.strip span{{border-radius:2px}}
.bar{{display:inline-flex;vertical-align:middle;border:1px solid #d4dbe3;margin:0 6px}}
.bar span{{width:8px;height:12px}}
</style></head><body>
<h1>{title}</h1>
<p class="meta">model {html.escape(record["model"])} · β {record["beta"]} · temperature {record["temperature"]}</p>
<p>Each row is one reward. A token's shade is q = sqrt(JS / ln 2), where JS compares the model's next-token
distributions at that position under P_all (every block added) and P_minus_k (without the reward's block). Hover a
token for q, its weight w = (1 − β) + β·q / mean(q) and JS. q scale:
0<span class="bar">{bar}</span>{scale:.3f} (largest q on this page).</p>
<details><summary>P_0, the rollout prompt (rendered)</summary><pre>{html.escape(record["p0"])}</pre></details>
{"".join(sections)}
</body></html>
"""


def render_png(record, stem, columns=100):
    """One PNG per response: one panel per reward, tokens on a monospace grid shaded by q. Returns the paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    scale = _page_max(record)
    font, char_w, line_h, head_h = 8.0, 8.0 * 0.6 / 72, 8.0 * 1.45 / 72, 0.28
    width = columns * char_w + 0.3
    stem = Path(stem)
    paths = []
    for index, response in enumerate(record["responses"], start=1):
        lines, col, line = [], 0, []
        for t_index, token in enumerate(response["tokens"]):
            for piece_index, piece in enumerate(token.replace("\t", "    ").split("\n")):
                if piece_index:
                    line.append((col, "↵", t_index))
                    lines.append(line)
                    line, col = [], 0
                while piece:
                    if col >= columns:
                        lines.append(line)
                        line, col = [], 0
                    take = piece[:columns - col]
                    line.append((col, take, t_index))
                    col += len(take)
                    piece = piece[len(take):]
        lines.append(line)
        panel_h = head_h + len(lines) * line_h + 0.12
        height = panel_h * len(response["rewards"]) + 0.1
        fig = plt.figure(figsize=(width, height))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis("off")
        for r_index, row in enumerate(response["rewards"]):
            top = 0.1 + r_index * panel_h
            label = row["name"] if len(row["name"]) <= 110 else row["name"][:107] + "..."
            note = (f"mean q {sum(row['q']) / len(row['q']):.4f}, max q {max(row['q']):.4f}" if row["replayed"]
                    else "empty block: not replayed")
            ax.text(0.15, top + 0.12, f"{label}   ({note})", fontsize=font, fontweight="bold", va="center",
                    color="#202633", parse_math=False)
            for l_index, items in enumerate(lines):
                y = top + head_h + l_index * line_h
                for col, text, t_index in items:
                    q = row["q"][t_index] if row["replayed"] else 0.0
                    if q > 0:
                        ax.add_patch(Rectangle((0.15 + col * char_w, y - line_h / 2), len(text) * char_w, line_h,
                                               lw=0, facecolor=[c / 255 for c in TEAL], alpha=min(q / scale, 1.0)))
                    ax.text(0.15 + col * char_w, y, text, fontsize=font, family="monospace", va="center",
                            color="#202633", parse_math=False)
        path = stem.parent / f"{stem.name}_response{index}.png"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Glyph .* missing from font")  # shown as boxes; see the HTML
            fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths
