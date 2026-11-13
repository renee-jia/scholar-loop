import sys
from pathlib import Path

THEMES = {
 "light": dict(panel="#f6f7fb", panel_stroke="#e2e4ef", node="#ffffff", node_stroke="#d3d6e4", text="#1b1d2a",
               muted="#5d6179", line="#8a8fa8", band="#f6f7fb", pill="#ffffff",
               acc=["#6246ea", "#0b9a86", "#d2741f"], tint=["#ece8ff", "#dcf5f0", "#fdeedd"], on_acc="#ffffff"),
 "dark": dict(panel="#151722", panel_stroke="#2a2e40", node="#1e2130", node_stroke="#383d54", text="#eceef8",
              muted="#a3a8c0", line="#737995", band="#151722", pill="#1e2130",
              acc=["#a08cff", "#3fd1bc", "#f2a65a"], tint=["#2a2550", "#143a36", "#43301c"], on_acc="#12131c"),
}
W, PW, GAP, X0 = 1200, 354, 49, 20
NW, NH, PITCH, Y0 = 280, 62, 90, 86
COLS = [
 ("IDEATION", "you choose what is worth running", [
   ("Topic", "one sentence from you", "in"),
   ("Literature review", "arXiv + OpenAlex, every citation checked", ""),
   ("Ideas", "tied to open gaps and to what was tried", ""),
   ("Tournament", "each pair judged twice, order swapped", ""),
   ("Pick an idea", "or push back, or bring your own", "you")]),
 ("EXPERIMENTS", "runs unattended", [
   ("Propose a run", "control or treatment of your idea", ""),
   ("Debate gate", "three critics vote: worth the compute?", ""),
   ("Funnel", "smoke → verify → full, most ideas die early", ""),
   ("Frozen scorer", "the only source of a reported number", ""),
   ("Reflect and calibrate", "lessons kept, predictions scored", "")]),
 ("REVIEW + WRITING", "you approve the argument", [
   ("Scientific review", "four critics question every claim", ""),
   ("Approve the release", "bound to a hash of the evidence", "you"),
   ("Staged writing", "outline → key points → draft", "you"),
   ("Reviewer panel", "three reviewers score the draft", ""),
   ("Paper + evidence map", "every number traces to a run", "out")]),
]
CHECKS = ["citations exist", "numbers are measured", "edits stay in train.py",
          "test set stays sealed", "approval tracks evidence", "every attempt logged"]
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"

def build(theme):
    t = THEMES[theme]
    px = [X0 + i * (PW + GAP) for i in range(3)]
    nx = [p + (PW - NW) // 2 + 12 for p in px]          # nodes sit slightly right: room for the loop
    ny = [Y0 + r * PITCH for r in range(5)]
    panel_h = ny[-1] + NH + 26 - 20
    band_y = 20 + panel_h + 22
    H = band_y + 96 + 20
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}" role="img" '
         f'aria-label="ScholarLoop workflow: ideation, experiments, review and writing, with the checks enforced in code">',
         "<defs>"]
    for name, color in [("n", t["line"])] + [(f"a{i}", c) for i, c in enumerate(t["acc"])]:
        o.append(f'<marker id="{name}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                 f'<path d="M1 1.5 L8.5 5 L1 8.5" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></marker>')
    o.append("</defs>")
    for i, (title, sub, nodes) in enumerate(COLS):
        acc, tint = t["acc"][i], t["tint"][i]
        o.append(f'<rect x="{px[i]}" y="20" width="{PW}" height="{panel_h}" rx="18" fill="{t["panel"]}" stroke="{t["panel_stroke"]}"/>')
        o.append(f'<rect x="{px[i] + 24}" y="42" width="5" height="26" rx="2.5" fill="{acc}"/>')
        o.append(f'<text x="{px[i] + 40}" y="54" font-size="14" font-weight="700" letter-spacing="1.4" fill="{t["text"]}">{title.replace("+", "&#43;")}</text>')
        o.append(f'<text x="{px[i] + 40}" y="70" font-size="12" fill="{t["muted"]}">{sub}</text>')
        for r, (name, desc, kind) in enumerate(nodes):
            x, y = nx[i], ny[r]
            if r:                                                    # arrow from the node above
                o.append(f'<path d="M{x + NW / 2} {ny[r - 1] + NH + 3} V{y - 4}" stroke="{t["line"]}" stroke-width="1.6" fill="none" marker-end="url(#n)"/>')
            you = kind == "you"
            fill = tint if you else t["node"]
            stroke = acc if (you or kind in ("in", "out")) else t["node_stroke"]
            dash = ' stroke-dasharray="5 4"' if kind in ("in", "out") else ""
            o.append(f'<rect x="{x}" y="{y}" width="{NW}" height="{NH}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="{1.6 if you else 1.2}"{dash}/>')
            o.append(f'<text x="{x + 16}" y="{y + 26}" font-size="15" font-weight="650" fill="{t["text"]}">{name}</text>')
            o.append(f'<text x="{x + 16}" y="{y + 45}" font-size="12" fill="{t["muted"]}">{desc}</text>')
            if you:
                o.append(f'<rect x="{x + NW - 54}" y="{y + 11}" width="42" height="20" rx="10" fill="{acc}"/>')
                o.append(f'<text x="{x + NW - 33}" y="{y + 25}" font-size="10.5" font-weight="700" letter-spacing="0.8" text-anchor="middle" fill="{t["on_acc"]}">YOU</text>')
    # the loop inside EXPERIMENTS: reflect -> propose
    lx = nx[1] - 24
    o.append(f'<path d="M{nx[1] - 3} {ny[4] + NH / 2} H{lx + 10} Q{lx} {ny[4] + NH / 2} {lx} {ny[4] + NH / 2 - 10} V{ny[0] + 46 + 10} Q{lx} {ny[0] + 46} {lx + 10} {ny[0] + 46} H{nx[1] - 4}" '
             f'stroke="{t["acc"][1]}" stroke-width="1.6" fill="none" marker-end="url(#a1)"/>')
    o.append(f'<text transform="translate({lx - 8} {(ny[0] + ny[4] + NH) / 2 + 20}) rotate(-90)" font-size="11.5" text-anchor="middle" fill="{t["acc"][1]}">next round, until the governor stops it</text>')
    # pick -> propose (handoff)
    g1 = px[0] + PW + GAP / 2
    o.append(f'<path d="M{nx[0] + NW + 3} {ny[4] + NH / 2} H{g1 - 10} Q{g1} {ny[4] + NH / 2} {g1} {ny[4] + NH / 2 - 10} V{ny[0] + 18 + 10} Q{g1} {ny[0] + 18} {g1 + 10} {ny[0] + 18} H{nx[1] - 4}" '
             f'stroke="{t["acc"][0]}" stroke-width="1.8" fill="none" marker-end="url(#a0)"/>')
    o.append(f'<text transform="translate({g1 - 9} {(ny[1] + ny[4]) / 2 + 30}) rotate(-90)" font-size="11.5" text-anchor="middle" fill="{t["acc"][0]}">handoff.json</text>')
    # scorer -> scientific review (evidence)
    g2 = px[1] + PW + GAP / 2
    o.append(f'<path d="M{nx[1] + NW + 3} {ny[3] + NH / 2} H{g2 - 10} Q{g2} {ny[3] + NH / 2} {g2} {ny[3] + NH / 2 - 10} V{ny[0] + 44 + 10} Q{g2} {ny[0] + 44} {g2 + 10} {ny[0] + 44} H{nx[2] - 4}" '
             f'stroke="{t["acc"][1]}" stroke-width="1.8" fill="none" marker-end="url(#a1)"/>')
    o.append(f'<text transform="translate({g2 - 9} {(ny[1] + ny[3]) / 2 + 30}) rotate(-90)" font-size="11.5" text-anchor="middle" fill="{t["acc"][1]}">ledger + registry</text>')
    # scientific review -> propose (more experiments), dashed
    o.append(f'<path d="M{nx[2] - 3} {ny[0] + 16} H{nx[1] + NW + 5}" stroke="{t["acc"][2]}" stroke-width="1.6" stroke-dasharray="6 4" fill="none" marker-end="url(#a2)"/>')
    o.append(f'<text x="{(nx[2] + nx[1] + NW) / 2}" y="{ny[0] + 8}" font-size="11.5" text-anchor="middle" fill="{t["acc"][2]}">more experiments</text>')
    # the band of checks
    o.append(f'<rect x="{X0}" y="{band_y}" width="{W - 2 * X0}" height="96" rx="18" fill="{t["band"]}" stroke="{t["panel_stroke"]}"/>')
    o.append(f'<text x="{X0 + 24}" y="{band_y + 32}" font-size="14" font-weight="700" letter-spacing="1.4" fill="{t["text"]}">ENFORCED IN CODE</text>')
    o.append(f'<text x="{X0 + 196}" y="{band_y + 32}" font-size="12" fill="{t["muted"]}">a model output that fails a check is rejected and retried</text>')
    widths = [len(c) * 6.1 + 40 for c in CHECKS]
    gap = (W - 2 * X0 - 48 - sum(widths)) / (len(CHECKS) - 1)
    x = X0 + 24
    for c, w in zip(CHECKS, widths):
        o.append(f'<rect x="{x:.1f}" y="{band_y + 48}" width="{w:.1f}" height="30" rx="15" fill="{t["pill"]}" stroke="{t["node_stroke"]}"/>')
        o.append(f'<path d="M{x + 13:.1f} {band_y + 63.5} l3.6 3.6 l6.4 -7.4" stroke="{t["acc"][1]}" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
        o.append(f'<text x="{x + 29:.1f}" y="{band_y + 67.5}" font-size="11.5" fill="{t["text"]}">{c}</text>')
        x += w + gap
    o.append("</svg>")
    return "\n".join(o), H

out = Path(sys.argv[1])
for theme in THEMES:
    svg, H = build(theme)
    (out / f"workflow-{theme}.svg").write_text(svg + "\n")
print("height", H)
