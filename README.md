# MRTPO project page

Static project page for *Each Reward to Its Relevant Tokens: Reward-Specific Token-Level Credit Assignment for Multi-Reward RL*.

- `index.html` — the page
- `static/figures/` — paper figures converted from PDF to SVG (`pdftocairo -svg`)
- `static/img/og_preview.png` — social preview image made from the teaser figure
- `static/css/style.css` — styles (light theme, one teal accent; no template)
- `static/js/main.js` — BibTeX copy button and KaTeX rendering

Preview locally:

    python3 -m http.server 8843 --directory .

To publish with GitHub Pages, put these files on a `gh-pages` branch (or a `docs/` folder) of the repo and enable Pages in the repo settings.
