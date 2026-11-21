# Slide deck generator

- `slide_figs.py` draws slide-sized figures (true 12–14 pt labels) from the result JSONs into `report/slide_figures/`.
- `tex.js` typesets LaTeX with MathJax (TeX → SVG) and rasterises at 1200 dpi, so every equation and symbol is
  real LaTeX and renders identically in PowerPoint, Keynote and Google Slides.
- `build_deck.js` builds `report/presentation.pptx` with pptxgenjs (theme, layouts, sections, speaker notes).
  Needs `pptxgenjs react react-dom react-icons sharp mathjax-full`, and `SKILL_DIR` pointing at a directory with
  `scripts/apply_theme.js`.
- `report/presentation.pdf` is a Keynote export of the deck.
