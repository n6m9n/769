# Slide deck generator

- `small_figs.py` renders the three small bar charts (coverage, hypervolume, solver times) into `report/figures/`
  (images instead of native charts: Keynote does not render pptxgenjs charts).
- `build_deck.js` builds `report/presentation.pptx` with pptxgenjs (structured deck: theme, layouts, sections,
  speaker notes). Needs `pptxgenjs react react-dom react-icons sharp`, and `SKILL_DIR` pointing at a directory
  with `scripts/apply_theme.js`.
- `report/presentation.pdf` is a Keynote export of the deck.
