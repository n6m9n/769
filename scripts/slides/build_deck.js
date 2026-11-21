// Final presentation: Multi-objective vocabulary selection for multilingual tokenization (CS 769)
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");
const fa = require("react-icons/fa");
const { applyTheme } = require(process.env.SKILL_DIR + "/scripts/apply_theme.js");

const FIG = "/Users/naman/Desktop/769/motok/report/figures/";
const OUT = "/Users/naman/Desktop/769/motok/report/presentation.pptx";

// ---------------------------------------------------------------- theme
const THEME = {
  name: "Tokenizer Pareto",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1E1E24", lt1: "FFFFFF", dk2: "1B1F3B", lt2: "F2F3F7",
    accent1: "E09F3E",  // saffron (accent)
    accent2: "2A9D8F",  // teal
    accent3: "2A78D6",  // LP blue (matches figures)
    accent4: "EB6834",  // BPE orange (matches figures)
    accent5: "6250D6",  // violet (Parity-aware BPE in figures)
    accent6: "8A8D99",  // muted grey
    hlink: "2A78D6", folHlink: "6250D6",
  },
};
const H = THEME.colors; // hex for hex-only options

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "Multi-objective vocabulary selection for multilingual tokenization";
pres.subject = "CS 769 final presentation";
pres.author = "CS 769 team";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333, X0 = 0.6, CW = 12.13;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.55, w: 11.7, h: 1.9, fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left", fontFace: THEME.headFontFace }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.55, w: 11.7, h: 0.8, fontSize: 20, color: C.accent1, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: X0, y: 0.32, w: CW, h: 0.8, fontSize: 32, bold: true, color: C.text2, valign: "middle", align: "left", fontFace: THEME.headFontFace }, text: "" } },
    { text: { text: "CS 769 · Multi-objective tokenization", options: { x: X0, y: 7.05, w: 6, h: 0.3, fontSize: 10, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.05, w: 0.55, h: 0.3, fontSize: 10, color: H.accent6, align: "right" },
});
pres.defineSlideMaster({
  title: "SECTION_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 2.4, w: 11.7, h: 0.6, fontSize: 18, color: C.accent1, bold: true }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 3.0, w: 11.7, h: 1.4, fontSize: 40, bold: true, color: C.background1, valign: "top", align: "left", fontFace: THEME.headFontFace }, text: "" } },
  ],
  slideNumber: { x: 12.2, y: 7.05, w: 0.55, h: 0.3, fontSize: 10, color: H.accent6, align: "right" },
});

// ---------------------------------------------------------------- helpers
async function iconPng(Icon, color = "FFFFFF", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Icon, { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).resize(size, size).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
const ICONS = {};
async function loadIcons() {
  const names = ["FaLanguage", "FaCompressArrowsAlt", "FaBalanceScale", "FaCut", "FaShareAlt", "FaDatabase", "FaTree",
    "FaProjectDiagram", "FaCheckCircle", "FaChartLine", "FaMicrochip", "FaFlask", "FaLightbulb", "FaExclamationTriangle",
    "FaCogs", "FaServer", "FaRandom", "FaBrain", "FaFileAlt", "FaSearch", "FaLayerGroup", "FaRulerHorizontal", "FaGlobeAsia"];
  for (const n of names) ICONS[n] = await iconPng(fa[n]);
}
function iconCircle(slide, name, x, y, d, fill) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: "icon circle " + name });
  const p = d * 0.26;
  slide.addImage({ data: ICONS[name], x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p, altText: name.replace("Fa", "") + " icon" });
}
function caption(slide, text, x, y, w, h = 0.35) {
  slide.addText(text, { x, y, w, h, fontSize: 11, italic: true, color: C.accent6, margin: 0, valign: "top", isTextBox: true, objectName: "caption" });
}
function body(slide, runs, x, y, w, h, opts = {}) {
  slide.addText(runs, { x, y, w, h, fontSize: 15, color: C.text1, valign: "top", margin: 0, paraSpaceAfter: 6, isTextBox: true, ...opts });
}
function bullets(items, size = 15) {
  return items.map((t, i) => {
    const runs = Array.isArray(t) ? t : [{ text: t }];
    return runs.map((r, j) => ({ text: r.text, options: { ...(r.options || {}), fontSize: size, ...(j === 0 ? { bullet: { indent: 16 } } : {}), ...(j === runs.length - 1 && i < items.length - 1 ? { breakLine: true } : {}) } }));
  }).flat();
}
function img(slide, file, x, y, w, alt) {
  const dims = { "fig_lm.png": [1879, 670], "fig_morph_front.png": [1877, 775], "fig_parity_front.png": [1877, 775], "fig_per_lang.png": [1426, 634],
    "fig_scaling.png": [1607, 580], "fig_shadow_prices.png": [1606, 580], "fig_train_fronts.png": [1877, 706], "fig_train_fronts_lp10k.png": [1876, 706],
    "fig_coverage.png": [1082, 779], "fig_hv.png": [1135, 778], "fig_solvers.png": [1093, 778] }[file];
  const h = w * dims[1] / dims[0];
  slide.addImage({ path: FIG + file, x, y, w, h, altText: alt, objectName: "figure " + file });
  return h;
}
function stat(slide, big, label, x, y, w, color) {
  slide.addText(big, { x, y, w, h: 0.75, fontSize: 36, bold: true, color, margin: 0, isTextBox: true, fontFace: THEME.headFontFace, objectName: "stat value" });
  slide.addText(label, { x, y: y + 0.75, w, h: 0.6, fontSize: 13, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "stat label" });
}
function card(slide, x, y, w, h, fill = C.background2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: "card" });
}
// "T_l", "x_{tok(k)}", "10^{-4}" -> runs with real subscripts / superscripts
function M(str, base = {}) {
  const runs = []; let buf = ""; let i = 0;
  const flush = () => { if (buf) { runs.push({ text: buf, options: { ...base } }); buf = ""; } };
  while (i < str.length) {
    const ch = str[i];
    if ((ch === "_" || ch === "^") && i + 1 < str.length && str[i + 1] !== " ") {
      flush();
      let t;
      if (str[i + 1] === "{") { const j = str.indexOf("}", i + 2); t = str.slice(i + 2, j); i = j + 1; }
      else { t = str[i + 1]; i += 2; }
      runs.push({ text: t, options: { ...base, [ch === "_" ? "subscript" : "superscript"]: true } });
    } else { buf += ch; i++; }
  }
  flush();
  return runs;
}
function eq(slide, text, num, x, y, w, size = 18) {
  slide.addText(M(text), { x, y, w: w - 0.7, h: 0.45, fontSize: size, color: C.text1, margin: 0, fontFace: "Cambria", valign: "middle", isTextBox: true, objectName: "equation " + num });
  slide.addText(`(${num})`, { x: x + w - 0.7, y, w: 0.7, h: 0.45, fontSize: 14, color: C.accent6, margin: 0, align: "right", valign: "middle", isTextBox: true, objectName: "equation number " + num });
}
const TBL_HEAD = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 12 } });
function table(slide, header, rows, x, y, w, colW, opts = {}) {
  const data = [header.map(TBL_HEAD)].concat(rows.map((r, ri) => r.map((c) => {
    const o = typeof c === "object" ? c : { text: String(c) };
    const txt = opts.math && /[_^]/.test(o.text) ? M(o.text) : o.text;
    return { text: txt, options: { fontSize: opts.size || 12, color: C.text1, fill: { color: ri % 2 ? C.background2 : C.background1 }, bold: !!o.bold, ...(o.options || {}) } };
  })));
  slide.addTable(data, { x, y, w, colW, border: { type: "solid", pt: 0.5, color: "D9DAE0" }, margin: [3, 5, 3, 5], valign: "middle", rowH: opts.rowH || 0.32, objectName: opts.name || "table" });
}
const CHART_BASE_UNUSED = {
  catAxisLabelColor: H.accent6, valAxisLabelColor: H.accent6, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11, dataLabelColor: H.dk1,
  valGridLine: { color: "E3E4E8", size: 0.5 }, catGridLine: { style: "none" }, titleFontFace: "+mn-lt", titleFontSize: 13, titleColor: H.dk2,
  showValue: true, dataLabelPosition: "outEnd", showLegend: false,
};

// ---------------------------------------------------------------- slides
async function build() {
  await loadIcons();
  let s;

  // ===== Section: Introduction
  pres.addSection({ title: "Introduction" });
  s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
  s.addText("Multi-objective vocabulary selection for multilingual tokenization", { placeholder: "title" });
  s.addText("Pareto fronts, optimality certificates, and what transfers to unseen text", { placeholder: "body" });
  s.addText("English · हिन्दी · मराठी · தமிழ் · Русский · Magyar", { x: 0.8, y: 0.75, w: 11.7, h: 0.5, fontSize: 18, color: C.accent6, margin: 0, isTextBox: true, objectName: "languages" });
  s.addText([
    { text: "[Name 1, Roll no.]   ·   [Name 2, Roll no.]", options: { breakLine: true, bold: true } },
    { text: "CS 769: Optimization in Machine Learning · Final presentation · [Date]" },
  ], { x: 0.8, y: 5.4, w: 11.7, h: 0.9, fontSize: 16, color: C.background1, margin: 0, isTextBox: true, objectName: "team details" });
  s.addNotes("Title. Introduce the team. One sentence: we turn tokenizer vocabulary selection into one multi-objective linear program and ask what it buys us over frequency-based tokenizers.");

  // Motivation
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("Frequency-based tokenizers tax some languages", { placeholder: "title" });
  stat(s, "1.60×", "token premium of Tamil over English (BPE, natural data mix)", X0, 1.55, 3.6, C.accent4);
  stat(s, "48.4 vs 30.1", "tokens per FLORES+ sentence: Tamil vs English, same content", X0, 3.05, 3.6, C.accent4);
  body(s, bullets([
    "More tokens = higher cost, shorter context, slower inference for the same text",
    "BPE / Unigram optimise one thing (compression) greedily; the dominant language of the training mix wins",
    "Splits also ignore morphology",
  ], 14), X0, 4.65, 3.7, 2.2);
  const hpl = img(s, "fig_per_lang.png", 4.65, 1.5, 8.05, "Bar chart of tokens per FLORES sentence per language for five tokenizers");
  caption(s, "Figure 1. Tokens per FLORES+ devtest sentence (identical content in every language) for BPE (α = 1, 0), Parity-aware BPE and our LP tokenizers. Lower is better.", 4.65, 1.5 + hpl + 0.1, 8.05, 0.6);
  s.addNotes("The same sentence costs Tamil 60% more tokens than English with standard BPE on the natural data mix. Temperature sampling (alpha = 0) helps, Parity-aware BPE helps more. Our question: can we optimise all of these goals jointly and see the trade-offs explicitly?");

  // Problem statement
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("Problem: one vocabulary, four competing objectives", { placeholder: "title" });
  const objs = [
    ["FaCompressArrowsAlt", "Compression", "fewest tokens over all text (data-weighted)", C.accent3],
    ["FaBalanceScale", "Parity", "lowest cost of the worst-served language", C.accent2],
    ["FaCut", "Morphology", "fewest token boundaries inside morphemes", C.accent1],
    ["FaShareAlt", "Sharing", "tokens reused across languages", C.accent5],
  ];
  objs.forEach(([ic, t, d, col], i) => {
    const x = X0 + i * 3.08;
    card(s, x, 1.45, 2.85, 2.3);
    iconCircle(s, ic, x + 0.25, 1.65, 0.75, col);
    s.addText(t, { x: x + 0.25, y: 2.5, w: 2.4, h: 0.4, fontSize: 18, bold: true, color: C.text2, margin: 0, isTextBox: true });
    s.addText(d, { x: x + 0.25, y: 2.9, w: 2.4, h: 0.75, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  });
  s.addText([
    { text: "Task. ", options: { bold: true, color: C.text2 } },
    { text: "Given pretoken counts in 6 languages, choose a vocabulary of m = 32 000 tokens; trace the Pareto front of these objectives and compare it with frequency-based tokenizers." },
  ], { x: X0, y: 4.05, w: CW, h: 0.75, fontSize: 16, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
  body(s, bullets([
    [{ text: "RQ1 ", options: { bold: true } }, { text: "Can all objectives be optimised in one LP, and does it stay near-integral at scale?" }],
    [{ text: "RQ2 ", options: { bold: true } }, { text: "What do the Pareto fronts and exchange rates (shadow prices) look like?" }],
    [{ text: "RQ3 ", options: { bold: true } }, { text: "Do the LP fronts beat BPE / Unigram, Parity-aware BPE and MorphBPE on held-out text?" }],
    [{ text: "RQ4 ", options: { bold: true } }, { text: "Which points give the best downstream language models (bits per byte)?" }],
  ], 15), X0, 4.95, CW, 1.95);
  s.addNotes("State the problem precisely: fixed vocabulary size, six languages, four objectives. Four research questions; the talk answers them in order.");

  // Related work
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("Related work and how we differ", { placeholder: "title" });
  table(s, ["Work", "What it does", "How we differ"], [
    [{ text: "ToaST (Schmidt et al. 2026)", bold: true }, "Vocabulary selection as an integer program over split trees; LP relaxation almost always integral", "Single objective → we add aggregate columns for 4 objectives, sweeps, certificates"],
    [{ text: "ConvexTok (Tempus et al. 2026)", bold: true }, "LP over all tokenizations, certified within ~1 %", "Single objective, monolingual focus"],
    [{ text: "Parity-aware BPE (Foroutan et al. 2026)", bold: true }, "Greedy merges for the currently worst language", "Greedy, one criterion → we trace the whole front with exchange rates"],
    [{ text: "MorphBPE (Asgari et al. 2026)", bold: true }, "Forbids merges across morpheme boundaries", "Hard constraint → morphology as a soft objective"],
    [{ text: "MUTANT (Rana et al. 2026)", bold: true }, "Indic tokenizer recipe; script-wise budgets; fertility / Rényi", "Evaluation protocol we follow"],
    [{ text: "MorphScore (Arnett et al. 2025)", bold: true }, "Gold morpheme boundaries for 70 languages", "Our gold evaluation (incl. hi / mr / ta)"],
  ], X0, 1.45, CW, [3.2, 4.6, 4.33], { size: 13, rowH: 0.62 });
  caption(s, "Table 1. Closest prior work. We build on ToaST's split-tree formulation and compare against Parity-aware BPE and MorphBPE as baselines.", X0, 6.05, CW);
  s.addNotes("ToaST is our starting point. Our contribution is the multi-objective layer on top of it, plus certificates and the analysis of what transfers to held-out text.");

  // ===== Section: Data & method
  pres.addSection({ title: "Data and method" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data and method" });
  s.addText("Data: six typologically diverse languages", { placeholder: "title" });
  table(s, ["Code", "Language", "Script", "Morphology", "Training text"], [
    ["en", "English", "Latin", "analytic", "200 MB (FineWeb)"],
    ["hi", "Hindi", "Devanagari", "fusional", "100 MB (FineWeb-2)"],
    ["mr", "Marathi", "Devanagari", "agglutinative", "50 MB"],
    ["ta", "Tamil", "Tamil", "highly agglutinative", "25 MB"],
    ["ru", "Russian", "Cyrillic", "fusional", "75 MB"],
    ["hu", "Hungarian", "Latin", "agglutinative", "50 MB"],
  ], X0, 1.45, 7.1, [0.8, 1.4, 1.5, 1.9, 1.5], { size: 13, rowH: 0.42 });
  caption(s, "Table 2. Deliberately imbalanced training mix (393 M characters) to create a fairness problem; pairs share scripts (hi–mr, en–hu).", X0, 4.5, 7.1, 0.55);
  const dataCards = [
    ["FaSearch", "Held-out", "FineWeb eval split: first 2 000 documents per language"],
    ["FaGlobeAsia", "Parallel", "FLORES+ devtest: 1 012 sentences, same content in all 6 languages"],
    ["FaCut", "Gold morphology", "MorphScore v2 (all 6, from UD) · SIGMORPHON 2022 (en/ru/hu)"],
    ["FaRulerHorizontal", "Content normalisation κ", "chars per English char on FLORES+ dev: ta 1.17, ru 1.10, hu 1.05"],
  ];
  dataCards.forEach(([ic, t, d], i) => {
    const y = 1.45 + i * 1.3;
    iconCircle(s, ic, 8.1, y, 0.6, C.accent2);
    s.addText(t, { x: 8.85, y: y - 0.05, w: 3.9, h: 0.35, fontSize: 15, bold: true, color: C.text2, margin: 0, isTextBox: true });
    s.addText(d, { x: 8.85, y: y + 0.3, w: 3.9, h: 0.75, fontSize: 13, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  });
  s.addNotes("Training data from FineWeb and FineWeb-2, imbalanced on purpose. Held-out FineWeb text, FLORES+ for content-matched comparisons, gold morphology for evaluation only. Morfessor (unsupervised) is the morphology training signal; we never evaluate on it.");

  // Split trees
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data and method" });
  s.addText("Background: split trees make tokenization linear", { placeholder: "title" });
  // tree diagram
  const node = (t, x, y, emit) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 1.55, h: 0.5, rectRadius: 0.08, fill: { color: emit ? C.accent1 : C.background2 }, line: { color: emit ? C.accent1 : "C9CBD3", width: 1 }, objectName: "tree node " + t });
    s.addText(t, { x, y, w: 1.55, h: 0.5, fontSize: 15, align: "center", valign: "middle", color: C.text1, bold: !!emit, margin: 0, fontFace: "Courier New", isTextBox: true });
  };
  const edge = (x1, y1, x2, y2) => s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: y1, w: Math.abs(x2 - x1), h: y2 - y1, flipH: x2 < x1, line: { color: "9A9DA8", width: 1.25 } });
  const T0 = 1.6;
  node("␣walking", 2.6, T0, false);
  node("␣walk", 1.2, T0 + 1.15, true); node("ing", 4.0, T0 + 1.15, true);
  node("␣w", 0.6, T0 + 2.3, false); node("alk", 1.9, T0 + 2.3, false); node("in", 3.4, T0 + 2.3, false); node("g", 4.7, T0 + 2.3, false);
  edge(3.375, T0 + 0.5, 1.975, T0 + 1.15); edge(3.375, T0 + 0.5, 4.775, T0 + 1.15);
  edge(1.975, T0 + 1.65, 1.375, T0 + 2.3); edge(1.975, T0 + 1.65, 2.675, T0 + 2.3);
  edge(4.775, T0 + 1.65, 4.175, T0 + 2.3); edge(4.775, T0 + 1.65, 5.475, T0 + 2.3);
  s.addText("… down to single characters (leaves)", { x: 0.6, y: T0 + 2.95, w: 5.6, h: 0.35, fontSize: 13, color: C.accent6, margin: 0, italic: true, isTextBox: true });
  caption(s, "Figure 2. Split tree of the pretoken “␣walking”. With ␣walk and ing in the vocabulary, inference emits the highlighted nodes: [␣walk][ing].", 0.6, 5.35, 5.7, 0.6);
  body(s, bullets([
    [{ text: "Tree per frequent pretoken: ", options: { bold: true } }, { text: "split where min(count(left), count(right)) is largest (n-gram counts)" }],
    [{ text: "Inference: ", options: { bold: true } }, { text: "walk down, emit the first in-vocabulary node on every root → leaf path" }],
    [{ text: "Trees never depend on the vocabulary, ", options: { bold: true } }, { text: "so “which nodes are emitted” is linear in node variables z" }],
    [{ text: "Unseen words ", options: { bold: true } }, { text: "get a tree built from the same counts at inference; unseen characters fall back to bytes" }],
  ], 15), 6.9, 1.6, 5.8, 4.5);
  s.addNotes("Split trees from ToaST. Each frequent pretoken gets a fixed binary tree built from counts only. Given a vocabulary, tokenization is: emit the first in-vocabulary node on each path. Because the tree is fixed, selecting the vocabulary becomes a linear program over node variables.");

  // LP formulation
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data and method" });
  s.addText("The vocabulary LP (core constraints)", { placeholder: "title" });
  card(s, X0, 1.45, 6.6, 3.85);
  s.addText(M("Variables:  x_i ∈ [0, 1] (token i in vocabulary),  z_k ∈ [0, 1] (node k emitted)"), { x: X0 + 0.3, y: 1.6, w: 6.1, h: 0.5, fontSize: 14, color: C.text2, bold: true, margin: 0, isTextBox: true });
  eq(s, "Σ_i x_i  =  m − 256", 1, X0 + 0.3, 2.25, 6.0);
  eq(s, "Σ_{k ∈ P(ℓ)} z_k  =  1        ∀ leaves ℓ", 2, X0 + 0.3, 2.95, 6.0);
  eq(s, "z_k  ≤  x_{tok(k)}        ∀ nodes k", 3, X0 + 0.3, 3.65, 6.0);
  eq(s, "x_i  =  1        ∀ base characters i", 4, X0 + 0.3, 4.35, 6.0);
  table(s, ["Symbol", "Meaning"], [
    ["m", "vocabulary size (32 000)"],
    ["256", "byte-fallback tokens reserved outside the LP"],
    ["P(ℓ)", "nodes on the path from leaf ℓ to the root"],
    ["tok(k)", "string of node k (several nodes can share a token)"],
    ["(2)", "every character covered exactly once"],
    ["(3)", "only vocabulary tokens can be emitted"],
  ], 7.55, 1.45, 5.18, [1.1, 4.08], { size: 13, rowH: 0.5, math: true });
  s.addText("LP relaxation of the integer program; ToaST observed it is almost always integral. Rounding + exact inference turn any LP solution into a real tokenizer.", { x: X0, y: 5.55, w: CW, h: 0.7, fontSize: 15, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
  s.addNotes("Two kinds of variables. Constraint 1 fixes the vocabulary size; 2 says each character is covered exactly once along its path; 3 links nodes to tokens; 4 keeps every character in the vocabulary so any text can be encoded.");

  // Aggregate columns & scalarisations
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data and method" });
  s.addText("Multi-objective layer: aggregate columns", { placeholder: "title" });
  eq(s, "T_l = (1000 κ_l / chars_l) · Σ_k c_{j(k),l} z_k", 5, X0, 1.45, 7.0, 17);
  eq(s, "M_l = (1000 κ_l / chars_l) · Σ_k c_{j(k),l} μ_k z_k", 6, X0, 2.0, 7.0, 17);
  eq(s, "Ttot = Σ_l π_l T_l ,    Mtot = Σ_l π_l M_l", 7, X0, 2.55, 7.0, 17);
  eq(s, "T_l ≤ t    ∀ l      (t = worst-language cost)", 8, X0, 3.1, 7.0, 17);
  table(s, ["Scalarisation", "Objective / constraint", "Changes only"], [
    ["Weighted sum", "min Ttot + λ t + ν Mtot", "costs"],
    ["ε-constraint (parity)", "min Ttot  s.t.  t ≤ τ", "1 bound"],
    ["ε-constraint (morph)", "min Ttot  s.t.  Mtot ≤ ε", "1 bound"],
    ["Min-max", "min t + δ Ttot", "costs"],
  ], X0, 3.85, 7.0, [2.2, 3.4, 1.4], { size: 13, rowH: 0.4 });
  caption(s, "Table 3. Every scalarisation touches only the costs or bounds of a few aggregate columns.", X0, 5.95, 7.0);
  table(s, ["Symbol", "Meaning"], [
    ["T_l", "tokens / 1000 English-equiv. chars, language l"],
    ["M_l", "spurious morpheme boundaries, same units"],
    ["κ_l", "content normalisation (FLORES+)"],
    ["c_{j,l}", "count of pretoken j in language l"],
    ["μ_k", "1 if node k starts inside a morpheme"],
    ["π_l", "data share of language l"],
    ["λ, ν, δ", "weights;  τ, ε: bounds"],
  ], 7.95, 1.45, 4.78, [1.15, 3.63], { size: 12, rowH: 0.4, math: true });
  card(s, 7.95, 4.95, 4.78, 1.55, C.text2);
  s.addText([
    { text: "Why this design: ", options: { bold: true, color: C.accent1 } },
    { text: "bound change → old basis stays dual feasible (warm dual simplex); cost change → primal feasible (warm primal). Reduced cost of a bounded column = dTtot / dτ, the exchange rate between objectives.", options: { color: C.background1 } },
  ], { x: 8.15, y: 5.05, w: 4.4, h: 1.35, fontSize: 13, margin: 0, valign: "middle", isTextBox: true });
  s.addNotes("Each objective is a column defined by one equality row. Then every scalarisation only changes costs or bounds on these few columns: sweeps warm-start, and reduced costs give exchange rates between objectives for free.");

  // Pipeline
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data and method" });
  s.addText("Pipeline", { placeholder: "title" });
  const steps = [
    ["FaDatabase", "1  Data", "FineWeb / FineWeb-2, FLORES+, gold morphology"],
    ["FaLayerGroup", "2  Count", "GPT-4o regex pretokens, Morfessor per language"],
    ["FaTree", "3  Trees", "head trees + sampled tail trees (SAA)"],
    ["FaProjectDiagram", "4  LP sweeps", "3 anchors ∥, 5 sweeps ∥, warm-started (HiGHS)"],
    ["FaCheckCircle", "5  Round + certify", "exact inference, bound shading"],
    ["FaChartLine", "6  Evaluate", "held-out, FLORES+, gold morphology, hypervolume"],
    ["FaBrain", "7  Small LMs", "35 M-param GPT per tokenizer, 3 × A6000"],
  ];
  steps.forEach(([ic, t, d], i) => {
    const row = i < 4 ? 0 : 1, col = row === 0 ? i : i - 4;
    const x = X0 + col * 3.1 + (row === 1 ? 1.55 : 0), y = 1.55 + row * 2.55;
    card(s, x, y, 2.75, 1.75);
    iconCircle(s, ic, x + 0.2, y + 0.2, 0.62, row === 0 ? C.accent3 : C.accent2);
    s.addText(t, { x: x + 0.95, y: y + 0.25, w: 1.7, h: 0.55, fontSize: 16, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true });
    s.addText(d, { x: x + 0.2, y: y + 0.95, w: 2.4, h: 1.0, fontSize: 13, color: C.text1, margin: 0, valign: "top", isTextBox: true });
    if (i < steps.length - 1 && i !== 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 2.78, y: y + 0.85, w: 0.3, h: 0.35, fill: { color: C.accent6 }, line: { color: C.accent6 }, objectName: "flow arrow" });
  });
  caption(s, "Figure 3. End-to-end pipeline (scripts 01–08 in the repository). Steps 1–6 run on 96 CPU cores; step 7 on GPUs.", X0, 6.35, CW);
  s.addNotes("Seven stages. The LP stage runs anchors in parallel and then each sweep in its own process, warm-started from saved simplex bases. Every Pareto point becomes a real tokenizer, evaluated on held-out text and in small language models.");

  // Certificates
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data and method" });
  s.addText("Certificates: rounding and bound shading", { placeholder: "title" });
  const certSteps = [
    ["Solve LP", "lower bound on the scalarised objective"],
    ["Round", "keep x ≈ 1, fill remaining slots by LP contribution (dual weights)"],
    ["Evaluate exactly", "real split-tree inference → upper bound; gap = certificate"],
    ["Bound shading (new)", "if the rounded vocabulary violates its ε-bound: tighten the bound by the overshoot, re-solve warm, re-round (≤ 6×)"],
  ];
  certSteps.forEach(([t, d], i) => {
    const y = 1.5 + i * 1.2;
    s.addShape(pres.shapes.OVAL, { x: X0, y, w: 0.6, h: 0.6, fill: { color: i === 3 ? C.accent1 : C.accent3 }, line: { color: i === 3 ? C.accent1 : C.accent3 } });
    s.addText(String(i + 1), { x: X0, y, w: 0.6, h: 0.6, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(t, { x: X0 + 0.85, y: y - 0.05, w: 6.0, h: 0.35, fontSize: 17, bold: true, color: C.text2, margin: 0, isTextBox: true });
    s.addText(d, { x: X0 + 0.85, y: y + 0.32, w: 6.0, h: 0.75, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  });
  card(s, 7.9, 1.45, 4.83, 4.9);
  stat(s, "14/20 → 0/123", "Pareto points whose rounded vocabulary violated its ε-bound (before → after)", 8.2, 1.65, 4.3, C.accent1);
  stat(s, "≤ 3·10⁻⁴", "largest integrality gap over all 123 points of the three main runs", 8.2, 3.15, 4.3, C.accent3);
  stat(s, "123 / 123", "points with LP ≤ integer value and monotone fronts (automatic checks)", 8.2, 4.65, 4.3, C.accent2);
  s.addNotes("Without the fix, rounding could overshoot the epsilon-bound, so the gap was not a valid certificate. Bound shading tightens the bound by the overshoot and re-solves warm; the LP value at the original bound stays the lower bound. Now every point carries a certificate.");

  // ===== Section: Experiments & results
  pres.addSection({ title: "Experiments and results" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Experiments and results" });
  s.addText("Experimental setup", { placeholder: "title" });
  const setup = [
    ["FaRandom", "Tokenizers (all 32 000 tokens)", ["BPE and Unigram × temperature α ∈ {1, .7, .5, .3, 0}", "Parity-aware BPE (authors' code)", "MorphBPE (reimplemented)", "LP: anchors + 4 sweeps per run (35–44 points)"]],
    ["FaFlask", "LP variants", ["head-only: 5k – 40k trees / language", "SAA: head + sampled tail trees (5k+5k, 10k+10k, 20k+20k)", "ablation: morph-first trees"]],
    ["FaServer", "Compute", ["96 cores, 1 TB RAM, 3 × RTX A6000", "HiGHS 1.16 dual / primal simplex", "LMs: 35.5 M params, 144 MB text, 4 epochs, 3 seeds"]],
  ];
  setup.forEach(([ic, t, items], i) => {
    const x = X0 + i * 4.1;
    card(s, x, 1.45, 3.85, 3.3);
    iconCircle(s, ic, x + 0.25, 1.65, 0.65, [C.accent3, C.accent1, C.accent2][i]);
    s.addText(t, { x: x + 1.05, y: 1.65, w: 2.65, h: 0.65, fontSize: 16, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true });
    body(s, bullets(items, 14), x + 0.25, 2.55, 3.4, 2.1);
  });
  card(s, X0, 5.05, CW, 1.25, C.text2);
  s.addText([
    { text: "Controlled comparison: ", options: { bold: true, color: C.accent1 } },
    { text: "every tokenizer uses the same GPT-4o pretokenizer, the same training text and a 32 000-token vocabulary, so differences come only from vocabulary selection. Every LP point is evaluated as a real tokenizer, never by its LP value alone.", options: { color: C.background1 } },
  ], { x: X0 + 0.3, y: 5.12, w: CW - 0.6, h: 1.1, fontSize: 15, margin: 0, valign: "middle", isTextBox: true });
  s.addNotes("Thirteen baseline tokenizers, four families of LP runs, all at vocabulary size 32k with the same pretokenizer, so only vocabulary selection differs.");

  // Metrics
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Experiments and results" });
  s.addText("Evaluation metrics", { placeholder: "title" });
  eq(s, "T̂_l = 1000 κ_l · #tokens_l / #chars_l      (held-out text)", 9, X0, 1.5, 7.4, 17);
  eq(s, "Gini(T̂) = Σ_{a,b} |T̂_a − T̂_b| / (2 L² · mean(T̂))", 10, X0, 2.2, 7.4, 17);
  eq(s, "HV(P) = vol{ y : p ≤ y ≤ r for some p ∈ P } / vol[ideal, r]", 11, X0, 2.9, 7.4, 17);
  eq(s, "bpb = Σ_{tokens} −ln p(token) / (ln 2 · #UTF-8 bytes)", 12, X0, 3.6, 7.4, 17);
  table(s, ["Symbol", "Meaning"], [
    ["T̂_l", "held-out tokens per 1000 English-equivalent chars"],
    ["L", "number of languages (6)"],
    ["P, r", "set of tokenizers' objective vectors; reference point r = 1.02 × nadir"],
    ["bpb", "LM bits per byte, independent of the tokenizer"],
  ], X0, 4.35, 7.4, [1.2, 6.2], { size: 13, rowH: 0.42, math: true });
  caption(s, "Table 4. Symbols of equations (9)–(12).", X0, 6.5, 7.4);
  card(s, 8.4, 1.45, 4.33, 4.95);
  s.addText("Also reported", { x: 8.65, y: 1.6, w: 3.9, h: 0.4, fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true });
  body(s, bullets([
    "FLORES+ tokens per sentence (content-matched) and token premium vs English",
    "Gold morphology: boundary F1 and spurious boundaries per word (MorphScore, SIGMORPHON)",
    "Training-side LP values, gaps, solve times",
    "Worst-language LM bits per FLORES sentence",
  ], 14), 8.65, 2.1, 3.85, 4.1);
  s.addNotes("Hypervolume measures how much of objective space a front dominates, normalised to a box from the ideal to 1.02 times the nadir point. Bits per byte lets us compare LMs with different tokenizers fairly.");

  // Result 1: head-only fronts
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Experiments and results" });
  s.addText("Head-only LP: fronts look very favourable…", { placeholder: "title" });
  const h12 = img(s, "fig_train_fronts_lp10k.png", X0, 1.4, 8.6, "Training-side Pareto fronts of the head-only LP at 10k trees per language");
  caption(s, "Figure 4. Training-side fronts, head-only LP (10k trees / language, 769 k node variables). Line: LP relaxation (lower bound); circles: rounded vocabularies; squares: weighted-sum points.", X0, 1.4 + h12 + 0.1, 8.6, 0.6);
  stat(s, "−21 %", "worst-language cost for +8 % total tokens (min-max anchor)", 9.6, 1.5, 3.1, C.accent2);
  stat(s, "−16 %", "for only +0.4 % total tokens (parity sweep)", 9.6, 3.0, 3.1, C.accent2);
  stat(s, "≤ 3·10⁻⁴", "integrality gap over all 44 points; weighted sum = ε-constraint", 9.6, 4.5, 3.1, C.accent3);
  s.addNotes("On the training side the fronts look great: fairness is almost free at first. All points are near-integral, and the two scalarisations agree. This is what the interim presentation showed. But training-side numbers can mislead...");

  // Result 2: do not transfer
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Experiments and results" });
  s.addText("…but they do not transfer to held-out text", { placeholder: "title" });
  table(s, ["Tokenizer", "FineWeb T̂ mean / worst", "FLORES mean / worst"], [
    ["BPE α = 1", "308.9 / 372.1", "39.5 / 48.4"],
    ["BPE α = 0", "296.8 / 336.3", "37.8 / 41.2"],
    ["head-only LP, compression", "348.8 / 406.6", "44.7 / 51.7"],
    ["head-only LP, best parity point", "343.3 / 404.4", "44.0 / 49.4"],
    [{ text: "head-only LP, min-max", bold: true }, { text: "354.3 / 490.3", bold: true }, { text: "45.5 / 61.0", bold: true }],
  ], X0, 1.45, 7.2, [3.0, 2.2, 2.0], { size: 13, rowH: 0.45 });
  caption(s, "Table 5. Held-out cost (lower is better). The LP loses to BPE, and the fairest training-side point is the least fair held out.", X0, 4.3, 7.2, 0.55);
  card(s, X0, 5.0, 7.2, 1.45, C.text2);
  s.addText([
    { text: "Diagnosis: ", options: { bold: true, color: C.accent1 } },
    { text: "trees cover only 66–89 % of the text. The LP never sees the long tail, so it spends the vocabulary on whole frequent words — and buys “fairness” with whole words of the worst language.", options: { color: C.background1 } },
  ], { x: X0 + 0.25, y: 5.1, w: 6.7, h: 1.25, fontSize: 14, margin: 0, valign: "middle", isTextBox: true });
  img(s, "fig_coverage.png", 8.2, 1.45, 4.53, "Bar chart: share of text covered by trees per language, 66 to 89 percent");
  caption(s, "Figure 5. Share of pretoken occurrences that have a tree (head-only, 10k trees per language).", 8.2, 4.85, 4.53, 0.55);
  s.addNotes("Held out, the story flips: the LP is about 13% worse than BPE, and the min-max point is the least fair of all. The reason is coverage: the LP optimises only the head of the distribution.");

  // SAA + optimizer's curse
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Experiments and results" });
  s.addText("Fix: sample the tail — and avoid the optimizer's curse", { placeholder: "title" });
  eq(s, "Ĉ_l(x) = Σ_{j ∈ head} c_{j,l} n_j(x) + (M_l / S) Σ_{s=1..S} n_s(x)", 13, X0, 1.45, CW, 17);
  s.addText(M("n_j(x): tokens of pretoken j under vocabulary x;  M_l: tail mass of language l;  S: tail samples per language;  n_s(x): tokens of the s-th sampled tail pretoken (drawn ∝ count). Unbiased for any fixed x (sample average approximation, SAA)."), { x: X0, y: 1.95, w: CW, h: 0.55, fontSize: 13, color: C.accent6, margin: 0, isTextBox: true });
  table(s, ["Tail trees may use…", "en", "hi", "mr", "ta", "ru", "hu"], [
    ["any frequent n-gram", "232 / 269", "290 / 302", "293 / 312", "379 / 396", { text: "290 / 381", bold: true }, { text: "289 / 353", bold: true }],
    ["only tokens of head trees", "272 / 277", "284 / 288", "282 / 289", "341 / 351", "357 / 369", "332 / 340"],
  ], X0, 2.7, 8.0, [2.6, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], { size: 13, rowH: 0.45 });
  caption(s, "Table 6. LP estimate / exact cost on the full training distribution (5k head + 5k tail trees per language).", X0, 4.1, 8.0);
  body(s, bullets([
    [{ text: "Optimizer's curse: ", options: { bold: true } }, { text: "the LP picks the vocabulary with the same sample; each sampled tail word weighs like a top-2000 word, so the LP buys tokens for individual samples (Russian estimate 24 % too low)" }],
    [{ text: "Fix: ", options: { bold: true } }, { text: "tail trees only reweight the objective, never add candidate tokens → estimate within 1–4 %, LP 3× faster" }],
  ], 14), X0, 4.6, 8.0, 2.2);
  card(s, 8.95, 2.7, 3.78, 3.75);
  stat(s, "−8 %", "held-out tokens vs head-only, with half the trees", 9.2, 2.9, 3.3, C.accent3);
  stat(s, "0", "mismatches between LP model and real tokenizer (20 000 trees)", 9.2, 4.55, 3.3, C.accent2);
  s.addNotes("We add sampled tail pretokens weighted so the objective estimates all text. The naive version overfits the sample: the classic optimizer's curse of sample average approximation. Restricting tail trees to head candidates removes the bias almost completely.");

  // Result 3: fronts transfer
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Experiments and results" });
  s.addText("With the corrected objective, the fronts transfer", { placeholder: "title" });
  const h15 = img(s, "fig_parity_front.png", X0, 1.35, 8.75, "Held-out Pareto fronts: compression vs worst-language cost on FineWeb and FLORES");
  caption(s, "Figure 6. Held-out fronts. (a) FineWeb: data-weighted tokens/1000 chars vs worst language. (b) FLORES+: mean vs worst tokens per sentence. Dark blue: SAA 10k+10k LP; light blue: head-only LP.", X0, 1.35 + h15 + 0.08, 8.75, 0.6);
  stat(s, "−13 / −14 %", "worst-language cost along the front (FineWeb / FLORES), mean unchanged", 9.65, 1.45, 3.1, C.accent2);
  stat(s, "0.004", "Gini across languages at the min-max point (lowest of all tokenizers)", 9.65, 3.0, 3.1, C.accent3);
  stat(s, "0.08 → 0.75", "held-out hypervolume: head-only → SAA (BPE α-sweep 0.78)", 9.65, 4.55, 3.1, C.accent1);
  s.addNotes("Now the parity sweep holds on unseen text: the worst language gets 13 to 14 percent cheaper while the mean stays flat. The min-max tokenizer is the most even of all, and its hypervolume is close to BPE's.");

  // Comparison with baselines
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Experiments and results" });
  s.addText("Comparison with the baselines", { placeholder: "title" });
  table(s, ["Tokenizer", "FineWeb mean / worst", "Gini", "FLORES mean / worst", "Morph F1"], [
    ["SAA LP, compression", "318.6 / 373.7", "0.069", "40.8 / 49.0", "0.19"],
    ["SAA LP, parity point", "315.5 / 342.3", "0.047", "40.3 / 43.9", "0.19"],
    [{ text: "SAA LP, min-max", bold: true }, { text: "321.4 / 325.9", bold: true }, { text: "0.004", bold: true }, "40.9 / 42.3", "0.19"],
    ["BPE α = 1", "308.9 / 372.1", "0.070", "39.5 / 48.4", "0.27"],
    ["BPE α = 0", { text: "296.8", bold: true, options: {} }, "0.044", { text: "37.8", bold: true }, "0.29"],
    ["Parity-aware BPE", { text: "300.4 / 312.1", bold: true }, "0.018", { text: "38.1 / 38.4", bold: true }, "0.29"],
    ["MorphBPE", "328.0 / 366.3", "0.048", "42.2 / 47.7", { text: "0.54", bold: true }],
    ["Unigram α = 1", "349.2 / 379.9", "0.030", "44.2 / 46.1", "0.50"],
  ], X0, 1.45, 8.1, [2.4, 1.85, 0.85, 1.85, 1.15], { size: 13, rowH: 0.47 });
  caption(s, "Table 7. Held-out comparison (lower is better except Morph F1, MorphScore gold). BPE α = 0 worst-language FineWeb cost: 336.3; FLORES: 41.2.", X0, 5.8, 8.1, 0.55);
  img(s, "fig_hv.png", 9.0, 1.5, 3.73, "Bar chart: normalised hypervolume of LP runs and baselines");
  caption(s, "Figure 7. Normalised hypervolume, compression × worst language (eq. 11). Higher is better.", 9.0, 4.25, 3.73, 0.55);
  s.addNotes("Where the LP wins: evenness across languages and the worst-language FineWeb cost versus BPE. Where it loses: mean compression to BPE, absolute fairness to Parity-aware BPE, morphology to MorphBPE and Unigram. Hypervolume shows the LP closing most of the gap.");

  // Morphology
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Experiments and results" });
  s.addText("Morphology: a weak lever inside split trees", { placeholder: "title" });
  const h17 = img(s, "fig_morph_front.png", X0, 1.35, 8.75, "Held-out morphology fronts: compression vs spurious boundaries and boundary F1");
  caption(s, "Figure 8. Held-out gold morphology (MorphScore v2, six languages). (a) spurious boundaries per word, (b) boundary F1, against compression.", X0, 1.35 + h17 + 0.08, 8.75, 0.6);
  body(s, bullets([
    [{ text: "LP morph sweep: ", options: { bold: true } }, { text: "gold F1 0.19 → 0.25 at +17 % tokens" }],
    [{ text: "MorphBPE (0.54) and Unigram (0.50) ", options: { bold: true } }, { text: "stay far ahead" }],
    [{ text: "Why: ", options: { bold: true } }, { text: "the LP can only choose among pieces the count-based trees offer; morpheme-shaped pieces are often not nodes" }],
  ], 15), 9.65, 1.5, 3.1, 4.9);
  s.addNotes("The morphology objective works on the training side but moves gold morphology only modestly; the candidate set from count-based trees is the limit, as the ablation will confirm.");

  // LMs
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Experiments and results" });
  s.addText("Downstream LMs disagree with intrinsic metrics", { placeholder: "title" });
  const h18 = img(s, "fig_lm.png", X0, 1.35, 8.75, "Bits per byte of small language models per tokenizer, and worst-language bits per FLORES sentence");
  caption(s, "Figure 9. Left: bits per byte on held-out FineWeb (mean of 6 languages, ±sd over 3 seeds). Right: tokenizer FLORES tokens vs LM worst-language bits per FLORES sentence.", X0, 1.35 + h18 + 0.08, 8.75, 0.6);
  body(s, bullets([
    [{ text: "Best LM: MorphBPE ", options: { bold: true } }, { text: "1.110 bpb; most compressive BPE (α = 0) is the worst baseline (1.153)" }],
    [{ text: "SAA LP tokenizers: ", options: { bold: true } }, { text: "better compression than head-only LP, but worse LMs (1.19 vs 1.14), consistently over seeds" }],
    [{ text: "Fairer tokenizer ≠ fairer LM: ", options: { bold: true } }, { text: "worst-language bits per sentence varies only 2–6 %" }],
  ], 14), 9.65, 1.45, 3.1, 5.0);
  s.addNotes("35M-parameter GPTs trained on identical bytes. Differences are small, under 8 percent. Morphology-aligned tokenizers win, the most compressive one loses, and our SAA tokenizers give worse LMs, which we report as an open question. Tokenizer fairness barely changes LM fairness.");

  // Scaling
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Experiments and results" });
  s.addText("Optimisation at scale", { placeholder: "title" });
  const h19 = img(s, "fig_scaling.png", X0, 1.4, 7.6, "Solve time versus LP size and coverage versus trees per language");
  caption(s, "Figure 10. Left: cold anchor and warm sweep-point solve times vs LP size (head-only). Right: text covered by trees vs trees per language.", X0, 1.4 + h19 + 0.08, 7.6, 0.6);
  img(s, "fig_solvers.png", 8.5, 1.4, 4.23, "Bar chart: solve time of five HiGHS solvers on the same LP");
  caption(s, "Figure 11. HiGHS solvers on the same LP (0.36 M nodes); lower is better.", 8.5, 4.6, 4.23, 0.55);
  body(s, bullets([
    "Up to 3.4 M node variables; time ≈ n^1.4",
    "Min-max anchors cost 5–7× a compression solve",
    "Warm sweep points: 10–40 % of a cold solve",
  ], 14), X0, 5.6, CW, 1.2);
  s.addNotes("Dual simplex beat interior-point and PDLP on these LPs. Compression anchors are exactly integral at every size. Min-max anchors are the bottleneck; warm starts make the sweeps affordable.");

  // Ablation
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Experiments and results" });
  s.addText("Ablation: morph-first trees", { placeholder: "title" });
  s.addText("Split positions prefer Morfessor boundaries: hierarchy [morph, grapheme, char], 10k trees per language", { x: X0, y: 1.3, w: CW, h: 0.45, fontSize: 15, color: C.text1, margin: 0, isTextBox: true });
  [[C.accent2, "FaCheckCircle", "Training side", [["0", "spurious morpheme boundaries for +2.2 % tokens (min-morph anchor)"], ["4×", "fewer spurious boundaries at the compression anchor (2.0 vs 7.8)"]]],
   [C.accent4, "FaExclamationTriangle", "Held-out", [["423 vs 349", "FineWeb tokens / 1000 chars (morph-first vs count-based)"], ["69 vs 52", "worst-language FLORES tokens per sentence"]]]].forEach(([col, ic, t, st], i) => {
    const x = X0 + i * 6.2;
    card(s, x, 2.0, 5.93, 3.9);
    iconCircle(s, ic, x + 0.3, 2.2, 0.65, col);
    s.addText(t, { x: x + 1.15, y: 2.2, w: 4.5, h: 0.65, fontSize: 20, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true });
    st.forEach(([b, l], k) => stat(s, b, l, x + 0.3, 3.1 + k * 1.4, 5.3, col));
  });
  s.addText("Unseen words need a morphological segmenter at inference; gold F1 improves only modestly (0.29 vs 0.24) — the same inference-cost trade-off MUTANT reports.", { x: X0, y: 6.1, w: CW, h: 0.6, fontSize: 14, color: C.text1, margin: 0, isTextBox: true });
  s.addNotes("Morph-first trees remove the compression-morphology conflict on the training side entirely, but held out they collapse because unseen words rely on Morfessor at inference.");

  // ===== Conclusion
  pres.addSection({ title: "Conclusion" });
  s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Conclusion" });
  s.addText("Key findings", { x: 0.8, y: 0.45, w: 11.7, h: 0.8, fontSize: 36, bold: true, color: C.background1, margin: 0, fontFace: THEME.headFontFace, isTextBox: true });
  const finds = [
    ["RQ1", "One LP optimises all objectives and stays near-integral at scale (|gap| ≤ 3·10⁻⁴); bound shading makes every point certified."],
    ["RQ2", "Head-only fronts look great but do not transfer. An SAA objective, guarded against the optimizer's curse, gives fronts that hold on unseen text."],
    ["RQ3", "LP: most even tokenizer (Gini 0.004), worst-language cost below BPE. Still behind BPE on mean, Parity-aware BPE on fairness, MorphBPE on morphology."],
    ["RQ4", "LM quality follows morphology more than compression; fairer tokenizers do not make fairer LMs."],
  ];
  finds.forEach(([k, t], i) => {
    const y = 1.55 + i * 1.12;
    s.addShape(pres.shapes.OVAL, { x: 0.8, y, w: 0.8, h: 0.8, fill: { color: C.accent1 }, line: { color: C.accent1 } });
    s.addText(k, { x: 0.8, y, w: 0.8, h: 0.8, fontSize: 15, bold: true, color: C.text2, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(t, { x: 1.85, y: y - 0.05, w: 10.6, h: 0.9, fontSize: 17, color: C.background1, margin: 0, valign: "middle", isTextBox: true });
  });
  s.addText([
    { text: "Next: ", options: { bold: true, color: C.accent1 } },
    { text: "the candidate set is the bottleneck → add BPE / Unigram pieces as extra columns; re-estimate SAA on an independent sample.", options: { color: C.background1 } },
  ], { x: 0.8, y: 6.15, w: 11.7, h: 0.6, fontSize: 16, margin: 0, isTextBox: true });
  s.addNotes("One finding per research question. The meta-lesson: what the LP optimises matters more than how well it optimises it. Next step: richer candidate sets.");

  // References
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Conclusion" });
  s.addText("References", { placeholder: "title" });
  body(s, bullets([
    "Schmidt et al. (2026). Tokenization with Split Trees (ToaST). arXiv:2605.22705",
    "Tempus et al. (2026). Tokenisation via Convex Relaxations (ConvexTok). arXiv:2605.22821",
    "Foroutan et al. (2026). Parity-Aware Byte-Pair Encoding. ACL 2026 · code: github.com/swiss-ai/parity-aware-bpe",
    "Asgari et al. (2026). MorphBPE. Findings of ACL 2026 (reimplemented; no public code)",
    "Rana et al. (2026). MUTANT: a multilingual tokenizer recipe for Indic languages. ACL 2026",
    "Arnett & Bergen (2025); Arnett et al. (2025). MorphScore. Data: huggingface.co/datasets/catherinearnett/morphscore",
    "Penedo et al. FineWeb and FineWeb-2 (HuggingFaceFW) · FLORES+ (openlanguagedata/flores_plus) · SIGMORPHON 2022 Shared Task",
    "Creutz & Lagus. Morfessor Baseline · Huangfu & Hall. HiGHS (highs.dev) · Hugging Face tokenizers",
    "Our code: github.com/n6m9n/769",
  ], 14), X0, 1.45, CW, 5.3);
  s.addNotes("Papers, datasets and code we used.");

  s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Conclusion" });
  s.addText("Thank you", { placeholder: "body" });
  s.addText("Questions?", { placeholder: "title" });
  s.addNotes("Backup slides follow for Q&A.");

  // ===== Backup
  pres.addSection({ title: "Backup" });
  s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Backup" });
  s.addText("Backup slides", { placeholder: "body" });
  s.addText("Technical details for questions", { placeholder: "title" });

  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  s.addText("B1 · Training-side fronts of the main run (SAA 10k+10k)", { placeholder: "title" });
  const hb1 = img(s, "fig_train_fronts.png", X0, 1.35, 9.0, "Training-side fronts of the SAA 10k+10k LP");
  caption(s, "Figure B1. 110 k trees, 1.7 M nodes, 2.6 M rows. 35 points, |gap| ≤ 2.1·10⁻⁴, 0 bound violations, monotone fronts.", X0, 1.35 + hb1 + 0.08, 9.0, 0.55);
  body(s, bullets([
    "Worst language 370 → 333 (−10 %) for +0.5 % Ttot",
    "→ 314 (−15 %) at min-max (+12 %)",
    "Spurious boundaries 53.1 → 48.6 for +4.4 %",
    "Anchors: 387 s / 2 524 s / 672 s; warm point median 260 s",
  ], 14), 9.85, 1.45, 2.9, 4.8);
  s.addNotes("Numbers for the main run's training side.");

  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  s.addText("B2 · Shadow prices along the sweeps", { placeholder: "title" });
  const hb2 = img(s, "fig_shadow_prices.png", X0, 1.4, 8.6, "Shadow prices dTtot/dbound along parity and morph sweeps");
  caption(s, "Figure B2. −dTtot/d(bound) from the reduced cost of the bounded aggregate column: total tokens saved per unit of relaxing the bound.", X0, 1.4 + hb2 + 0.08, 8.6, 0.55);
  body(s, bullets([
    "Read directly from the simplex solution, no extra solves",
    "Small near the compression anchor (fairness is cheap), growing as τ tightens",
    "Exact local exchange rate of the LP relaxation",
  ], 14), 9.5, 1.5, 3.23, 4.5);
  s.addNotes("Shadow prices: the price of fairness in total tokens, read off the reduced costs.");

  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  s.addText("B3 · LP sizes and solve times", { placeholder: "title" });
  table(s, ["Run", "Trees", "Nodes", "Coverage mean / worst", "Compression", "Min-max", "Min-morph", "Warm point"], [
    ["head-only 5k", "27 k", "0.36 M", "71 / 59 %", "65 s", "–", "–", "–"],
    ["head-only 10k", "54 k", "0.77 M", "77 / 66 %", "312 s", "1 690 s", "222 s", "119 s"],
    ["head-only 20k", "109 k", "1.6 M", "83 / 73 %", "1 171 s", "8 708 s", "819 s", "361 s"],
    ["head-only 40k", "216 k", "3.4 M", "88 / 80 %", "3 323 s", "–", "–", "–"],
    ["SAA 5k+5k", "56 k", "0.83 M", "(all text)", "76 s", "437 s", "122 s", "29 s"],
    ["SAA 10k+10k", "111 k", "1.7 M", "(all text)", "387 s", "2 524 s", "672 s", "260 s"],
    ["SAA 20k+20k", "219 k", "3.5 M", "(all text)", "2 012 s", "17 107 s", "1 170 s", "318 s"],
  ], X0, 1.45, CW, [1.9, 1.0, 1.0, 2.13, 1.55, 1.55, 1.5, 1.5], { size: 13, rowH: 0.46 });
  caption(s, "Table B3. HiGHS 1.16 on one core per process; warm point = median sweep point. All compression anchors are exactly integral.", X0, 5.3, CW);
  body(s, bullets([
    "Vectorised LP build: 2.6 M rows in 7 s; exact level-wise n-gram counting keeps memory ∝ frequent n-grams",
    "Warm-starting min-max from the compression basis (primal simplex): inconclusive",
  ], 14), X0, 5.85, CW, 1.0);
  s.addNotes("Timings for every run.");

  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  s.addText("B4 · SAA details", { placeholder: "title" });
  body(s, bullets([
    [{ text: "Head set H: ", options: { bold: true } }, { text: "top-K pretokens per language (exact counts in every language)" }],
    [{ text: "Tail of language l: ", options: { bold: true } }, ...M("pretokens of l not in H, total mass M_l")],
    [{ text: "Sampling: ", options: { bold: true } }, ...M("S draws with replacement, probability ∝ count; each draw has weight M_l / S → unbiased for fixed x")],
    [{ text: "Bias: ", options: { bold: true } }, ...M("E[min_x Ĉ(x)] ≤ min_x E[Ĉ(x)] (optimizer's curse); large per-sample weights make it severe")],
    [{ text: "Candidate rule: ", options: { bold: true } }, ...M("variables x_i of tokens that occur only in tail trees are fixed to 0 (55 % of token variables at 5k+5k)")],
    [{ text: "Check: ", options: { bold: true } }, { text: "exact cost over every training pretoken type vs LP estimate (Table 6); exact training cost = held-out cost (ru 381 vs 380)" }],
    [{ text: "Not yet done: ", options: { bold: true } }, { text: "re-estimate the chosen x on an independent sample to report the statistical SAA gap" }],
  ], 15), X0, 1.45, CW, 5.3);
  s.addNotes("Formal details of the sample average approximation and the bias.");

  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  s.addText("B5 · LM protocol and noise floor", { placeholder: "title" });
  table(s, ["Setting", "Value"], [
    ["Model", "GPT, 6 layers, d = 512, 8 heads, context 512, tied embeddings (35.5 M params, 18.9 M non-embedding)"],
    ["Data", "same 144 MB for every tokenizer: 24 MB per language, shuffled with a fixed seed"],
    ["Training", "4 epochs, AdamW (lr 1.5e-3, wd 0.1), cosine schedule, 64 × 512 tokens per step, bf16"],
    ["Evaluation", "sliding window (stride 256); bits per byte on FineWeb eval and FLORES+ devtest"],
    ["Seeds", "3 per tokenizer; sd 0.001–0.02 bpb"],
    ["Noise floor", "Parity-aware BPE base vs window: identical tokenization, permuted ids → up to 0.015 bpb apart"],
    ["Caveat", "matched bytes ⇒ more tokens get more steps (MorphBPE +7 %, Unigram +17 % vs BPE)"],
  ], X0, 1.45, CW, [2.2, 9.93], { size: 13, rowH: 0.5 });
  caption(s, "Table B5. Language-model protocol (scripts/07_lm.py).", X0, 5.6, CW);
  s.addNotes("LM protocol details and the measured noise floor.");

  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  s.addText("B6 · Baselines and correctness checks", { placeholder: "title" });
  card(s, X0, 1.45, 5.9, 3.3);
  s.addText("Baselines", { x: X0 + 0.25, y: 1.6, w: 5.4, h: 0.4, fontSize: 17, bold: true, color: C.text2, margin: 0, isTextBox: true });
  body(s, bullets([
    "Same GPT-4o pretokenizer and 32 k vocabulary for all",
    M("BPE / Unigram: HF tokenizers, temperature mixture q_l ∝ p_l^α (p_l: data share)"),
    "Parity-aware BPE: authors' learner, FLORES+ dev as parity set; base and window variants give the identical vocabulary",
    "MorphBPE: reimplemented — invisible separator at every Morfessor boundary, so no merge crosses a morpheme",
  ], 14), X0 + 0.25, 2.1, 5.4, 4.2);
  card(s, 6.83, 1.45, 5.9, 3.3);
  s.addText("Checks (all passing)", { x: 7.08, y: 1.6, w: 5.4, h: 0.4, fontSize: 17, bold: true, color: C.text2, margin: 0, isTextBox: true });
  body(s, bullets([
    "LP ≤ integer at 123 / 123 points, 0 bound violations",
    "Monotone training-side fronts (LP and integer)",
    "decode(encode(s)) = s: 0 failures over 123 tokenizers × 1 804 strings, incl. unseen scripts",
    "LP model = real tokenizer on 20 000 trees",
    "13 unit tests, incl. brute-force LP optimality on a toy problem",
  ], 14), 7.08, 2.1, 5.4, 4.2);
  s.addNotes("How the baselines were built and how we verified correctness.");
}

build().then(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("written", OUT);
}).catch((e) => { console.error(e); process.exit(1); });
