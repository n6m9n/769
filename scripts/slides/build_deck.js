// CS 769 final presentation v2: LaTeX-typeset maths (MathJax -> high-res PNG), slide-sized figures.
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { tex } = require("./tex.js");
const { applyTheme } = require(process.env.SKILL_DIR + "/scripts/apply_theme.js");

const FIG = "/Users/naman/Desktop/769/motok/report/slide_figures/";
const OUT = "/Users/naman/Desktop/769/motok/report/presentation.pptx";

const THEME = {
  name: "Pareto Tokenization", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1E1E24", lt1: "FFFFFF", dk2: "1B1F3B", lt2: "F3F4F7", accent1: "D9912B", accent2: "2A78D6",
            accent3: "2A9D8F", accent4: "EB6834", accent5: "6250D6", accent6: "8A8D99", hlink: "2A78D6", folHlink: "6250D6" },
};
const H = THEME.colors;
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Multi-objective Vocabulary Selection for Multilingual Tokenization";
pres.subject = "CS 769 final presentation"; pres.author = "CS 769 team";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const X0 = 0.6, CW = 12.13, TOP = 1.5;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "TITLE_DARK", background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.7, w: 11.7, h: 1.8, fontSize: 40, color: C.background1, valign: "bottom", align: "left", fontFace: THEME.headFontFace }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.62, w: 11.7, h: 0.8, fontSize: 20, color: C.accent1, valign: "top", align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 }, margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "body", type: "body", x: X0, y: 0.34, w: CW, h: 0.3, fontSize: 12, bold: true, color: C.accent1, charSpacing: 2, valign: "middle", align: "left" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: X0, y: 0.62, w: CW, h: 0.72, fontSize: 30, color: C.text2, valign: "middle", align: "left", fontFace: THEME.headFontFace }, text: "" } },
    { text: { text: "Multi-objective vocabulary selection · CS 769", options: { x: X0, y: 7.06, w: 6, h: 0.28, fontSize: 10, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 12.15, y: 7.06, w: 0.58, h: 0.28, fontSize: 10, color: H.accent6, align: "right" },
});
pres.defineSlideMaster({
  title: "SECTION_DARK", background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 2.55, w: 11.7, h: 0.5, fontSize: 16, bold: true, color: C.accent1, charSpacing: 3, align: "left" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 3.05, w: 11.7, h: 1.3, fontSize: 40, color: C.background1, valign: "top", align: "left", fontFace: THEME.headFontFace }, text: "" } },
  ],
  slideNumber: { x: 12.15, y: 7.06, w: 0.58, h: 0.28, fontSize: 10, color: H.accent6, align: "right" },
});

// ---------------------------------------------------------------- helpers
const ICON = {};
async function icon(name, color = "FFFFFF") {
  const k = name + color;
  if (!ICON[k]) {
    const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(fa[name], { color: "#" + color, size: "256" }));
    ICON[k] = "image/png;base64," + (await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer()).toString("base64");
  }
  return ICON[k];
}
async function iconCircle(s, name, x, y, d, fill) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: "icon " + name });
  const p = d * 0.27;
  s.addImage({ data: await icon(name), x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p, altText: name.replace("Fa", "") + " icon" });
}
const TEXC = {};
async function TEX(src, pt = 24, color = H.dk1) {
  const k = src + pt + color;
  if (!TEXC[k]) TEXC[k] = await tex(src, pt, color);
  return TEXC[k];
}
// place LaTeX; align: "l" | "c" | "r" inside box (x,w); vertically centred on yc; scaled down to maxW
async function math(s, src, x, yc, pt = 24, opts = {}) {
  const r = await TEX(src, pt, opts.color || H.dk1);
  let w = r.w, h = r.h;
  if (opts.maxW && w > opts.maxW) { h *= opts.maxW / w; w = opts.maxW; }
  let xx = x;
  if (opts.boxW && opts.align === "c") xx = x + (opts.boxW - w) / 2;
  if (opts.boxW && opts.align === "r") xx = x + opts.boxW - w;
  s.addImage({ data: r.data, x: xx, y: yc - h / 2, w, h, altText: "Equation: " + src, objectName: "math" });
  return { w, h, x: xx };
}
// numbered display equation in a row [x, x+w]; number at the right
async function eqn(s, src, num, x, yc, w, pt = 24, align = "l") {
  const r = await math(s, src, x, yc, pt, { maxW: w - 1.0, boxW: w - 0.9, align });
  s.addText(`(${num})`, { x: x + w - 0.8, y: yc - 0.2, w: 0.8, h: 0.4, fontSize: 16, color: C.accent6, align: "right", valign: "middle", margin: 0, isTextBox: true, objectName: "equation number " + num });
  return r;
}
// "where" list: rows of [LaTeX symbol, description]
async function defs(s, rows, x, y, w, opts = {}) {
  const symW = opts.symW || 1.3, rowH = opts.rowH || 0.44, pt = opts.pt || 17, fs = opts.fs || 15;
  if (opts.head !== false) s.addText(opts.head || "where", { x, y: y - 0.38, w, h: 0.32, fontSize: 13, italic: true, color: C.accent6, margin: 0, isTextBox: true });
  for (let i = 0; i < rows.length; i++) {
    const yc = y + i * rowH + rowH / 2;
    await math(s, rows[i][0], x, yc, pt, { maxW: symW - 0.1 });
    s.addText(rows[i][1], { x: x + symW, y: yc - rowH / 2, w: w - symW, h: rowH, fontSize: fs, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
  }
  return y + rows.length * rowH;
}
function T(s, text, x, y, w, h, o = {}) {
  s.addText(text, { x, y, w, h, fontSize: 17, color: C.text1, margin: 0, valign: "top", isTextBox: true, paraSpaceAfter: 6, ...o });
}
function bullets(items, size = 17, gap = 8) {
  return items.map((t, i) => {
    const runs = Array.isArray(t) ? t : [{ text: t }];
    return runs.map((r, j) => ({ text: r.text, options: { ...(r.options || {}), fontSize: size, ...(j === 0 ? { bullet: { indent: 18 }, paraSpaceAfter: gap } : {}), ...(j === runs.length - 1 && i < items.length - 1 ? { breakLine: true } : {}) } }));
  }).flat();
}
const B = (t) => ({ text: t, options: { bold: true, color: C.text2 } });
function caption(s, text, x, y, w, h = 0.36) {
  s.addText(text, { x, y, w, h, fontSize: 12, italic: true, color: C.accent6, margin: 0, valign: "top", isTextBox: true, objectName: "caption" });
}
function cite(s, text, x = X0, y = 6.68, w = CW) {
  s.addText(text, { x, y, w, h: 0.28, fontSize: 11, color: C.accent6, margin: 0, isTextBox: true, objectName: "citation" });
}
async function fig(s, file, x, y, w, alt) {
  const m = await sharp(FIG + file).metadata();
  const h = w * m.height / m.width;
  s.addImage({ path: FIG + file, x, y, w, h, altText: alt, objectName: "figure " + file });
  return h;
}
function card(s, x, y, w, h, fill = C.background2) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: "card" });
}
function pill(s, text, x, y, fill = C.accent1) {
  const w = 0.22 + text.length * 0.095;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.27, rectRadius: 0.13, fill: { color: fill }, line: { color: fill }, objectName: "tag " + text });
  s.addText(text, { x, y, w, h: 0.27, fontSize: 10, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, charSpacing: 1, isTextBox: true });
  return w;
}
const HEX = () => ({ [C.accent1]: H.accent1, [C.accent2]: H.accent2, [C.accent3]: H.accent3, [C.accent4]: H.accent4, [C.accent5]: H.accent5, [C.accent6]: H.accent6, [C.text2]: H.dk2, [C.text1]: H.dk1 });
async function stat(s, big, label, x, y, w, color, isTex = false) {
  if (isTex) await math(s, big, x, y + 0.36, 30, { color: HEX()[color] || color });
  else s.addText(big, { x, y, w, h: 0.72, fontSize: 34, color, margin: 0, isTextBox: true, fontFace: THEME.headFontFace, objectName: "stat value" });
  s.addText(label, { x, y: y + 0.76, w, h: 0.62, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "stat label" });
}
function table(s, header, rows, x, y, w, colW, o = {}) {
  const hd = header.map((t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: o.size || 15 } }));
  const body = rows.map((r, ri) => r.map((c) => {
    const v = typeof c === "object" ? c : { text: String(c) };
    return { text: v.text, options: { fontSize: o.size || 15, color: v.color || C.text1, fill: { color: v.fill || (ri % 2 ? C.background2 : C.background1) }, bold: !!v.bold } };
  }));
  s.addTable([hd, ...body], { x, y, w, colW, border: { type: "solid", pt: 0.5, color: "DCDDE3" }, margin: [4, 6, 4, 6], valign: "middle", rowH: o.rowH || 0.4, objectName: o.name || "table" });
}
function slide(section, eyebrow, title, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(eyebrow.toUpperCase(), { placeholder: "body" });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}

// ---------------------------------------------------------------- deck
async function build() {
  let s, y;

  // ======================= 1. Title
  pres.addSection({ title: "Introduction" });
  s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
  s.addText("Multi-objective Vocabulary Selection for Multilingual Tokenization", { placeholder: "title" });
  s.addText("Certified Pareto fronts with linear programming, and what transfers to unseen text", { placeholder: "body" });
  s.addText("English   ·   हिन्दी   ·   मराठी   ·   தமிழ்   ·   Русский   ·   Magyar", { x: 0.8, y: 0.8, w: 11.7, h: 0.45, fontSize: 18, color: C.accent6, margin: 0, isTextBox: true, objectName: "languages" });
  s.addText([
    { text: "[Student name 1, Roll no.]   ·   [Student name 2, Roll no.]", options: { bold: true, breakLine: true } },
    { text: "[Department, Institute]   ·   Supervisor: [Prof. Name]", options: { breakLine: true } },
    { text: "CS 769: Optimization in Machine Learning   ·   Final presentation   ·   [Date]", options: { color: C.accent6 } },
  ], { x: 0.8, y: 5.2, w: 11.7, h: 1.2, fontSize: 16, color: C.background1, margin: 0, paraSpaceAfter: 4, isTextBox: true, objectName: "team details" });
  s.addNotes("Introduce the team and supervisor. One sentence: we cast tokenizer vocabulary selection as a single multi-objective linear program, and study what its Pareto fronts buy on unseen text.");

  // ======================= 2. Motivation
  s = slide("Introduction", "Introduction and motivation", "The same sentence costs some languages far more tokens",
    "Tokens are the unit of cost for language models. With standard BPE trained on a natural, English-heavy mix, a Tamil sentence needs 60 % more tokens than the same English sentence. That means higher API cost, less usable context and slower inference for Tamil speakers.");
  await stat(s, "1.60×", "token premium of Tamil over English for the same FLORES+ sentences (BPE, natural data mix)", X0, TOP + 0.05, 3.7, C.accent4);
  await stat(s, "48.4 vs 30.1", "tokens per sentence: Tamil vs English, identical content", X0, TOP + 1.65, 3.7, C.accent4);
  T(s, bullets(["More tokens: higher cost, less context, slower inference", "Frequency-based tokenizers let the dominant language of the training mix win"], 16), X0, TOP + 3.35, 3.8, 1.6);
  y = await fig(s, "s_perlang.png", 4.75, TOP - 0.05, 7.98, "Grouped bar chart of tokens per FLORES sentence per language for five tokenizers");
  caption(s, "Figure 1. Tokens per FLORES+ devtest sentence (same content in every language). Lower is better; equal bars mean parity.", 4.75, TOP + y + 0.05, 7.98, 0.55);
  cite(s, "Data: FLORES+ [8]; BPE [12]; Parity-aware BPE [3]. LP tokenizers are ours.", X0, 6.68, 3.9);

  // ======================= 3. Limitations of existing methods
  s = slide("Introduction", "Introduction and motivation", "Existing tokenizers optimise one goal, greedily",
    "Each existing method fixes one objective or one heuristic. None lets us see the trade-off between compression, fairness and morphology, and none can certify how far it is from the best possible vocabulary.");
  table(s, ["Method", "What it optimises", "Limitation"], [
    [{ text: "BPE / Unigram [12]", bold: true }, "compression (greedy merges / likelihood)", "single objective; dominant language wins"],
    [{ text: "Temperature sampling", bold: true }, "re-weights the training mix", "heuristic knob; no explicit trade-off"],
    [{ text: "Parity-aware BPE [3]", bold: true }, "worst language, greedily", "one criterion; no optimality guarantee"],
    [{ text: "MorphBPE [4]", bold: true }, "morphology as a hard constraint", "all-or-nothing; no exchange rate"],
    [{ text: "ToaST [1]", bold: true }, "compression via an LP over split trees", "single objective"],
  ], X0, TOP + 0.05, CW, [3.0, 4.5, 4.63], { size: 16, rowH: 0.56 });
  caption(s, "Table 1. Closest prior work. Our formulation builds directly on ToaST's split-tree LP.", X0, TOP + 3.5, CW);
  card(s, X0, 5.55, CW, 0.95, C.text2);
  s.addText([{ text: "The gap.  ", options: { bold: true, color: C.accent1 } },
    { text: "No method exposes the full trade-off between objectives, or certifies how close a vocabulary is to optimal.", options: { color: C.background1 } }],
  { x: X0 + 0.35, y: 5.55, w: CW - 0.7, h: 0.95, fontSize: 18, margin: 0, valign: "middle", isTextBox: true });

  // ======================= 4. Problem statement
  s = slide("Introduction", "Problem statement", "One vocabulary, competing objectives",
    "Formally a vector optimisation problem over vocabularies of fixed size: we want the Pareto set, not one point. Four research questions structure the talk.");
  await eqn(s, "\\min_{\\mathbf{x}\\,\\in\\,\\mathcal{X}_m}\\;\\Big(\\,T_{\\mathrm{tot}}(\\mathbf{x}),\\;\\; t(\\mathbf{x}),\\;\\; M_{\\mathrm{tot}}(\\mathbf{x})\\,\\Big)", 1, X0, TOP + 0.45, 7.0, 28, "c");
  await defs(s, [
    ["\\mathbf{x}", "vocabulary (one indicator per candidate token)"],
    ["\\mathcal{X}_m", "all vocabularies of the fixed size (32 000)"],
    ["T_{\\mathrm{tot}}", "compression: tokens over all text (data-weighted)"],
    ["t", "parity: cost of the worst-served language"],
    ["M_{\\mathrm{tot}}", "morphology: token boundaries inside morphemes"],
  ], X0, TOP + 1.65, 7.0, { rowH: 0.5, fs: 16 });
  card(s, 7.95, TOP, 4.78, 4.95);
  s.addText("Research questions", { x: 8.25, y: TOP + 0.2, w: 4.2, h: 0.4, fontSize: 18, bold: true, color: C.text2, margin: 0, isTextBox: true });
  const rqs = [["RQ1", "Can one LP optimise all objectives, near-integrally, at scale?"], ["RQ2", "What do the Pareto fronts and exchange rates look like?"],
    ["RQ3", "Do the fronts beat strong baselines on held-out text?"], ["RQ4", "Which points give the best language models?"]];
  rqs.forEach(([k, t], i) => {
    const yy = TOP + 0.85 + i * 1.0;
    s.addShape(pres.shapes.OVAL, { x: 8.25, y: yy, w: 0.62, h: 0.62, fill: { color: C.accent1 }, line: { color: C.accent1 } });
    s.addText(k, { x: 8.25, y: yy, w: 0.62, h: 0.62, fontSize: 12, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(t, { x: 9.05, y: yy - 0.08, w: 3.5, h: 0.8, fontSize: 15, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
  });

  // ======================= 5. Data
  pres.addSection({ title: "Data" });
  s = slide("Data", "Data used", "Six languages, deliberately imbalanced",
    "Training text from FineWeb (English) and FineWeb-2, imbalanced on purpose to create a fairness problem. Held-out text, the parallel FLORES+ set and gold morphology are used for evaluation only.");
  table(s, ["Language", "Script", "Morphology", "Text", "Characters", "Pretoken types"], [
    ["English", "Latin", "analytic", "200 MB", "197.9 M", "610 k"],
    ["Hindi", "Devanagari", "fusional", "100 MB", "39.7 M", "240 k"],
    ["Marathi", "Devanagari", "agglutinative", "50 MB", "19.0 M", "290 k"],
    ["Tamil", "Tamil", "agglutinative", "25 MB", "9.4 M", "242 k"],
    ["Russian", "Cyrillic", "fusional", "75 MB", "41.7 M", "519 k"],
    ["Hungarian", "Latin", "agglutinative", "50 MB", "45.4 M", "669 k"],
  ], X0, TOP + 0.05, 7.4, [1.35, 1.35, 1.5, 0.95, 1.15, 1.1], { size: 15, rowH: 0.48 });
  caption(s, "Table 2. Training corpora (FineWeb sample-10BT [7] for English, FineWeb-2 [7] otherwise): 393 M characters in total. Devanagari and Tamil use 3 bytes per character.", X0, TOP + 3.5, 7.4, 0.6);
  const ev = [["FaSearch", "Held-out text", "first 2 000 documents per language"], ["FaGlobeAsia", "Parallel text", "FLORES+ devtest [8]: 1 012 sentences, same content in all six languages"],
    ["FaCut", "Gold morphology", "MorphScore [6] (all six) and SIGMORPHON 2022 [9] (en, ru, hu)"], ["FaCogs", "Preprocessing", "GPT-4o pretokenizer; Morfessor [10] segmentation as the training signal only"]];
  for (let i = 0; i < ev.length; i++) {
    const yy = TOP + 0.05 + i * 1.22;
    await iconCircle(s, ev[i][0], 8.35, yy, 0.6, C.accent3);
    s.addText(ev[i][1], { x: 9.1, y: yy - 0.04, w: 3.6, h: 0.34, fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true });
    s.addText(ev[i][2], { x: 9.1, y: yy + 0.32, w: 3.6, h: 0.78, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  }

  // ======================= 6. Approach: pipeline
  pres.addSection({ title: "Proposed approach" });
  s = slide("Proposed approach", "Proposed approach", "From text to certified Pareto-optimal tokenizers",
    "The pipeline. Steps marked OURS are the contributions of this project; the split-tree representation comes from ToaST. The LP stage runs three anchors in parallel, then each sweep in its own process, warm-started from saved simplex bases. Every Pareto point becomes a real tokenizer that is evaluated on held-out text and in small language models.");
  const P = [
    ["FaDatabase", "Corpora", "FineWeb / FineWeb-2, six languages", null, C.accent6],
    ["FaLayerGroup", "Pretokenize & count", "GPT-4o regex; Morfessor morph marks", null, C.accent6],
    ["FaTree", "Split trees", "head trees + sampled tail trees (SAA)", "OURS: SAA", C.accent2],
    ["FaProjectDiagram", "Multi-objective LP", "objectives as aggregate columns", "OURS", C.accent2],
    ["FaRandom", "Warm-started sweeps", "3 anchors in parallel, then 5 sweeps in parallel (HiGHS)", "OURS", C.accent2],
    ["FaCheckCircle", "Round & certify", "exact inference + bound shading", "OURS", C.accent2],
    ["FaLanguage", "Pareto set of tokenizers", "one tokenizer per Pareto point", null, C.accent3],
    ["FaChartLine", "Evaluation", "held-out, FLORES+, gold morphology, LMs", null, C.accent3],
  ];
  const cw = 2.78, ch = 1.72, gx = (CW - 4 * cw) / 3;
  for (let i = 0; i < 8; i++) {
    const row = Math.floor(i / 4), col = i % 4;
    const x = X0 + col * (cw + gx), yy = TOP + 0.1 + row * 2.45;
    card(s, x, yy, cw, ch);
    await iconCircle(s, P[i][0], x + 0.2, yy + 0.22, 0.55, P[i][4]);
    s.addText(P[i][1], { x: x + 0.88, y: yy + 0.18, w: cw - 1.0, h: 0.62, fontSize: 15.5, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true });
    s.addText(P[i][2], { x: x + 0.2, y: yy + 0.92, w: cw - 0.35, h: 0.7, fontSize: 13.5, color: C.text1, margin: 0, valign: "top", isTextBox: true });
    if (P[i][3]) pill(s, P[i][3], x + 0.2, yy - 0.14);
    if (col < 3) s.addShape(pres.shapes.LINE, { x: x + cw + 0.04, y: yy + ch / 2, w: gx - 0.08, h: 0, line: { color: H.accent6, width: 1.75, endArrowType: "triangle" } });
  }
  // elbow connector row 1 -> row 2
  const xe = X0 + 3 * (cw + gx) + cw / 2, xs = X0 + cw / 2, y1 = TOP + 0.1 + ch, y2 = TOP + 0.1 + 2.45;
  s.addShape(pres.shapes.LINE, { x: xe, y: y1 + 0.04, w: 0, h: (y2 - y1) / 2 - 0.04, line: { color: H.accent6, width: 1.75 } });
  s.addShape(pres.shapes.LINE, { x: xs, y: y1 + (y2 - y1) / 2, w: xe - xs, h: 0, line: { color: H.accent6, width: 1.75 } });
  s.addShape(pres.shapes.LINE, { x: xs, y: y1 + (y2 - y1) / 2, w: 0, h: (y2 - y1) / 2 - 0.04, line: { color: H.accent6, width: 1.75, endArrowType: "triangle" } });
  caption(s, "Figure 2. End-to-end pipeline (scripts 01–08 of our repository). Grey: standard preprocessing; blue: our optimisation contributions; teal: outputs and evaluation.", X0, TOP + 4.8, CW);
  cite(s, "Split trees: ToaST [1]. Solver: HiGHS [11]. Morphology: Morfessor [10].");

  // ======================= 7. Split trees
  pres.addSection({ title: "Methodology" });
  s = slide("Methodology", "Methodology · representation", "Split trees turn tokenization into a linear problem",
    "Split trees from ToaST. Each frequent pretoken gets a fixed binary tree built only from n-gram counts. Given a vocabulary, tokenization emits the first in-vocabulary node on every root-to-leaf path. Because the trees never depend on the vocabulary, choosing the vocabulary is linear in the node variables.");
  const nodeW = 1.75, nodeH = 0.56, T0 = TOP + 0.35;
  const node = (t, x, yy, emit) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: yy, w: nodeW, h: nodeH, rectRadius: 0.1, fill: { color: emit ? C.accent1 : C.background2 }, line: { color: emit ? C.accent1 : "C9CBD3", width: 1 }, objectName: "tree node " + t });
    s.addText(t, { x, y: yy, w: nodeW, h: nodeH, fontSize: 17, align: "center", valign: "middle", color: emit ? C.background1 : C.text1, bold: !!emit, margin: 0, fontFace: "Courier New", isTextBox: true });
  };
  const edge = (x1, y1_, x2, y2_) => s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: y1_, w: Math.abs(x2 - x1), h: y2_ - y1_, flipH: x2 < x1, line: { color: "9A9DA8", width: 1.5 } });
  const cx = (x) => x + nodeW / 2;
  node("␣walking", 2.55, T0, false);
  node("␣walk", 1.15, T0 + 1.25, true); node("ing", 3.95, T0 + 1.25, true);
  node("␣w", 0.6, T0 + 2.5, false); node("alk", 1.95, T0 + 2.5, false); node("in", 3.4, T0 + 2.5, false); node("g", 4.75, T0 + 2.5, false);
  edge(cx(2.55), T0 + nodeH, cx(1.15), T0 + 1.25); edge(cx(2.55), T0 + nodeH, cx(3.95), T0 + 1.25);
  edge(cx(1.15), T0 + 1.25 + nodeH, cx(0.6), T0 + 2.5); edge(cx(1.15), T0 + 1.25 + nodeH, cx(1.95), T0 + 2.5);
  edge(cx(3.95), T0 + 1.25 + nodeH, cx(3.4), T0 + 2.5); edge(cx(3.95), T0 + 1.25 + nodeH, cx(4.75), T0 + 2.5);
  s.addText("…  down to single characters (the leaves)", { x: 0.6, y: T0 + 3.2, w: 5.9, h: 0.35, fontSize: 14, italic: true, color: C.accent6, margin: 0, isTextBox: true });
  caption(s, "Figure 3. Split tree of the pretoken “␣walking” (␣ = space). With ␣walk and ing in the vocabulary, inference emits the highlighted nodes, giving the tokens [␣walk] [ing].", 0.6, T0 + 3.7, 5.9, 0.62);
  T(s, bullets([
    [B("Build: "), { text: "one tree per frequent pretoken; split where the smaller side's n-gram count is largest" }],
    [B("Infer: "), { text: "walk down; emit the first in-vocabulary node on every root-to-leaf path" }],
    [B("Key property: "), { text: "trees never depend on the vocabulary, so emissions are linear in node variables" }],
    [B("Unseen words: "), { text: "a tree from the same counts at inference; unseen characters fall back to bytes" }],
  ], 17, 12), 7.05, TOP + 0.3, 5.68, 4.6);
  cite(s, "Split trees and the vocabulary IP: Schmidt et al. (ToaST) [1].");

  // ======================= 8. Core LP
  s = slide("Methodology", "Methodology · core LP", "The vocabulary-selection linear program",
    "Equations 2 to 5. Two kinds of variables: tokens and tree nodes. The vocabulary has a fixed size, every character is covered exactly once along its root path, nodes can only be emitted if their token is in the vocabulary, and single characters are always available. This is the LP relaxation of an integer program; ToaST observed it is almost always integral.");
  const rowY = [TOP + 0.45, TOP + 1.35, TOP + 2.25, TOP + 3.15];
  await eqn(s, "\\sum_{i} x_i \\;=\\; m - 256", 2, X0, rowY[0], 6.6, 24);
  await eqn(s, "\\sum_{k \\in P(v)} z_k \\;=\\; 1 \\qquad \\forall\\ \\text{leaves } v", 3, X0, rowY[1], 6.6, 24);
  await eqn(s, "z_k \\;\\le\\; x_{\\mathrm{tok}(k)} \\qquad \\forall\\ \\text{nodes } k", 4, X0, rowY[2], 6.6, 24);
  await eqn(s, "x_i = 1 \\ \\ \\forall i \\in \\mathcal{B}, \\qquad 0 \\le x_i,\\, z_k \\le 1", 5, X0, rowY[3], 6.6, 24);
  await defs(s, [
    ["x_i", "candidate token is in the vocabulary"],
    ["z_k", "tree node is emitted"],
    ["m", "vocabulary size (32 000); 256 byte tokens are reserved"],
    ["P(v)", "nodes on the path from a leaf to its root"],
    ["\\mathrm{tok}(k)", "token string of a node"],
    ["\\mathcal{B}", "single characters (always in the vocabulary)"],
  ], 7.75, TOP + 0.45, 4.98, { rowH: 0.52, symW: 1.25, fs: 15 });
  card(s, X0, 5.55, CW, 0.85);
  s.addText([{ text: "Meaning.  ", options: { bold: true, color: C.text2 } },
    { text: "(2) fixes the vocabulary size, (3) covers every character exactly once, (4) emits only vocabulary tokens. Rounding plus exact inference turns any LP solution into a real tokenizer." }],
  { x: X0 + 0.3, y: 5.55, w: CW - 0.6, h: 0.85, fontSize: 15.5, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
  cite(s, "Constraints (2)–(5) follow ToaST [1].");

  // ======================= 9. Aggregate columns
  s = slide("Methodology", "Methodology · multi-objective layer", "Every objective becomes one column of the LP",
    "Our multi-objective layer. Each objective is an extra LP column defined by a single equality row: per-language cost T, spurious boundaries M, the data-weighted totals, and the worst-language cost t through the rows T_l <= t. Because everything is linear in z, adding objectives keeps it an LP.");
  pill(s, "OURS", X0, TOP - 0.02);
  await eqn(s, "T_\\ell \\;=\\; \\frac{1000\\,\\kappa_\\ell}{C_\\ell} \\sum_{k} c_{j(k),\\ell}\\; z_k", 6, X0, TOP + 0.75, 6.9, 24);
  await eqn(s, "M_\\ell \\;=\\; \\frac{1000\\,\\kappa_\\ell}{C_\\ell} \\sum_{k} c_{j(k),\\ell}\\; \\mu_k\\, z_k", 7, X0, TOP + 1.95, 6.9, 24);
  await eqn(s, "T_{\\mathrm{tot}} = \\sum_\\ell \\pi_\\ell\\, T_\\ell, \\qquad M_{\\mathrm{tot}} = \\sum_\\ell \\pi_\\ell\\, M_\\ell", 8, X0, TOP + 3.05, 6.9, 24);
  await eqn(s, "T_\\ell \\;\\le\\; t \\qquad \\forall\\, \\ell", 9, X0, TOP + 3.95, 6.9, 24);
  await defs(s, [
    ["\\ell", "language (six)"],
    ["T_\\ell", "tokens per 1000 English-equivalent characters"],
    ["M_\\ell", "spurious morpheme boundaries, same units"],
    ["C_\\ell", "characters of the language covered by trees"],
    ["\\kappa_\\ell", "content normalisation from FLORES+ (ta 1.17)"],
    ["c_{j,\\ell}", "count of a pretoken in a language"],
    ["j(k)", "pretoken (tree) that a node belongs to"],
    ["\\mu_k", "1 if the node starts inside a morpheme"],
    ["\\pi_\\ell", "data share of the language"],
    ["t", "worst-language cost"],
  ], 8.1, TOP + 0.4, 4.63, { rowH: 0.45, symW: 0.95, fs: 14 });

  // ======================= 10. Scalarisations
  s = slide("Methodology", "Methodology · scalarisation", "Tracing fronts by changing a few costs or bounds",
    "Weighted sums, epsilon-constraints and the min-max anchor only change costs or bounds of the aggregate columns. Bound changes keep the old basis dual feasible, so dual simplex warm-starts; cost changes keep it primal feasible. And the reduced cost of the bounded column is the local exchange rate between objectives, for free.");
  const sc = [["Weighted sum", "\\min\\; T_{\\mathrm{tot}} + \\lambda\\, t + \\nu\\, M_{\\mathrm{tot}}", 10],
    ["ε-constraint", "\\min\\; T_{\\mathrm{tot}} \\quad \\text{s.t.} \\quad t \\le \\tau \\ \\ \\big(\\text{or } M_{\\mathrm{tot}} \\le \\varepsilon\\big)", 11],
    ["Min-max anchor", "\\min\\; t + \\delta\\, T_{\\mathrm{tot}}", 12]];
  for (let i = 0; i < 3; i++) {
    const yc = TOP + 0.45 + i * 1.0;
    s.addText(sc[i][0], { x: X0, y: yc - 0.25, w: 2.2, h: 0.5, fontSize: 16, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true });
    await eqn(s, sc[i][1], sc[i][2], X0 + 2.3, yc, 5.3, 22);
  }
  await defs(s, [
    ["\\lambda,\\ \\nu \\ge 0", "weights of parity and morphology"],
    ["\\tau,\\ \\varepsilon", "bounds swept from loose to tight"],
    ["\\delta", "small tie-break weight (0.001)"],
  ], X0, TOP + 3.45, 7.6, { rowH: 0.45, symW: 1.6, fs: 15 });
  card(s, 8.55, TOP, 4.18, 4.9, C.text2);
  s.addText("Two properties we exploit", { x: 8.85, y: TOP + 0.2, w: 3.7, h: 0.4, fontSize: 17, bold: true, color: C.accent1, margin: 0, isTextBox: true });
  s.addText("Warm starts: a bound change keeps the basis dual feasible (dual simplex); a cost change keeps it primal feasible.", { x: 8.85, y: TOP + 0.7, w: 3.6, h: 1.25, fontSize: 15, color: C.background1, margin: 0, valign: "top", isTextBox: true });
  s.addText("Exchange rate for free:", { x: 8.85, y: TOP + 2.05, w: 3.6, h: 0.35, fontSize: 15, color: C.background1, margin: 0, isTextBox: true });
  await math(s, "\\frac{\\partial\\, T^{*}_{\\mathrm{tot}}}{\\partial\\, \\tau} \\;=\\; r_t \\;\\le\\; 0", 8.85, TOP + 2.9, 24, { color: H.lt1 });
  s.addText("(13)", { x: 11.95, y: TOP + 2.7, w: 0.6, h: 0.4, fontSize: 15, color: C.accent6, margin: 0, align: "right", isTextBox: true });
  await math(s, "r_t", 8.85, TOP + 3.72, 17, { color: H.lt1 });
  s.addText("reduced cost of the worst-language column at its bound: total tokens gained per unit of fairness given up.", { x: 9.3, y: TOP + 3.55, w: 3.15, h: 1.25, fontSize: 14, color: "C9CBD3", margin: 0, valign: "top", isTextBox: true });
  cite(s, "Exchange rate (13) = LP sensitivity of the optimal value to the bound; derivation in backup (B.3). Solver: HiGHS [11].");

  // ======================= 11. Certificates
  s = slide("Methodology", "Methodology · certificates", "Every Pareto point comes with an optimality certificate",
    "The LP gives a lower bound, the rounded tokenizer evaluated exactly gives an upper bound, and the gap certifies the point. Our bound shading fixes the case where rounding overshoots an epsilon-bound: tighten by the overshoot and re-solve warm, so the certificate is valid at every point.");
  const steps = [["Solve the LP", "lower bound on the scalarised objective"], ["Round", "keep the tokens the LP fully selects; fill the remaining slots by LP contribution"],
    ["Evaluate exactly", "real split-tree inference: an upper bound; the gap certifies the point"], ["Bound shading", "if the rounded vocabulary violates its bound, tighten it and re-solve warm (at most 6 times):"]];
  steps.forEach(([t, d], i) => {
    const yy = TOP + 0.05 + i * 1.0;
    s.addShape(pres.shapes.OVAL, { x: X0, y: yy, w: 0.56, h: 0.56, fill: { color: i === 3 ? C.accent1 : C.accent2 }, line: { color: i === 3 ? C.accent1 : C.accent2 } });
    s.addText(String(i + 1), { x: X0, y: yy, w: 0.56, h: 0.56, fontSize: 17, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(t, { x: X0 + 0.8, y: yy - 0.06, w: 6.4, h: 0.36, fontSize: 17, bold: true, color: C.text2, margin: 0, isTextBox: true });
    s.addText(d, { x: X0 + 0.8, y: yy + 0.3, w: 6.4, h: 0.6, fontSize: 15, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  });
  pill(s, "OURS", X0 + 3.0, TOP + 3.0);
  await eqn(s, "\\tau' \\;=\\; \\tau - 1.25\\,\\big(t(\\hat{\\mathbf{x}}) - \\tau\\big) - 10^{-4}\\,\\tau", 14, X0 + 0.8, TOP + 4.4, 6.5, 22);
  await defs(s, [["\\hat{\\mathbf{x}},\\ t(\\hat{\\mathbf{x}})", "rounded vocabulary and its exact worst-language cost"]], X0 + 0.8, TOP + 4.85, 6.5, { head: false, symW: 1.5, fs: 14 });
  card(s, 7.95, TOP, 4.78, 4.95);
  await stat(s, "14 / 20  →  0 / 123", "Pareto points whose rounded vocabulary violated its bound (before → after bound shading)", 8.25, TOP + 0.2, 4.3, C.accent1);
  await stat(s, "|\\mathrm{gap}| \\le 3 \\times 10^{-4}", "largest integrality gap over all 123 points of the three main runs", 8.25, TOP + 1.75, 4.3, C.accent2, true);
  await stat(s, "123 / 123", "points where the LP bound lies below the integer value, with monotone fronts (automatic checks)", 8.25, TOP + 3.3, 4.3, C.accent3);

  // ======================= 12. Experimental setup
  pres.addSection({ title: "Experimental setup" });
  s = slide("Experimental setup", "Experimental setup", "Controlled comparison: only vocabulary selection differs",
    "All tokenizers share the GPT-4o pretokenizer, the training text and a 32 000-token vocabulary. Baselines include BPE and Unigram with temperature sampling, Parity-aware BPE with the authors' code, and our MorphBPE reimplementation. LP variants: head-only trees at several scales, our SAA version, and a morph-first ablation.");
  const colsS = [["FaRandom", "Baselines (13)", C.accent4], ["FaFlask", "LP variants", C.accent2], ["FaServer", "Environment", C.accent3]];
  for (let i = 0; i < 3; i++) {
    const x = X0 + i * 4.1;
    card(s, x, TOP, 3.85, 3.55);
    await iconCircle(s, colsS[i][0], x + 0.25, TOP + 0.2, 0.6, colsS[i][2]);
    s.addText(colsS[i][1], { x: x + 1.0, y: TOP + 0.2, w: 2.7, h: 0.6, fontSize: 18, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true });
  }
  T(s, bullets(["BPE and Unigram [12] at 5 temperatures", "Parity-aware BPE [3], authors' code", "MorphBPE [4], reimplemented"], 15, 6), X0 + 0.25, TOP + 1.0, 3.45, 1.3);
  await eqn(s, "q_\\ell \\;\\propto\\; p_\\ell^{\\,\\alpha}", 15, X0 + 0.25, TOP + 2.55, 3.4, 22);
  await math(s, "\\alpha", X0 + 0.25, TOP + 3.12, 15, { color: H.accent6 });
  s.addText("temperature: 1 = natural mix, 0 = uniform over languages", { x: X0 + 0.55, y: TOP + 2.97, w: 3.15, h: 0.32, fontSize: 12.5, color: C.accent6, margin: 0, valign: "middle", isTextBox: true });
  T(s, bullets(["Head-only trees: 5k, 10k, 20k, 40k per language", "SAA: head + sampled tail trees (5k+5k, 10k+10k, 20k+20k)", "Ablation: morph-first trees", "8 points per sweep + 3 anchors + 3×3 grid"], 15, 6), X0 + 4.35, TOP + 1.0, 3.45, 2.5);
  T(s, bullets(["96 CPU cores, 1 TB RAM, 3 × RTX A6000", "HiGHS 1.16 dual / primal simplex [11]", "Python, PyTorch, Hugging Face tokenizers", "LMs: 35.5 M params, 3 seeds each"], 15, 6), X0 + 8.45, TOP + 1.0, 3.45, 2.5);
  card(s, X0, TOP + 3.8, CW, 1.1, C.text2);
  s.addText([{ text: "Protocol.  ", options: { bold: true, color: C.accent1 } },
    { text: "Same pretokenizer, training text and vocabulary size (32 000) for every tokenizer. Every LP point is evaluated as a real tokenizer on held-out text, never by its LP value alone.", options: { color: C.background1 } }],
  { x: X0 + 0.3, y: TOP + 3.8, w: CW - 0.6, h: 1.1, fontSize: 16, margin: 0, valign: "middle", isTextBox: true });

  // ======================= 13. Metrics
  s = slide("Experimental setup", "Experimental setup · metrics", "How we measure compression, fairness and model quality",
    "Held-out cost per language in the same units as the LP. Gini across languages for evenness. Hypervolume for whole fronts. Bits per byte for language models, which does not depend on the tokenizer.");
  const met = [["\\hat{T}_\\ell \\;=\\; 1000\\,\\kappa_\\ell\\,\\frac{\\#\\mathrm{tokens}_\\ell}{\\#\\mathrm{chars}_\\ell}", 16, "held-out cost per language"],
    ["\\mathrm{Gini} \\;=\\; \\frac{\\sum_{a,b}\\big|\\hat{T}_a - \\hat{T}_b\\big|}{2L^{2}\\,\\bar{T}}", 17, "evenness across languages"],
    ["\\mathrm{HV}(\\mathcal{P}) \\;=\\; \\frac{\\operatorname{vol}\\{\\mathbf{y} : \\exists\\, \\mathbf{p} \\in \\mathcal{P},\\ \\mathbf{p} \\le \\mathbf{y} \\le \\mathbf{r}\\}}{\\operatorname{vol}\\,[\\mathbf{y}^{\\mathrm{ideal}},\\, \\mathbf{r}]}", 18, "quality of a whole front"],
    ["\\mathrm{bpb} \\;=\\; \\frac{-\\sum_{n} \\log_2 p\\,(w_n \\mid w_{<n})}{\\#\\,\\text{UTF-8 bytes}}", 19, "language-model quality"]];
  for (let i = 0; i < 4; i++) {
    const yc = TOP + 0.5 + i * 1.12;
    await eqn(s, met[i][0], met[i][1], X0, yc, 7.2, 21);
  }
  await defs(s, [
    ["L,\\ \\bar{T}", "number of languages (6) and mean cost"],
    ["\\mathcal{P}", "objective vectors of a set of tokenizers"],
    ["\\mathbf{r}", "reference point, 1.02 × nadir"],
    ["\\mathbf{y}^{\\mathrm{ideal}}", "best value of each objective"],
    ["w_n", "tokens of the evaluation text"],
  ], 8.2, TOP + 0.45, 4.53, { rowH: 0.5, symW: 1.05, fs: 14.5 });
  card(s, 8.2, TOP + 3.15, 4.53, 1.75);
  s.addText("Also reported", { x: 8.45, y: TOP + 3.25, w: 4.1, h: 0.35, fontSize: 15, bold: true, color: C.text2, margin: 0, isTextBox: true });
  T(s, bullets(["FLORES+ tokens per sentence", "gold boundary F1 (MorphScore [6])", "LP gaps and solve times"], 14, 2), 8.45, TOP + 3.65, 4.1, 1.2);

  // ======================= 14. Results 1: head-only fronts
  pres.addSection({ title: "Results" });
  s = slide("Results", "Results · head-only LP", "On the training side, fairness looks almost free",
    "The standard ToaST problem with trees for the 10k most frequent pretokens per language. On the training side the parity front is very favourable: the worst language gets 16 percent cheaper for 0.4 percent more total tokens. All points are near-integral and weighted sums agree with epsilon-constraints. But these are training-side numbers.");
  y = await fig(s, "s_train_headonly.png", X0 + 0.05, TOP - 0.1, 12.0, "Training-side Pareto fronts of the head-only LP: compression vs worst-language cost and vs spurious morpheme boundaries");
  caption(s, "Figure 4. Training-side fronts, head-only LP (10k trees per language, 769 k node variables). The line is the LP lower bound; markers are rounded vocabularies.", X0, TOP - 0.1 + y + 0.05, CW);
  const st14 = [["−16 %", "worst-language cost for +0.4 % total tokens", C.accent3, false], ["−21 %", "at the min-max anchor (+8.3 % total)", C.accent3, false],
    ["|\\mathrm{gap}| \\le 3 \\times 10^{-4}", "over all 44 points; both scalarisations agree", C.accent2, true]];
  for (let i = 0; i < 3; i++) await stat(s, st14[i][0], st14[i][1], X0 + i * 4.1, 5.45, 3.8, st14[i][2], st14[i][3]);

  // ======================= 15. Results 2: failure on held-out
  s = slide("Results", "Results · held-out evaluation", "…but the head-only fronts do not transfer",
    "On held-out text the picture flips: the LP is clearly worse than BPE, and the fairest training-side point is the least fair. The reason: trees cover only 66 to 89 percent of the text. The LP never sees the long tail, so it spends the vocabulary on whole frequent words.");
  table(s, ["Tokenizer", "FineWeb: mean / worst", "FLORES+: mean / worst"], [
    ["BPE (natural mix)", "308.9 / 372.1", "39.5 / 48.4"],
    ["BPE (uniform mix)", "296.8 / 336.3", "37.8 / 41.2"],
    ["head-only LP, compression anchor", "348.8 / 406.6", "44.7 / 51.7"],
    ["head-only LP, best parity point", "343.3 / 404.4", "44.0 / 49.4"],
    [{ text: "head-only LP, min-max anchor", bold: true }, { text: "354.3 / 490.3", bold: true }, { text: "45.5 / 61.0", bold: true }],
  ], X0, TOP, 7.45, [3.45, 2.0, 2.0], { size: 15, rowH: 0.5 });
  caption(s, "Table 3. Held-out cost (FineWeb: tokens per 1000 characters, eq. 16; FLORES+: tokens per sentence). Lower is better.", X0, TOP + 3.1, 7.45, 0.55);
  card(s, X0, TOP + 3.8, 7.45, 1.15, C.text2);
  s.addText([{ text: "Diagnosis.  ", options: { bold: true, color: C.accent1 } },
    { text: "The LP only sees the head of the distribution, spends the vocabulary on whole frequent words, and buys “fairness” with whole words of the worst language.", options: { color: C.background1 } }],
  { x: X0 + 0.3, y: TOP + 3.8, w: 6.85, h: 1.15, fontSize: 15.5, margin: 0, valign: "middle", isTextBox: true });
  y = await fig(s, "s_coverage.png", 8.3, TOP - 0.05, 4.43, "Bar chart: share of text covered by trees per language, 66 to 89 percent");
  caption(s, "Figure 5. Share of pretoken occurrences that have a tree (head-only, 10k per language).", 8.3, TOP - 0.05 + y + 0.08, 4.43, 0.55);

  // ======================= 16. SAA fix
  s = slide("Results", "Results · fixing the objective", "Fix: estimate the cost of all text by sampling the tail",
    "Our fix is a sample average approximation. Keep exact trees for the head, and add trees for tail pretokens drawn with probability proportional to their count, each weighted by tail mass over S. For any fixed vocabulary this is an unbiased estimate of the token count over all text.");
  pill(s, "OURS", X0, TOP - 0.02);
  await eqn(s, "\\hat{C}_\\ell(\\mathbf{x}) \\;=\\; \\sum_{j \\in H} c_{j,\\ell}\\; n_j(\\mathbf{x}) \\;+\\; \\frac{W_\\ell}{S} \\sum_{s=1}^{S} n_{j_s}(\\mathbf{x}), \\qquad j_s \\sim \\frac{c_{\\cdot,\\ell}}{W_\\ell}", 20, X0, TOP + 0.85, CW, 26, "c");
  await defs(s, [
    ["H", "head: the 10k most frequent pretokens per language (exact trees)"],
    ["n_j(\\mathbf{x})", "number of tokens a pretoken gets under the vocabulary"],
    ["W_\\ell", "tail mass: occurrences of pretokens outside the head"],
    ["S,\\ j_s", "number of tail samples (10k) and the sampled pretokens"],
  ], X0 + 0.9, TOP + 2.15, 7.4, { rowH: 0.5, symW: 1.3, fs: 15.5 });
  card(s, 8.8, TOP + 1.8, 3.93, 2.4);
  s.addText("Property", { x: 9.05, y: TOP + 1.95, w: 3.5, h: 0.35, fontSize: 15, bold: true, color: C.text2, margin: 0, isTextBox: true });
  await math(s, "\\mathbb{E}\\big[\\hat{C}_\\ell(\\mathbf{x})\\big] = C_\\ell(\\mathbf{x})", 9.05, TOP + 2.65, 21);
  s.addText("for any fixed vocabulary (proof: backup B.1). The objective now covers all text, not only the head.", { x: 9.05, y: TOP + 3.05, w: 3.5, h: 1.0, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  T(s, "But the vocabulary is not fixed: the LP chooses it using the same sample. Next slide: what goes wrong, and how we fix it.", X0, TOP + 4.55, CW, 0.6, { fontSize: 16, italic: true, color: C.accent6 });
  cite(s, "Sample average approximation: Kleywegt et al. [13].");

  // ======================= 17. Optimizer's curse
  s = slide("Results", "Results · fixing the objective", "Naive sampling falls into the optimizer's curse",
    "Because the LP picks the vocabulary with the sample, the estimate at the optimum is biased low: the optimizer's curse. With any frequent n-gram allowed as a token, each sampled tail word weighs as much as a top-2000 word and the LP buys tokens for individual samples. Restricting tail trees to tokens that occur in head trees removes the bias: estimates within 1 to 4 percent.");
  await eqn(s, "\\mathbb{E}\\Big[\\min_{\\mathbf{x}} \\hat{C}(\\mathbf{x})\\Big] \\;\\le\\; \\min_{\\mathbf{x}} \\mathbb{E}\\big[\\hat{C}(\\mathbf{x})\\big] \\;=\\; \\min_{\\mathbf{x}} C(\\mathbf{x})", 21, X0, TOP + 0.4, 7.5, 22);
  table(s, ["Tail trees may use…", "en", "hi", "mr", "ta", "ru", "hu"], [
    ["any frequent n-gram", "232 / 269", "290 / 302", "293 / 312", "379 / 396", { text: "290 / 381", bold: true, color: C.accent4 }, { text: "289 / 353", bold: true, color: C.accent4 }],
    ["head-tree tokens only", "272 / 277", "284 / 288", "282 / 289", "341 / 351", "357 / 369", "332 / 340"],
  ], X0, TOP + 1.05, 7.6, [2.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], { size: 13, rowH: 0.46 });
  caption(s, "Table 4. LP estimate / exact cost over the full training distribution (5k head + 5k tail trees per language).", X0, TOP + 2.55, 7.6);
  s.addText("Fix (ours): tail trees only re-weight the objective and never add candidate tokens", { x: X0, y: TOP + 3.15, w: 7.5, h: 0.4, fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true });
  await eqn(s, "x_i = 0 \\qquad \\forall\\, i \\notin \\mathrm{tok}\\big(\\text{head trees}\\big)", 22, X0, TOP + 4.0, 7.5, 21);
  card(s, 8.45, TOP, 4.28, 4.95);
  await stat(s, "24 % → 1–4 %", "error of the LP's own estimate (Russian, before → after the fix)", 8.75, TOP + 0.2, 3.8, C.accent1);
  await stat(s, "−8 %", "held-out tokens vs head-only, with half the trees", 8.75, TOP + 1.75, 3.8, C.accent2);
  await stat(s, "0", "mismatches between LP model and real tokenizer on 20 000 trees", 8.75, TOP + 3.3, 3.8, C.accent3);
  cite(s, "Optimizer's curse: Smith & Winkler [14]; inequality (21) proved in backup (B.2).");

  // ======================= 18. Fronts transfer
  s = slide("Results", "Results · corrected objective", "With the corrected objective, the fronts transfer",
    "Main result. The SAA LP's parity front now holds on unseen text: moving from the compression anchor to the min-max anchor lowers the worst-language cost by 13 percent on FineWeb and 14 percent on FLORES while the mean stays flat. The min-max tokenizer is the most even of all tokenizers. Parity-aware BPE is still fairer in absolute terms.");
  y = await fig(s, "s_fronts_heldout.png", X0 + 0.6, TOP - 0.15, 10.9, "Held-out Pareto fronts on FineWeb and FLORES: total cost vs worst-language cost for LP runs and baselines");
  caption(s, "Figure 6. Held-out fronts: (a) FineWeb text, (b) FLORES+ parallel sentences. Lower-left is better. Dark blue: SAA LP; light blue: head-only LP.", X0, TOP - 0.15 + y + 0.04, CW);
  const st18 = [["−13 % / −14 %", "worst-language cost along the front (FineWeb / FLORES+), mean unchanged", C.accent3],
    ["0.004", "Gini across languages at the min-max point: lowest of all tokenizers", C.accent2], ["0.08 → 0.75", "held-out hypervolume: head-only → SAA (BPE family: 0.78)", C.accent1]];
  for (let i = 0; i < 3; i++) {
    s.addText(st18[i][0], { x: X0 + i * 4.1, y: 5.78, w: 3.85, h: 0.5, fontSize: 24, color: st18[i][2], margin: 0, isTextBox: true, fontFace: THEME.headFontFace });
    s.addText(st18[i][1], { x: X0 + i * 4.1, y: 6.24, w: 3.85, h: 0.45, fontSize: 12.5, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  }

  // ======================= 19. Baselines comparison
  s = slide("Results", "Results · comparison with baselines", "Where the LP wins, and where it does not",
    "Honest comparison. The LP min-max tokenizer is the most even and beats BPE at every temperature on worst-language FineWeb cost. BPE with uniform sampling still has the best mean, Parity-aware BPE the best absolute fairness, and MorphBPE and Unigram the best morphology. Hypervolume shows the SAA LP closing most of the gap to the BPE family.");
  const g = (t) => ({ text: t, bold: true, color: C.accent3 });
  table(s, ["Tokenizer", "FineWeb mean / worst", "Gini", "FLORES+ mean / worst", "Morph F1"], [
    ["SAA LP, compression anchor", "318.6 / 373.7", "0.069", "40.8 / 49.0", "0.19"],
    ["SAA LP, parity point", "315.5 / 342.3", "0.047", "40.3 / 43.9", "0.19"],
    [{ text: "SAA LP, min-max anchor", bold: true }, "321.4 / 325.9", g("0.004"), "40.9 / 42.3", "0.19"],
    ["BPE (natural mix)", "308.9 / 372.1", "0.070", "39.5 / 48.4", "0.27"],
    ["BPE (uniform mix)", g("296.8 / 336.3"), "0.044", g("37.8 / 41.2"), "0.29"],
    ["Parity-aware BPE", "300.4 / 312.1", "0.018", "38.1 / 38.4", "0.29"],
    ["MorphBPE", "328.0 / 366.3", "0.048", "42.2 / 47.7", g("0.54")],
    ["Unigram (natural mix)", "349.2 / 379.9", "0.030", "44.2 / 46.1", "0.50"],
  ], X0, TOP - 0.05, 7.85, [2.65, 1.65, 0.75, 1.75, 1.05], { size: 14, rowH: 0.47 });
  caption(s, "Table 5. Held-out comparison; lower is better except morphology F1 (MorphScore gold). Best mean values in teal; Parity-aware BPE has the best worst-language values (312.1, 38.4).", X0, TOP + 4.25, 7.85, 0.6);
  y = await fig(s, "s_hv.png", 8.65, TOP - 0.05, 4.08, "Bar chart: normalised hypervolume of LP runs and baselines");
  caption(s, "Figure 7. Normalised hypervolume (eq. 18), FineWeb compression × worst language. Higher is better.", 8.65, TOP - 0.05 + y + 0.08, 4.08, 0.6);

  // ======================= 20. Morphology
  s = slide("Results", "Results · morphology", "Morphology is a weak lever inside split trees",
    "Along the SAA morphology sweep, gold boundary F1 rises from 0.19 to 0.25 while total tokens rise 17 percent. MorphBPE and Unigram stay far ahead. Our interpretation: the LP can only choose among pieces the count-based trees offer, and morpheme-shaped pieces are often not tree nodes. The ablation on the next slide tests exactly this.");
  y = await fig(s, "s_morph.png", X0, TOP - 0.1, 7.5, "Held-out morphology front: total cost vs gold morpheme-boundary F1");
  caption(s, "Figure 8. Held-out gold morphology (MorphScore v2 [6], six languages) against compression. Higher F1 is better.", X0, TOP - 0.1 + y + 0.05, 7.5, 0.55);
  pill(s, "MEASURED", 8.5, TOP + 0.05, C.accent2);
  T(s, bullets(["LP morph sweep: F1 0.19 → 0.25 at +17 % tokens", "MorphBPE 0.54 and Unigram 0.50 remain far ahead"], 16, 8), 8.5, TOP + 0.45, 4.23, 1.6);
  pill(s, "INTERPRETATION", 8.5, TOP + 2.25, C.accent6);
  T(s, "The LP can only choose among pieces the count-based trees offer; morpheme-shaped pieces are often not nodes of those trees.", 8.5, TOP + 2.65, 4.23, 1.5, { fontSize: 16 });

  // ======================= 21. Language models
  s = slide("Results", "Results · downstream language models", "Language models disagree with intrinsic metrics",
    "35-million-parameter GPTs trained on identical bytes, three seeds each. Differences are small, under 8 percent. Morphology-aligned tokenizers are best; the most compressive tokenizer is the worst baseline; and our SAA LP tokenizers, although better compressors than the head-only LP, give worse language models. The explanation is a hypothesis, not a measured fact.");
  y = await fig(s, "s_lm.png", X0, TOP - 0.1, 7.5, "Horizontal bar chart of bits per byte per tokenizer with standard deviations over three seeds");
  caption(s, "Figure 9. Bits per byte (eq. 19) on held-out FineWeb, mean of six languages, ± standard deviation over 3 seeds. 35.5 M-parameter GPT, identical 144 MB of text.", X0, TOP - 0.1 + y + 0.05, 7.5, 0.55);
  pill(s, "MEASURED", 8.5, TOP + 0.05, C.accent2);
  T(s, bullets(["Best: MorphBPE 1.110; most compressive BPE is the worst baseline (1.153)", "SAA LP: better compression than head-only LP, worse LM (1.19 vs 1.14)", "A fairer tokenizer does not give a fairer LM (worst-language bits vary by 2–6 %)"], 15, 8), 8.5, TOP + 0.45, 4.23, 3.0);
  pill(s, "HYPOTHESIS", 8.5, TOP + 3.55, C.accent6);
  T(s, "Split-tree inference cuts mid-frequency words into word-dependent fragments (e.g. an|i|ma|tion), which may be harder to model.", 8.5, TOP + 3.95, 4.23, 1.2, { fontSize: 14.5 });

  // ======================= 22. Ablation
  s = slide("Results", "Results · ablation", "Ablation: morph-first trees fix training, not inference",
    "Trees that split at Morfessor boundaries first remove the compression-morphology conflict on the training side entirely: zero spurious boundaries for 2 percent more tokens. Held out, compression collapses because unseen words need a segmenter at inference.");
  T(s, "Split positions prefer Morfessor boundaries: hierarchy [morph, grapheme, char], 10k trees per language.", X0, TOP - 0.05, CW, 0.45, { fontSize: 16 });
  const ab = [[C.accent3, "FaCheckCircle", "Training side", [["0", "spurious morpheme boundaries for +2.2 % tokens (min-morph anchor)"], ["4× fewer", "spurious boundaries at the compression anchor (2.0 vs 7.8)"]]],
    [C.accent4, "FaExclamationTriangle", "Held-out text", [["423 vs 349", "FineWeb tokens per 1000 chars, morph-first vs count-based"], ["69 vs 52", "worst-language FLORES+ tokens per sentence"]]]];
  for (let i = 0; i < 2; i++) {
    const x = X0 + i * 6.2;
    card(s, x, TOP + 0.6, 5.93, 3.75);
    await iconCircle(s, ab[i][1], x + 0.3, TOP + 0.8, 0.62, ab[i][0]);
    s.addText(ab[i][2], { x: x + 1.15, y: TOP + 0.8, w: 4.5, h: 0.62, fontSize: 20, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true });
    for (let k = 0; k < 2; k++) await stat(s, ab[i][3][k][0], ab[i][3][k][1], x + 0.3, TOP + 1.65 + k * 1.35, 5.3, ab[i][0]);
  }
  T(s, "Gold F1 improves only modestly (0.29 vs 0.24). Morphology-first segmentation is costly at inference, consistent with MUTANT [5].", X0, TOP + 4.55, CW, 0.6, { fontSize: 16 });

  // ======================= 23. Discussion
  pres.addSection({ title: "Discussion" });
  s = slide("Discussion", "Discussion", "What the evidence says",
    "Three columns. Why it works: the LP structure gives integrality and cheap warm starts. Limitations: the candidate set from split trees and the single SAA sample. Implications: what an objective optimises matters more than how well it is optimised, and intrinsic metrics do not predict LM quality.");
  const disc = [["Why it works", C.accent2, [["MEASURED", "Near-integral LP at every scale; warm starts make whole fronts affordable"], ["MEASURED", "SAA objective: LP estimate within 1–4 % of the true cost"]]],
    ["Limitations", C.accent4, [["MEASURED", "Behind BPE on mean, Parity-aware BPE on fairness, MorphBPE on morphology"], ["INTERPRETATION", "Bottleneck is the split-tree candidate set, not the optimiser"]]],
    ["Implications", C.accent1, [["INTERPRETATION", "What the LP optimises matters more than how well it optimises it"], ["MEASURED", "Compression and fairness do not predict LM quality"]]]];
  for (let i = 0; i < 3; i++) {
    const x = X0 + i * 4.1;
    card(s, x, TOP, 3.85, 4.95);
    s.addText(disc[i][0], { x: x + 0.3, y: TOP + 0.2, w: 3.3, h: 0.45, fontSize: 20, bold: true, color: disc[i][1], margin: 0, isTextBox: true, fontFace: THEME.headFontFace });
    for (let k = 0; k < 2; k++) {
      const yy = TOP + 0.95 + k * 1.95;
      pill(s, disc[i][2][k][0], x + 0.3, yy, disc[i][2][k][0] === "MEASURED" ? C.accent2 : C.accent6);
      s.addText(disc[i][2][k][1], { x: x + 0.3, y: yy + 0.38, w: 3.3, h: 1.4, fontSize: 16, color: C.text1, margin: 0, valign: "top", isTextBox: true });
    }
  }

  // ======================= 24. Conclusion
  pres.addSection({ title: "Conclusion" });
  s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Conclusion" });
  s.addText("CONCLUSION", { x: 0.8, y: 0.45, w: 11.7, h: 0.3, fontSize: 12, bold: true, color: C.accent1, charSpacing: 2, margin: 0, isTextBox: true });
  s.addText("Certified fronts that hold on unseen text", { x: 0.8, y: 0.75, w: 11.7, h: 0.7, fontSize: 32, color: C.background1, margin: 0, isTextBox: true, fontFace: THEME.headFontFace });
  s.addText("We cast multilingual vocabulary selection as one multi-objective LP, certified every Pareto point, and found that the objective, not the solver, decides whether fronts transfer.", { x: 0.8, y: 1.55, w: 11.7, h: 0.8, fontSize: 17, color: "C9CBD3", margin: 0, isTextBox: true });
  await math(s, "\\le 3 \\times 10^{-4}", 0.8, 2.97, 26, { color: H.accent1 });
  const cs = [["", "integrality gap, 123 points (RQ1)"], ["−13 / −14 %", "worst-language held-out cost along the front (RQ2, RQ3)"], ["0.004", "lowest cross-language Gini of all tokenizers (RQ3)"], ["1.110 bpb", "best LM: MorphBPE, morphology beats compression (RQ4)"]];
  cs.forEach(([b, l], i) => {
    const x = 0.8 + i * 2.95;
    s.addText(b, { x, y: 2.65, w: 2.8, h: 0.65, fontSize: 28, color: C.accent1, margin: 0, isTextBox: true, fontFace: THEME.headFontFace });
    s.addText(l, { x, y: 3.32, w: 2.7, h: 0.9, fontSize: 14, color: C.background1, margin: 0, valign: "top", isTextBox: true });
  });
  s.addText([
    { text: "Takeaways", options: { bold: true, color: C.accent1, breakLine: true } },
    { text: "Head-only fronts mislead; an unbiased, curse-free objective (SAA) makes them transfer.", options: { bullet: { indent: 18 }, breakLine: true } },
    { text: "The LP gives the most even tokenizer, but strong baselines still win on mean, absolute fairness and morphology.", options: { bullet: { indent: 18 }, breakLine: true } },
    { text: "Next: richer candidate sets (BPE / Unigram pieces as extra columns) and independent SAA re-estimation.", options: { bullet: { indent: 18 } } },
  ], { x: 0.8, y: 4.55, w: 11.7, h: 2.1, fontSize: 16.5, color: C.background1, margin: 0, paraSpaceAfter: 6, isTextBox: true });
  s.addNotes("Close with the problem, our solution and four numbers, one per research question.");

  // ======================= 25. References
  s = slide("Conclusion", "References", "References",
    "Papers, datasets and code used in this work.");
  const refs = [
    "[1] Schmidt et al. (2026). Tokenization with Split Trees (ToaST). arXiv:2605.22705.",
    "[2] Tempus et al. (2026). Tokenisation via Convex Relaxations. arXiv:2605.22821.",
    "[3] Foroutan et al. (2026). Parity-Aware Byte-Pair Encoding. ACL. github.com/swiss-ai/parity-aware-bpe",
    "[4] Asgari et al. (2026). MorphBPE. Findings of ACL. (reimplemented; no public code)",
    "[5] Rana et al. (2026). MUTANT: a multilingual tokenizer recipe for Indic languages. ACL.",
    "[6] Arnett & Bergen (2025); Arnett et al. (2025). MorphScore. huggingface.co/datasets/catherinearnett/morphscore",
    "[7] Penedo et al. (2024, 2025). FineWeb and FineWeb-2. huggingface.co/HuggingFaceFW",
    "[8] NLLB Team et al. (2022); OLDI. FLORES+. huggingface.co/datasets/openlanguagedata/flores_plus",
    "[9] Batsuren et al. (2022). The SIGMORPHON 2022 Shared Task on Morpheme Segmentation.",
    "[10] Virpioja et al. (2013). Morfessor 2.0: Python implementation and extensions for Morfessor Baseline.",
    "[11] Huangfu & Hall (2018). Parallelizing the dual revised simplex method (HiGHS). Math. Prog. Comp.",
    "[12] Sennrich et al. (2016). Subword units (BPE). ACL · Kudo (2018). Subword regularization (Unigram). ACL.",
    "[13] Kleywegt, Shapiro & Homem-de-Mello (2002). The sample average approximation method. SIAM J. Optim.",
    "[14] Smith & Winkler (2006). The optimizer's curse. Management Science.",
    "[15] Loshchilov & Hutter (2019). Decoupled weight decay regularization (AdamW). ICLR.",
    "[16] Our code, data processing and results: github.com/n6m9n/769",
  ];
  T(s, refs.slice(0, 8).map((r, i) => ({ text: r, options: { breakLine: i < 7, paraSpaceAfter: 9 } })), X0, TOP - 0.1, 5.95, 5.2, { fontSize: 13 });
  T(s, refs.slice(8).map((r, i) => ({ text: r, options: { breakLine: i < 7, paraSpaceAfter: 9 } })), X0 + 6.2, TOP - 0.1, 5.93, 5.2, { fontSize: 13 });

  // ======================= 26. Questions
  s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Conclusion" });
  s.addText("THANK YOU", { placeholder: "body" });
  s.addText("Questions?", { placeholder: "title" });
  s.addNotes("Backup slides follow, for questions.");

  // ======================= Backup
  pres.addSection({ title: "Backup slides" });
  s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Backup slides" });
  s.addText("APPENDIX", { placeholder: "body" });
  s.addText("Backup Slides — Q&A", { placeholder: "title" });
  s.addText("B1 main-run fronts · B2 derivations · B3 shadow prices · B4 scaling · B5 timings · B6 data statistics · B7 hyperparameters · B8 per-language results · B9 baselines and checks · B10 limitations", { x: 0.8, y: 4.5, w: 11.7, h: 0.9, fontSize: 15, color: "C9CBD3", margin: 0, isTextBox: true });

  // B1
  s = slide("Backup slides", "Backup · B1", "Training-side fronts of the main run (SAA 10k+10k)", "Main run, training side.");
  y = await fig(s, "s_train_saa.png", X0 + 0.05, TOP - 0.1, 12.0, "Training-side Pareto fronts of the SAA 10k+10k LP");
  caption(s, "Figure B1. SAA 10k+10k: 110 k trees, 1.7 M node variables, 2.6 M rows. 35 points, all relative gaps below 0.03 %, 0 bound violations, monotone fronts.", X0, TOP - 0.1 + y + 0.05, CW);
  T(s, bullets(["Worst language 370 → 333 (−10 %) for +0.5 % total; 314 (−15 %) at the min-max anchor (+12 %)", "Spurious boundaries 53.1 → 48.6 (−8 %) for +4.4 % total", "Anchors: compression 387 s, min-max 2 524 s, min-morph 672 s; warm sweep point median 260 s"], 14, 2), X0, 5.85, CW, 1.1);

  // B2 derivations
  s = slide("Backup slides", "Backup · B2", "Derivations", "Unbiasedness of the SAA estimator, the optimizer's curse, and the exchange rate as a reduced cost.");
  s.addText("SAA estimator is unbiased for a fixed vocabulary", { x: X0, y: TOP - 0.1, w: CW, h: 0.38, fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true });
  await eqn(s, "\\mathbb{E}\\Big[\\frac{W_\\ell}{S}\\sum_{s=1}^{S} n_{j_s}(\\mathbf{x})\\Big] \\;=\\; W_\\ell \\sum_{j \\notin H} \\frac{c_{j,\\ell}}{W_\\ell}\\, n_j(\\mathbf{x}) \\;=\\; \\sum_{j \\notin H} c_{j,\\ell}\\, n_j(\\mathbf{x})", "B.1", X0, TOP + 0.75, CW, 22);
  s.addText("Optimizer's curse (inequality 21)", { x: X0, y: TOP + 1.45, w: CW, h: 0.38, fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true });
  await eqn(s, "\\min_{\\mathbf{x}'} \\hat{C}(\\mathbf{x}') \\le \\hat{C}(\\mathbf{x})\\ \\ \\forall \\mathbf{x} \\;\\;\\Rightarrow\\;\\; \\mathbb{E}\\Big[\\min_{\\mathbf{x}'} \\hat{C}(\\mathbf{x}')\\Big] \\le \\min_{\\mathbf{x}} \\mathbb{E}\\big[\\hat{C}(\\mathbf{x})\\big] = \\min_{\\mathbf{x}} C(\\mathbf{x})", "B.2", X0, TOP + 2.25, CW, 22);
  s.addText("Exchange rate (13): LP sensitivity analysis", { x: X0, y: TOP + 2.95, w: CW, h: 0.38, fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true });
  await eqn(s, "T^{*}_{\\mathrm{tot}}(\\tau + \\Delta) \\;=\\; T^{*}_{\\mathrm{tot}}(\\tau) + r_t\\,\\Delta \\qquad \\text{while the optimal basis is unchanged}", "B.3", X0, TOP + 3.75, CW, 22);
  T(s, "The optimal value is convex and piecewise linear in the bound; its slope, the reduced cost in (13), is the dual value of that bound and is read directly from the simplex solution. The large weight per sample in (20) makes the gap in (B.2) severe; the candidate restriction (22) removes it.", X0, TOP + 4.35, CW, 0.9, { fontSize: 15 });

  // B3 shadow prices
  s = slide("Backup slides", "Backup · B3", "Shadow prices along the sweeps", "Exchange rates read off the reduced costs.");
  y = await fig(s, "s_shadow.png", X0 + 0.3, TOP, 11.5, "Shadow prices along parity and morphology sweeps of the main run");
  caption(s, "Figure B2. Total tokens saved per unit of relaxing each bound (negative reduced cost, eq. 13), SAA 10k+10k. Fairness is cheap near the compression anchor and expensive near min-max.", X0, TOP + y + 0.1, CW, 0.55);

  // B4 scaling
  s = slide("Backup slides", "Backup · B4", "Optimisation at scale", "Scaling of solve times and solver comparison.");
  y = await fig(s, "s_scaling.png", X0, TOP, 6.0, "Solve time versus LP size, log-log");
  caption(s, "Figure B3. Head-only LPs: cold anchors and warm sweep points vs size; time grows roughly as size to the power 1.4.", X0, TOP + y + 0.08, 6.0, 0.55);
  y = await fig(s, "s_solvers.png", 6.95, TOP, 5.78, "Bar chart: solve time of five HiGHS solvers on the same LP");
  caption(s, "Figure B4. HiGHS solvers on the same 0.36 M-node LP; dual simplex wins.", 6.95, TOP + y + 0.08, 5.78, 0.55);

  // B5 timings
  s = slide("Backup slides", "Backup · B5", "LP sizes and solve times", "Every LP run.");
  table(s, ["Run", "Trees", "Nodes", "Coverage mean / worst", "Compression", "Min-max", "Min-morph", "Warm point"], [
    ["head-only 5k", "27 k", "0.36 M", "71 / 59 %", "65 s", "–", "–", "–"],
    ["head-only 10k", "54 k", "0.77 M", "77 / 66 %", "312 s", "1 690 s", "222 s", "119 s"],
    ["head-only 20k", "109 k", "1.6 M", "83 / 73 %", "1 171 s", "8 708 s", "819 s", "361 s"],
    ["head-only 40k", "216 k", "3.4 M", "88 / 80 %", "3 323 s", "–", "–", "–"],
    ["SAA 5k+5k", "56 k", "0.83 M", "all text (sampled)", "76 s", "437 s", "122 s", "29 s"],
    ["SAA 10k+10k", "111 k", "1.7 M", "all text (sampled)", "387 s", "2 524 s", "672 s", "260 s"],
    ["SAA 20k+20k", "219 k", "3.5 M", "all text (sampled)", "2 012 s", "17 107 s", "1 170 s", "318 s"],
  ], X0, TOP, CW, [1.85, 0.95, 1.0, 2.33, 1.5, 1.5, 1.5, 1.5], { size: 15, rowH: 0.5 });
  caption(s, "Table B1. HiGHS 1.16, one process per anchor or sweep; warm point = median sweep point. All compression anchors are exactly integral.", X0, TOP + 4.15, CW);

  // B6 data statistics
  s = slide("Backup slides", "Backup · B6", "Dataset statistics and preprocessing", "Data details.");
  table(s, ["Language", "Lines", "Words", "Bytes", "Tree coverage 5k / 10k / 20k / 40k", "Gold words (MorphScore)"], [
    ["English", "980 k", "33.4 M", "199 MB", "82 / 87 / 92 / 95 %", "4 121"],
    ["Hindi", "166 k", "8.0 M", "100 MB", "84 / 89 / 93 / 96 %", "1 324"],
    ["Marathi", "69 k", "2.8 M", "50 MB", "70 / 77 / 83 / 88 %", "178"],
    ["Tamil", "43 k", "1.1 M", "25 MB", "59 / 66 / 73 / 80 %", "1 195"],
    ["Russian", "207 k", "5.8 M", "75 MB", "66 / 72 / 79 / 84 %", "21 575"],
    ["Hungarian", "196 k", "6.1 M", "50 MB", "67 / 72 / 78 / 83 %", "6 376"],
  ], X0, TOP, CW, [1.6, 1.2, 1.2, 1.3, 4.0, 2.83], { size: 15, rowH: 0.48 });
  caption(s, "Table B2. Training corpora after paragraph splitting (paragraphs > 20 characters); MorphScore words after its default filtering (we sample up to 3 000 per language).", X0, TOP + 3.5, CW, 0.55);
  T(s, bullets(["Pretokenizer: ToaST's length-limited GPT-4o regex; keeps Indic vowel signs attached", "Morfessor Baseline per language on the 50 k most frequent words: training signal only, never used for evaluation", "Content normalisation from FLORES+ dev (characters per English character): en 1.000, hi 1.001, mr 1.004, ta 1.167, ru 1.103, hu 1.055"], 15, 4), X0, TOP + 4.15, CW, 1.2);

  // B7 hyperparameters
  s = slide("Backup slides", "Backup · B7", "Hyperparameters and configurations", "All settings.");
  table(s, ["Component", "Setting"], [
    ["Trees", "hierarchy [grapheme, char]; language-balanced n-gram counts; pretokens up to 64 characters"],
    ["LP", "vocabulary 32 000 (31 744 + 256 bytes); HiGHS dual simplex; anchors cold, sweeps warm"],
    ["Sweeps", "8 bounds per ε-constraint sweep; parity weights 0.02–5 and morphology weights 0.05–20 (geometric); 3 × 3 grid; tie-break 0.001"],
    ["Certificates", "bound shading: at most 6 steps; warm-start time limit 7 200 s with cold fallback"],
    ["SAA", "tail samples per language = number of head trees; tail tokens restricted to head-tree candidates"],
    ["Baselines", "Parity-aware BPE: FLORES+ dev parity set, word types seen at least 3 times; MorphBPE: joint Morfessor, 20 k words per language"],
    ["LM", "GPT, 6 layers, width 512, 8 heads, context 512, tied embeddings; AdamW [15], learning rate 0.0015, betas 0.9 / 0.95, weight decay 0.1"],
    ["LM training", "64 × 512 tokens per step, 4 epochs, cosine to 10 %, grad clip 1.0, bf16; evaluation stride 256; 3 seeds"],
  ], X0, TOP, CW, [2.1, 10.03], { size: 14.5, rowH: 0.52 });
  caption(s, "Table B3. Configuration of all experiments (configs/full.yaml and scripts 02–07 of the repository).", X0, TOP + 4.75, CW);

  // B8 per-language
  s = slide("Backup slides", "Backup · B8", "Per-language results on FLORES+", "Per-language tokens per sentence.");
  table(s, ["Tokenizer", "English", "Hindi", "Marathi", "Tamil", "Russian", "Hungarian", "Worst"], [
    ["BPE (natural mix)", "30.1", "39.3", "38.9", "48.4", "41.8", "38.4", "48.4"],
    ["BPE (uniform mix)", "32.4", "39.7", "37.1", "40.1", "41.2", "36.2", "41.2"],
    ["Parity-aware BPE", "38.4", "37.8", "38.2", "38.2", "37.6", "38.1", "38.4"],
    ["MorphBPE", "36.5", "43.8", "41.1", "47.7", "44.4", "39.5", "47.7"],
    ["Unigram (natural mix)", "43.8", "46.0", "41.8", "45.6", "46.1", "41.7", "46.1"],
    ["SAA LP, compression", "31.3", "39.5", "40.2", "49.0", "43.8", "41.2", "49.0"],
    ["SAA LP, parity point", "31.8", "40.6", "41.5", "43.9", "41.6", "42.3", "43.9"],
    ["SAA LP, min-max", "40.0", "42.2", "42.3", "41.4", "39.1", "40.3", "42.3"],
  ], X0, TOP, CW, [2.93, 1.3, 1.3, 1.3, 1.3, 1.3, 1.4, 1.3], { size: 15, rowH: 0.48 });
  caption(s, "Table B4. Tokens per FLORES+ devtest sentence (lower is better). The LP min-max point equalises languages around 39–42 tokens.", X0, TOP + 4.45, CW);

  // B9 baselines and checks
  s = slide("Backup slides", "Backup · B9", "Baseline details and correctness checks", "Implementation details and the checks we ran.");
  card(s, X0, TOP, 5.95, 4.3);
  s.addText("Baselines", { x: X0 + 0.3, y: TOP + 0.2, w: 5.4, h: 0.4, fontSize: 18, bold: true, color: C.text2, margin: 0, isTextBox: true });
  T(s, bullets(["Same pretokenizer and 32 k vocabulary for all", "BPE / Unigram: Hugging Face tokenizers, temperature mix (15)", "Parity-aware BPE: authors' learner; base and window variants give the identical vocabulary", "MorphBPE: invisible separator at every Morfessor boundary, so no merge crosses a morpheme"], 15, 6), X0 + 0.3, TOP + 0.75, 5.4, 3.4);
  card(s, X0 + 6.18, TOP, 5.95, 4.3);
  s.addText("Checks (all passing)", { x: X0 + 6.48, y: TOP + 0.2, w: 5.4, h: 0.4, fontSize: 18, bold: true, color: C.text2, margin: 0, isTextBox: true });
  T(s, bullets(["LP bound below the integer value at 123 / 123 points; 0 bound violations", "Monotone training-side fronts (LP and integer)", "Exact round trip: 0 failures over 123 tokenizers × 1 804 strings", "LP model equals real tokenizer on 20 000 trees", "13 unit tests incl. brute-force LP optimality on a toy problem"], 15, 6), X0 + 6.48, TOP + 0.75, 5.4, 3.4);

  // B10 limitations
  s = slide("Backup slides", "Backup · B10", "Limitations and future work", "Open issues and next steps.");
  const lim = [["Candidate set", "Split-tree nodes only. Next: add BPE merges / Unigram pieces as extra LP columns, or per-language tree budgets."],
    ["SAA statistics", "One sample. Next: re-estimate the chosen vocabulary on an independent sample and report the SAA optimality gap [13]."],
    ["LM penalty of SAA tokenizers", "Unexplained (+4.5 % bpb). Next: measure segmentation consistency across words."],
    ["Evaluation scale", "35 M-parameter LMs, 25–30 M tokens; matched bytes give more steps to longer tokenizations."],
    ["Morphology data", "MorphBPE reimplemented; small Indic gold sets (Marathi: 178 words)."]];
  lim.forEach(([t, d], i) => {
    const yy = TOP + i * 0.98;
    s.addText(t, { x: X0, y: yy, w: 3.3, h: 0.85, fontSize: 16, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true });
    s.addText(d, { x: X0 + 3.4, y: yy, w: CW - 3.4, h: 0.85, fontSize: 15.5, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
  });
}

build().then(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("written", OUT);
}).catch((e) => { console.error(e); process.exit(1); });
