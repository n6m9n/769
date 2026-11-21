// TeX -> SVG (MathJax) -> high-resolution PNG. Returns {data, w, h} in inches for a given font size (pt).
const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { SVG } = require("mathjax-full/js/output/svg.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
const { AllPackages } = require("mathjax-full/js/input/tex/AllPackages.js");
const sharp = require("sharp");
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const doc = mathjax.document("", { InputJax: new TeX({ packages: AllPackages }), OutputJax: new SVG({ fontCache: "none" }) });
const EX_PER_PT = 1 / 2.26;   // MathJax: 1ex ~ 0.442em
async function tex(src, pt = 22, color = "1E1E24", display = true) {
  const node = doc.convert(src, { display, em: 16, ex: 8, containerWidth: 1200 });
  let svg = adaptor.innerHTML(node);
  const wEx = parseFloat(svg.match(/width="([\d.]+)ex"/)[1]);
  const hEx = parseFloat(svg.match(/height="([\d.]+)ex"/)[1]);
  const wPt = wEx * pt * EX_PER_PT, hPt = hEx * pt * EX_PER_PT;
  svg = svg.replace(/width="[\d.]+ex"/, `width="${wPt}pt"`).replace(/height="[\d.]+ex"/, `height="${hPt}pt"`)
           .replace(/currentColor/g, "#" + color);
  if (/data-mjx-error|merror/.test(svg)) throw new Error("TeX error in: " + src);
  const buf = await sharp(Buffer.from(svg), { density: 1200 }).png().toBuffer();
  return { data: "image/png;base64," + buf.toString("base64"), w: wPt / 72, h: hPt / 72, svg };
}
module.exports = { tex };
