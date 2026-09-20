// Usage: KATEX=/path/to/katex node tools/render_math.js < in.json > out.json
// in.json: {key: latex}; keys starting with "D:" render in display mode.
const katex = require(process.env.KATEX || "katex");
let s = ""; process.stdin.on("data", d => s += d).on("end", () => {
  const inp = JSON.parse(s), out = {};
  for (const [k, v] of Object.entries(inp))
    out[k] = katex.renderToString(v, {displayMode: k.startsWith("D:"), throwOnError: true, output: "htmlAndMathml"});
  process.stdout.write(JSON.stringify(out));
});
