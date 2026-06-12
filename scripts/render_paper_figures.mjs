#!/usr/bin/env node
/*
 * Regenerate PARC paper figures without a Python or TeX dependency.
 *
 * The script reads the checked-in CSV result tables and writes SVG sources,
 * high-resolution PNG files used by the paper, HTML renderers, and Chrome PDF
 * previews under figures/.  Chrome is used only as a deterministic SVG renderer.
 */
import fs from "node:fs/promises";
import fssync from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..");
const FIG_DIR = path.join(ROOT, "figures");
const DATA_DIR = path.join(ROOT, "data", "results");
const DPI = 600;

const LABELS = {
  no_repair: "No repair",
  majority: "Majority",
  provenance_unaware: "Prov.-unaware",
  iterative_truth: "Iter. truth",
  dependency_truth: "Dep.+truth",
  parc: "PARC",
};

const STYLE = {
  parc: { color: "#006D77", dash: "", marker: "circle", width: 2.6 },
  dependency_truth: { color: "#333333", dash: "", marker: "square", width: 2.0 },
  iterative_truth: { color: "#D55E00", dash: "7 4", marker: "diamond", width: 1.9 },
  provenance_unaware: { color: "#7B3F98", dash: "6 3 2 3", marker: "triangle", width: 1.9 },
  majority: { color: "#0072B2", dash: "5 4", marker: "square", width: 1.9 },
  no_repair: { color: "#777777", dash: "2 4", marker: "cross", width: 1.8 },
};

function csvParse(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (quoted) {
      if (ch === '"' && next === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (ch !== "\r") {
      field += ch;
    }
  }
  if (field.length || row.length) {
    row.push(field);
    rows.push(row);
  }
  const header = rows.shift();
  return rows
    .filter((r) => r.length && r.some((v) => v !== ""))
    .map((r) => Object.fromEntries(header.map((h, i) => [h, coerce(r[i] ?? "")])));
}

function coerce(value) {
  if (value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) && /^[-+]?\d*\.?\d+(e[-+]?\d+)?$/i.test(value) ? n : value;
}

async function readCsv(name) {
  return csvParse(await fs.readFile(path.join(DATA_DIR, name), "utf8"));
}

function esc(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function pt(x, y) {
  return `${x.toFixed(2)},${y.toFixed(2)}`;
}

function pathLine(points) {
  return points.map(([x, y]) => pt(x, y)).join(" ");
}

function text(x, y, value, opts = {}) {
  const {
    size = 18,
    color = "#222",
    anchor = "middle",
    weight = 400,
    rotate = 0,
    italic = false,
    baseline = "middle",
  } = opts;
  const transform = rotate ? ` transform="rotate(${rotate} ${x} ${y})"` : "";
  const style = italic ? "font-style:italic;" : "";
  return `<text x="${x}" y="${y}"${transform} font-size="${size}" font-weight="${weight}" fill="${color}" text-anchor="${anchor}" dominant-baseline="${baseline}" style="${style}">${esc(value)}</text>`;
}

function richText(x, y, spans, opts = {}) {
  const {
    size = 18,
    color = "#222",
    anchor = "middle",
    weight = 400,
    rotate = 0,
    italic = false,
    baseline = "middle",
  } = opts;
  const transform = rotate ? ` transform="rotate(${rotate} ${x} ${y})"` : "";
  const style = italic ? "font-style:italic;" : "";
  const body = spans
    .map((span) => {
      if (typeof span === "string") return esc(span);
      const attrs = [];
      if (span.size) attrs.push(`font-size="${span.size}"`);
      if (span.shift) attrs.push(`baseline-shift="${span.shift}"`);
      if (span.weight) attrs.push(`font-weight="${span.weight}"`);
      if (span.italic) attrs.push(`style="font-style:italic;"`);
      return `<tspan ${attrs.join(" ")}>${esc(span.text)}</tspan>`;
    })
    .join("");
  return `<text x="${x}" y="${y}"${transform} font-size="${size}" font-weight="${weight}" fill="${color}" text-anchor="${anchor}" dominant-baseline="${baseline}" style="${style}">${body}</text>`;
}

function rect(x, y, w, h, opts = {}) {
  const { fill = "#fff", stroke = "#333", sw = 1, rx = 4, opacity = 1 } = opts;
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}" opacity="${opacity}"/>`;
}

function line(x1, y1, x2, y2, opts = {}) {
  const { color = "#333", width = 1, dash = "", opacity = 1, markerEnd = "" } = opts;
  const dashAttr = dash ? ` stroke-dasharray="${dash}"` : "";
  const marker = markerEnd ? ` marker-end="url(#${markerEnd})"` : "";
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}"${dashAttr} opacity="${opacity}"${marker}/>`;
}

function polyline(points, opts = {}) {
  const { color = "#333", width = 2, dash = "", opacity = 1 } = opts;
  const dashAttr = dash ? ` stroke-dasharray="${dash}"` : "";
  return `<polyline points="${pathLine(points)}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round"${dashAttr} opacity="${opacity}"/>`;
}

function marker(cx, cy, method, scale = 1) {
  const s = STYLE[method] ?? STYLE.parc;
  const r = 4.1 * scale;
  if (s.marker === "circle") return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${s.color}" stroke="white" stroke-width="${1.1 * scale}"/>`;
  if (s.marker === "square") return `<rect x="${cx - r}" y="${cy - r}" width="${2 * r}" height="${2 * r}" fill="${s.color}" stroke="white" stroke-width="${0.9 * scale}"/>`;
  if (s.marker === "diamond") return `<path d="M ${cx} ${cy - 1.25 * r} L ${cx + 1.25 * r} ${cy} L ${cx} ${cy + 1.25 * r} L ${cx - 1.25 * r} ${cy} Z" fill="${s.color}" stroke="white" stroke-width="${0.8 * scale}"/>`;
  if (s.marker === "triangle") return `<path d="M ${cx} ${cy - 1.2 * r} L ${cx + 1.15 * r} ${cy + r} L ${cx - 1.15 * r} ${cy + r} Z" fill="${s.color}" stroke="white" stroke-width="${0.8 * scale}"/>`;
  return `${line(cx - r, cy - r, cx + r, cy + r, { color: s.color, width: 1.7 * scale })}${line(cx - r, cy + r, cx + r, cy - r, { color: s.color, width: 1.7 * scale })}`;
}

function linear(min, max, a, b) {
  return (v) => a + ((v - min) / (max - min)) * (b - a);
}

function logScale(min, max, a, b) {
  const lo = Math.log10(min);
  const hi = Math.log10(max);
  return (v) => a + ((Math.log10(Math.max(v, min)) - lo) / (hi - lo)) * (b - a);
}

function axes({ x0, y0, w, h, xTicks, yTicks, xScale, yScale, xFormat, yFormat, xLabel, yLabel }) {
  let out = "";
  for (const t of yTicks) {
    const y = yScale(t);
    out += line(x0, y, x0 + w, y, { color: "#D7DCE2", width: 0.75, opacity: 0.85 });
    out += text(x0 - 10, y + 1, yFormat(t), { size: 16, anchor: "end", color: "#333" });
  }
  for (const t of xTicks) {
    const x = xScale(t);
    out += line(x, y0, x, y0 + h, { color: "#EEF0F2", width: 0.6, opacity: 0.9 });
    out += line(x, y0 + h, x, y0 + h + 5, { color: "#333", width: 0.9 });
    out += text(x, y0 + h + 23, xFormat(t), { size: 16, color: "#333" });
  }
  out += line(x0, y0 + h, x0 + w, y0 + h, { color: "#222", width: 1.15 });
  out += line(x0, y0, x0, y0 + h, { color: "#222", width: 1.15 });
  out += text(x0 + w / 2, y0 + h + 52, xLabel, { size: 18, color: "#222" });
  out += text(x0 - 50, y0 + h / 2, yLabel, { size: 18, color: "#222", rotate: -90 });
  return out;
}

function seriesByMethod(rows, method, xField, yField, yFloor = null) {
  return rows
    .filter((r) => r.method === method)
    .sort((a, b) => a[xField] - b[xField])
    .map((r) => [r[xField], yFloor == null ? r[yField] : Math.max(yFloor, r[yField])]);
}

function ciBand(rows, method, xField, yField, sdField, xScale, yScale, n, color) {
  const data = rows
    .filter((r) => r.method === method)
    .sort((a, b) => a[xField] - b[xField])
    .map((r) => {
      const ci = r[sdField] == null ? 0 : (1.96 * r[sdField]) / Math.sqrt(n);
      return { x: xScale(r[xField]), upper: yScale(r[yField] + ci), lower: yScale(r[yField] - ci) };
    });
  const points = [...data.map((d) => [d.x, d.upper]), ...data.slice().reverse().map((d) => [d.x, d.lower])];
  return `<polygon points="${pathLine(points)}" fill="${color}" opacity="0.10" stroke="none"/>`;
}

function directLabel(x1, y1, x2, y2, label, method, opts = {}) {
  const color = (STYLE[method] ?? STYLE.parc).color;
  const { size = 15.5, anchor = "start" } = opts;
  return `${line(x1, y1, x2 - 8, y2, { color, width: 0.9, opacity: 0.65 })}${text(x2, y2, label, { size, anchor, color, weight: method === "parc" ? 700 : 500 })}`;
}

function chartFigure({ width = 700, height = 420, body }) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
<rect x="0" y="0" width="${width}" height="${height}" fill="white"/>
<g font-family="Times New Roman, Times, serif">${body}</g>
</svg>`;
}

function lineChart(rows, config) {
  const { methods, xField, yField, yFloor, xMin, xMax, yMin, yMax, yLog = false, xTicks, yTicks, xLabel, yLabel, xFormat, yFormat, labelPositions, ci = [] } = config;
  const x0 = 72;
  const y0 = 30;
  const w = 458;
  const h = 286;
  const xScale = linear(xMin, xMax, x0, x0 + w);
  const yScale = yLog ? logScale(yMin, yMax, y0 + h, y0) : linear(yMin, yMax, y0 + h, y0);
  let out = axes({ x0, y0, w, h, xTicks, yTicks, xScale, yScale, xFormat, yFormat, xLabel, yLabel });
  for (const item of ci) {
    out += ciBand(rows, item.method, xField, yField, item.stdField, xScale, yScale, item.n, STYLE[item.method].color);
  }
  for (const method of methods) {
    const data = seriesByMethod(rows, method, xField, yField, yFloor);
    const points = data.map(([x, y]) => [xScale(x), yScale(y)]);
    const s = STYLE[method];
    out += polyline(points, { color: s.color, width: s.width, dash: s.dash });
    for (const [x, y] of points) out += marker(x, y, method, 0.92);
  }
  for (const spec of labelPositions) {
    const data = seriesByMethod(rows, spec.method, xField, yField, yFloor);
    const last = data[data.length - 1];
    if (last) out += directLabel(xScale(last[0]), yScale(last[1]), spec.x, spec.y, spec.label ?? LABELS[spec.method], spec.method);
  }
  if (config.note) out += text(x0 + 8, y0 + 8, config.note, { size: 14, anchor: "start", baseline: "hanging", color: "#555" });
  return chartFigure({ body: out });
}

function buildAccuracy(rows) {
  return lineChart(rows, {
    methods: ["no_repair", "majority", "provenance_unaware", "iterative_truth", "dependency_truth", "parc"],
    xField: "budget",
    yField: "integrated_quality_mean",
    xMin: 0,
    xMax: 0.385,
    yMin: 0.44,
    yMax: 1.015,
    xTicks: [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
    yTicks: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    xLabel: "Adversarial budget (%)",
    yLabel: "Integrated quality",
    xFormat: (x) => `${Math.round(100 * x)}`,
    yFormat: (y) => y.toFixed(1),
    ci: [
      { method: "dependency_truth", stdField: "integrated_quality_std", n: 5 },
      { method: "parc", stdField: "integrated_quality_std", n: 5 },
    ],
    labelPositions: [
      { method: "dependency_truth", x: 548, y: 39 },
      { method: "parc", x: 548, y: 70 },
      { method: "iterative_truth", x: 548, y: 116 },
      { method: "majority", x: 548, y: 148 },
      { method: "provenance_unaware", x: 548, y: 190 },
      { method: "no_repair", x: 548, y: 302 },
    ],
  });
}

function buildDistortion(rows) {
  const floor = 0.0025;
  return lineChart(rows, {
    methods: ["no_repair", "provenance_unaware", "iterative_truth", "dependency_truth", "parc"],
    xField: "budget",
    yField: "aggregate_distortion_mean",
    yFloor: floor,
    yLog: true,
    xMin: 0,
    xMax: 0.385,
    yMin: floor,
    yMax: 1.7,
    xTicks: [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
    yTicks: [0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
    xLabel: "Adversarial budget (%)",
    yLabel: "Aggregate distortion",
    xFormat: (x) => `${Math.round(100 * x)}`,
    yFormat: (y) => String(y),
    note: "log scale; lower is better",
    labelPositions: [
      { method: "no_repair", x: 548, y: 94 },
      { method: "provenance_unaware", x: 548, y: 156 },
      { method: "iterative_truth", x: 548, y: 214 },
      { method: "dependency_truth", x: 548, y: 273 },
      { method: "parc", x: 548, y: 303 },
    ],
  });
}

function buildTrust(rows) {
  const pctRows = rows.map((r) => ({ ...r, fnr_pct: 100 * r.false_negative_rate_mean }));
  return lineChart(pctRows, {
    methods: ["provenance_unaware", "dependency_truth", "parc"],
    xField: "trust_alpha",
    yField: "fnr_pct",
    xMin: 0,
    xMax: 1,
    yMin: 0,
    yMax: 4.7,
    xTicks: [0, 0.25, 0.5, 0.75, 1.0],
    yTicks: [0, 1, 2, 3, 4],
    xLabel: "Declared-trust manipulation",
    yLabel: "Dirty-claim FNR (pct. pts.)",
    xFormat: (x) => x.toFixed(x === 0 || x === 1 ? 0 : 2),
    yFormat: (y) => `${y}`,
    labelPositions: [
      { method: "provenance_unaware", x: 548, y: 65 },
      { method: "dependency_truth", x: 548, y: 277 },
      { method: "parc", x: 548, y: 301 },
    ],
  });
}

function panelAxes({ x0, y0, w, h, xTicks, yTicks, xScale, yScale, xFormat, yFormat, xLabel, yLabel }) {
  return axes({ x0, y0, w, h, xTicks, yTicks, xScale, yScale, xFormat, yFormat, xLabel, yLabel });
}

function buildRuntime(entityRows, sourceRows, largeRows) {
  const width = 760;
  const height = 390;
  let out = "";
  const left = { x0: 66, y0: 37, w: 290, h: 244 };
  const right = { x0: 456, y0: 37, w: 230, h: 244 };
  const eAll = [...entityRows, ...largeRows].filter((r) => ["majority", "dependency_truth", "parc"].includes(r.method));
  const lx = logScale(140, 3300, left.x0, left.x0 + left.w);
  const ly = logScale(0.075, 13.0, left.y0 + left.h, left.y0);
  out += panelAxes({
    ...left,
    xScale: lx,
    yScale: ly,
    xTicks: [150, 300, 600, 900, 3000],
    yTicks: [0.1, 0.3, 1, 3, 10],
    xFormat: (x) => `${x}`,
    yFormat: (y) => `${y}`,
    xLabel: "Entities",
    yLabel: "Runtime (s)",
  });
  for (const method of ["majority", "dependency_truth", "parc"]) {
    const data = eAll.filter((r) => r.method === method).sort((a, b) => a.n_entities - b.n_entities);
    const points = data.map((r) => [lx(r.n_entities), ly(r.runtime_sec_mean)]);
    const s = STYLE[method];
    out += polyline(points, { color: s.color, width: s.width, dash: s.dash });
    for (const [x, y] of points) out += marker(x, y, method, 0.85);
  }
  out += text(left.x0 + 6, left.y0 - 14, "(a) Entity scale", { size: 17, anchor: "start", color: "#222", weight: 600 });
  const legend = [
    ["parc", "PARC"],
    ["dependency_truth", "Dep.+truth"],
    ["majority", "Majority"],
  ];
  out += `<rect x="${left.x0 + 13}" y="${left.y0 + 14}" width="104" height="58" rx="3" fill="white" opacity="0.88" stroke="#d9dee4" stroke-width="0.6"/>`;
  for (const [i, [method, label]] of legend.entries()) {
    const y = left.y0 + 29 + i * 17;
    const s = STYLE[method];
    out += line(left.x0 + 22, y, left.x0 + 44, y, { color: s.color, width: s.width, dash: s.dash });
    out += marker(left.x0 + 33, y, method, 0.52);
    out += text(left.x0 + 51, y + 0.4, label, { size: 10.8, anchor: "start", color: s.color, weight: method === "parc" ? 700 : 500 });
  }

  const sx = linear(6, 36, right.x0, right.x0 + right.w);
  const sy = linear(0.9, 1.95, right.y0 + right.h, right.y0);
  out += panelAxes({
    ...right,
    xScale: sx,
    yScale: sy,
    xTicks: [6, 12, 18, 24, 30, 36],
    yTicks: [1.0, 1.25, 1.5, 1.75],
    xFormat: (x) => `${x}`,
    yFormat: (y) => y.toFixed(2).replace(/0$/, ""),
    xLabel: "Sources",
    yLabel: "Runtime (s)",
  });
  const spoints = sourceRows.filter((r) => r.method === "parc").sort((a, b) => a.n_sources - b.n_sources).map((r) => [sx(r.n_sources), sy(r.runtime_sec_mean)]);
  out += polyline(spoints, { color: STYLE.parc.color, width: STYLE.parc.width });
  for (const [x, y] of spoints) out += marker(x, y, "parc", 0.85);
  out += text(right.x0 + 6, right.y0 - 14, "(b) Source scale", { size: 17, anchor: "start", color: "#222", weight: 600 });
  out += text(right.x0 + right.w - 4, sy(1.828), "PARC", { size: 15.5, anchor: "end", color: STYLE.parc.color, weight: 700 });

  return chartFigure({ width, height, body: out });
}

function buildCorruptionModel() {
  const width = 1420;
  const height = 300;
  let out = "";
  out += `<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#1F2937"/></marker></defs>`;
  out += line(58, 42, 1362, 42, { color: "#E5E7EB", width: 0.75 });
  out += line(58, 258, 1362, 258, { color: "#E5E7EB", width: 0.75 });

  const table = (x, y, w, h, rows, opts = {}) => {
    let s = rect(x, y, w, h, { fill: opts.fill ?? "#FFFFFF", stroke: opts.stroke ?? "#475569", sw: opts.sw ?? 1.1, rx: 1 });
    for (let i = 1; i < rows.length; i++) s += line(x, y + (h / rows.length) * i, x + w, y + (h / rows.length) * i, { color: opts.grid ?? "#CBD5E1", width: 0.7 });
    rows.forEach((r, i) => {
      s += text(x + w / 2, y + (h / rows.length) * (i + 0.5), r, { size: opts.size ?? 14.5, color: opts.color ?? "#111827" });
    });
    return s;
  };

  out += table(64, 66, 238, 154, [
    "C'(e,a,s,p,t)",
    "value and key",
    "mapping and Sigma",
    "trust prior",
    "prov. group",
  ], { fill: "#FFFFFF", size: 15.8 });

  out += line(314, 143, 430, 143, { color: "#1F2937", width: 1.25, markerEnd: "arrow" });

  out += rect(448, 52, 422, 184, { fill: "#FFFFFF", stroke: "#0F766E", sw: 1.1, rx: 1 });
  out += richText(659, 84, ["R", { text: "PARC", size: 11.8, shift: "sub" }, "(C')"], { size: 18.2, color: "#111827", italic: true });
  out += text(659, 122, "Q = dependence(C', prov)", { size: 15.7, color: "#111827" });
  out += richText(659, 153, ["supp(e,a,v) = sum", { text: "q", size: 10.5, shift: "sub" }, " max", { text: "s", size: 10.5, shift: "sub" }, " clip(t", { text: "s", size: 10.5, shift: "sub" }, ")"], { size: 15.7, color: "#111827" });
  out += text(659, 184, "choose v with map and FD penalties", { size: 15.7, color: "#111827" });
  out += text(659, 215, "emit margins and replay state", { size: 15.9, color: "#111827" });

  out += line(882, 143, 954, 143, { color: "#1F2937", width: 1.25, markerEnd: "arrow" });

  out += table(972, 66, 150, 154, [
    "J",
    "entity key",
    "attribute",
    "value",
    "SQL interval",
  ], { fill: "#FFFFFF", size: 15.1 });
  out += table(1144, 54, 210, 178, [
    "cert(e,a)",
    "candidate values",
    "independent Q",
    "weights and maps",
    "FD witnesses",
    "margin / warning",
  ], { fill: "#FFFFFF", stroke: "#0F766E", size: 14.7 });
  return chartFigure({ width, height, body: out });
}

function findChrome() {
  const candidates = [
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  ];
  const env = process.env.CHROME_PATH ? [process.env.CHROME_PATH, ...candidates] : candidates;
  const found = env.find((p) => fssync.existsSync(p));
  if (!found) throw new Error("Chrome or Edge executable not found; set CHROME_PATH.");
  return found;
}

async function render(name, svg, inchesW, inchesH) {
  await fs.mkdir(FIG_DIR, { recursive: true });
  const pxW = Math.round(inchesW * DPI);
  const pxH = Math.round(inchesH * DPI);
  const svgPath = path.join(FIG_DIR, `${name}.svg`);
  const htmlPath = path.join(FIG_DIR, `${name}.html`);
  const pdfPath = path.join(FIG_DIR, `${name}.pdf`);
  const pngPath = path.join(FIG_DIR, `${name}.png`);
  await fs.writeFile(svgPath, svg, "utf8");
  const html = `<!doctype html><html><head><meta charset="utf-8"><style>
@page { size: ${inchesW}in ${inchesH}in; margin: 0; }
html, body { margin: 0; padding: 0; width: ${pxW}px; height: ${pxH}px; overflow: hidden; background: white; }
svg { display: block; width: ${pxW}px; height: ${pxH}px; }
* { box-sizing: border-box; }
</style></head><body>${svg}</body></html>`;
  await fs.writeFile(htmlPath, html, "utf8");

  const chrome = findChrome();
  const profile = path.join(os.tmpdir(), `parc-figure-chrome-${name}`);
  await fs.rm(profile, { recursive: true, force: true });
  const fileUrl = pathToFileURL(htmlPath).href;
  const common = ["--headless=new", "--disable-gpu", "--no-sandbox", "--allow-file-access-from-files", `--user-data-dir=${profile}`];
  let result = spawnSync(chrome, [...common, `--print-to-pdf=${pdfPath}`, "--print-to-pdf-no-header", fileUrl], { encoding: "utf8" });
  if (result.status !== 0) {
    throw new Error(`PDF render failed for ${name}:\n${result.stderr || result.stdout}`);
  }
  result = spawnSync(chrome, [...common, "--force-device-scale-factor=1", `--window-size=${pxW},${pxH}`, `--screenshot=${pngPath}`, fileUrl], { encoding: "utf8" });
  if (result.status !== 0) {
    throw new Error(`PNG render failed for ${name}:\n${result.stderr || result.stdout}`);
  }
  await fs.rm(profile, { recursive: true, force: true });
}

async function main() {
  const benchmark = await readCsv("benchmark_summary.csv");
  const trust = await readCsv("trust_sensitivity_summary.csv");
  const entityScale = await readCsv("scalability_summary.csv");
  const sourceScale = await readCsv("source_scale_parc_summary.csv");
  const largeScale = await readCsv("large_scale_3000_summary.csv");
  const jobs = [
    ["corruption_model", buildCorruptionModel(), 7.16, 1.52],
    ["accuracy_budget", buildAccuracy(benchmark), 3.48, 2.10],
    ["aggregate_distortion_budget", buildDistortion(benchmark), 3.48, 2.10],
    ["trust_sensitivity", buildTrust(trust), 3.48, 2.03],
    ["runtime_scalability", buildRuntime(entityScale, sourceScale, largeScale), 3.48, 1.96],
  ];
  for (const [name, svg, w, h] of jobs) {
    await render(name, svg, w, h);
    console.log(`rendered figures/${name}.svg, .png, and .pdf preview`);
  }
}

main().catch((err) => {
  console.error(err.stack || err.message);
  process.exit(1);
});
