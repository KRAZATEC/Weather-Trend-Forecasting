const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const FIG = path.join(__dirname, "..", "outputs", "figures");
const REP = path.join(__dirname, "..", "outputs", "reports");

// ---- Ocean Gradient palette ----
const DEEP = "065A82";
const TEAL = "1C7293";
const MIDNIGHT = "21295C";
const WHITE = "FFFFFF";
const OFFWHITE = "F5F8FA";
const INK = "1B2733";
const MUTED = "5B6B79";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.author = "Weather Trend Forecasting Project";
pres.title = "Weather Trend Forecasting";

const W = 13.33, H = 7.5;

function darkSlide() {
  const s = pres.addSlide();
  s.background = { color: MIDNIGHT };
  return s;
}
function lightSlide() {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  return s;
}

function sectionHeader(slide, kicker, title, dark = false) {
  slide.addText(kicker.toUpperCase(), {
    x: 0.6, y: 0.45, w: 10, h: 0.35, fontFace: "Calibri", fontSize: 13, bold: true,
    color: dark ? "8FD3E8" : TEAL, charSpacing: 2,
  });
  slide.addText(title, {
    x: 0.6, y: 0.75, w: 11.5, h: 0.9, fontFace: "Cambria", fontSize: 30, bold: true,
    color: dark ? WHITE : INK,
  });
}
function pageNum(slide, n) {
  slide.addText(String(n), {
    x: W - 0.9, y: H - 0.55, w: 0.5, h: 0.35, fontFace: "Calibri", fontSize: 11, color: MUTED, align: "right",
  });
}
function img(name) {
  return path.join(FIG, name);
}
function fitImage(slide, filename, boxX, boxY, boxW, boxH) {
  const { execSync } = require("child_process");
  const dims = JSON.parse(execSync(
    `python3 -c "from PIL import Image; im=Image.open('${filename}'); print('{\\"w\\":%d,\\"h\\":%d}' % im.size)"`
  ).toString());
  const ratio = dims.w / dims.h;
  let w = boxW, h = boxW / ratio;
  if (h > boxH) { h = boxH; w = boxH * ratio; }
  const x = boxX + (boxW - w) / 2;
  const y = boxY + (boxH - h) / 2;
  slide.addImage({ path: filename, x, y, w, h });
}

const modelComparison = fs.readFileSync(path.join(REP, "model_comparison.csv"), "utf8")
  .trim().split("\n").map(l => l.split(","));

// =============================================================
// Slide 1: Title
// =============================================================
{
  const s = darkSlide();
  s.addShape(pres.shapes.OVAL, { x: 9.2, y: -2.2, w: 7, h: 7, fill: { color: DEEP, transparency: 55 }, line: { type: "none" } });
  s.addShape(pres.shapes.OVAL, { x: -2.5, y: 4.5, w: 6, h: 6, fill: { color: TEAL, transparency: 65 }, line: { type: "none" } });
  s.addText("WEATHER TREND FORECASTING", {
    x: 0.9, y: 2.5, w: 11.5, h: 1.3, fontFace: "Cambria", fontSize: 46, bold: true, color: WHITE,
  });
  s.addText("Machine Learning & Advanced Analytics on Global Weather Data", {
    x: 0.9, y: 3.65, w: 10.5, h: 0.6, fontFace: "Calibri", fontSize: 20, color: "AFC9DA",
  });
  s.addText("Data Cleaning  \u2022  40+ Visualizations  \u2022  7 ML Models  \u2022  Forecasting  \u2022  Dashboard", {
    x: 0.9, y: 4.35, w: 10.5, h: 0.5, fontFace: "Calibri", fontSize: 14, italic: true, color: "8FD3E8",
  });
  s.addText("Technical Presentation \u2022 July 2026", {
    x: 0.9, y: 6.55, w: 6, h: 0.4, fontFace: "Calibri", fontSize: 12, color: "7C93A6",
  });
}

// =============================================================
// Slide 2: Problem Statement
// =============================================================
{
  const s = lightSlide();
  sectionHeader(s, "01 · Context", "Problem Statement");
  s.addText(
    "Weather variability affects agriculture, energy demand, disaster response, and public health. Reliable, data-driven forecasting and pattern detection at a global scale remain challenging due to the volume, noise, and geographic diversity of weather data.",
    { x: 0.6, y: 2.0, w: 6.4, h: 2.2, fontFace: "Calibri", fontSize: 16, color: INK, valign: "top" }
  );
  const cards = [
    ["Scale", "60 countries, multi-variable daily observations"],
    ["Noise", "Missing values, duplicates, and sensor outliers"],
    ["Uncertainty", "Short-term forecasting under natural variability"],
  ];
  cards.forEach((c, i) => {
    const x = 0.6 + i * 2.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: 4.5, w: 1.95, h: 1.9, rectRadius: 0.08, fill: { color: OFFWHITE },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 3, angle: 90, opacity: 0.10 },
    });
    s.addText(c[0], { x: x + 0.15, y: 4.65, w: 1.65, h: 0.4, fontFace: "Cambria", fontSize: 16, bold: true, color: DEEP });
    s.addText(c[1], { x: x + 0.15, y: 5.05, w: 1.65, h: 1.2, fontFace: "Calibri", fontSize: 11, color: MUTED });
  });
  fitImage(s, img("world_temperature_map_static.png"), 7.3, 1.9, 5.4, 4.8);
  pageNum(s, 2);
}

// =============================================================
// Slide 3: Objectives
// =============================================================
{
  const s = lightSlide();
  sectionHeader(s, "01 · Context", "Project Objectives");
  const objs = [
    "Clean and validate a multi-country weather dataset",
    "Explore patterns through 40+ visualizations",
    "Engineer domain-relevant predictive features",
    "Train & compare 7 regression models + ensemble",
    "Forecast short-term trends (ARIMA, SARIMA, Prophet)",
    "Apply clustering, anomaly detection & SHAP explainability",
    "Map weather geospatially",
    "Ship an interactive Streamlit dashboard",
  ];
  const half = Math.ceil(objs.length / 2);
  [objs.slice(0, half), objs.slice(half)].forEach((col, ci) => {
    s.addText(col.map((t, i) => ({
      text: t, options: { bullet: { code: "25CF" }, breakLine: i < col.length - 1, color: INK, fontSize: 16 },
    })), { x: 0.7 + ci * 6.1, y: 2.1, w: 5.7, h: 4.6, fontFace: "Calibri", valign: "top", paraSpaceAfter: 14 });
  });
  pageNum(s, 3);
}

// =============================================================
// Slide 4: Dataset Overview
// =============================================================
{
  const s = lightSlide();
  sectionHeader(s, "02 · Data", "Dataset Overview");
  s.addText("Kaggle Global Weather Repository (schema-matched sample used in this build \u2014 see notes)", {
    x: 0.6, y: 1.75, w: 11, h: 0.4, fontFace: "Calibri", fontSize: 13, italic: true, color: MUTED,
  });
  const stats = [["12,040", "Rows"], ["41", "Raw Columns"], ["60", "Countries"], ["~200", "Days Observed"]];
  stats.forEach((st, i) => {
    const x = 0.6 + i * 3.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.4, w: 2.8, h: 1.6, rectRadius: 0.08, fill: { color: DEEP } });
    s.addText(st[0], { x, y: 2.55, w: 2.8, h: 0.75, align: "center", fontFace: "Cambria", fontSize: 32, bold: true, color: WHITE });
    s.addText(st[1], { x, y: 3.3, w: 2.8, h: 0.5, align: "center", fontFace: "Calibri", fontSize: 13, color: "BFE0EC" });
  });
  s.addText("Column Groups", { x: 0.6, y: 4.35, w: 5, h: 0.4, fontFace: "Cambria", fontSize: 16, bold: true, color: INK });
  const groups = ["Temperature & Comfort", "Atmosphere (pressure, humidity, cloud, UV)", "Wind (speed, gust, direction)",
                  "Precipitation & Conditions", "Air Quality Indices", "Astronomy (sunrise/sunset, moon phase)"];
  s.addText(groups.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, breakLine: i < groups.length - 1 } })), {
    x: 0.6, y: 4.8, w: 6, h: 2.2, fontFace: "Calibri", fontSize: 13.5, color: INK, valign: "top",
  });
  fitImage(s, img("dist_temperature_celsius.png"), 7.0, 4.3, 5.7, 2.8);
  pageNum(s, 4);
}

// =============================================================
// Slide 5: Data Cleaning
// =============================================================
{
  const s = lightSlide();
  sectionHeader(s, "02 · Data", "Data Cleaning Pipeline");
  const steps = [
    ["1", "Remove Duplicates", "40 injected duplicates detected & dropped"],
    ["2", "Parse Datetime", "last_updated parsed & validated"],
    ["3", "Impute Missing", "Median (numeric) / mode (categorical), ~2% missingness"],
    ["4", "Treat Outliers", "IQR (1.5x) & Z-score (|z|>3) detection, clipped to bounds"],
  ];
  steps.forEach((st, i) => {
    const x = 0.6 + i * 3.05;
    s.addShape(pres.shapes.OVAL, { x, y: 2.1, w: 0.55, h: 0.55, fill: { color: TEAL } });
    s.addText(st[0], { x, y: 2.1, w: 0.55, h: 0.55, align: "center", valign: "middle", fontFace: "Cambria", fontSize: 18, bold: true, color: WHITE });
    s.addText(st[1], { x, y: 2.8, w: 2.8, h: 0.5, fontFace: "Cambria", fontSize: 14.5, bold: true, color: INK });
    s.addText(st[2], { x, y: 3.3, w: 2.8, h: 1.3, fontFace: "Calibri", fontSize: 11.5, color: MUTED, valign: "top" });
    if (i < 3) s.addShape(pres.shapes.LINE, { x: x + 2.9, y: 2.37, w: 0.2, h: 0, line: { color: "C7D6DE", width: 1.5 } });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.0, w: 12.1, h: 1.6, rectRadius: 0.08, fill: { color: OFFWHITE } });
  s.addText("Validation checks confirm humidity in [0,100], latitude in [-90,90], longitude in [-180,180], and zero remaining duplicate rows after cleaning.", {
    x: 0.9, y: 5.15, w: 11.5, h: 1.3, fontFace: "Calibri", fontSize: 14, italic: true, color: INK, valign: "middle",
  });
  pageNum(s, 5);
}

// =============================================================
// Slide 6: EDA Highlights
// =============================================================
{
  const s = lightSlide();
  sectionHeader(s, "03 · Exploration", "Exploratory Data Analysis");
  s.addText("40+ charts generated across distributions, relationships, rankings, and time trends", {
    x: 0.6, y: 1.75, w: 11, h: 0.4, fontFace: "Calibri", fontSize: 13, italic: true, color: MUTED,
  });
  fitImage(s, img("correlation_heatmap.png"), 0.6, 2.2, 5.6, 4.7);
  fitImage(s, img("ranking_hottest_countries.png"), 6.6, 2.2, 6.2, 4.7);
  pageNum(s, 6);
}

// =============================================================
// Slide 7: EDA Trends
// =============================================================
{
  const s = lightSlide();
  sectionHeader(s, "03 · Exploration", "Temporal & Relational Patterns");
  fitImage(s, img("trend_temperature_celsius_D.png"), 0.6, 2.1, 12.1, 2.6);
  fitImage(s, img("scatter_humidity_vs_temperature_celsius.png"), 0.6, 4.85, 5.9, 2.2);
  const notes = [
    "Humidity & temperature: r \u2248 -0.47 (moderate negative)",
    "UV index & temperature: r \u2248 0.83 (strong positive)",
    "Wind speed & gust: r \u2248 0.96 (near-perfect, physically expected)",
  ];
  s.addText(notes.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, breakLine: i < notes.length - 1 } })), {
    x: 6.9, y: 4.95, w: 5.9, h: 2.1, fontFace: "Calibri", fontSize: 14, color: INK, valign: "top", paraSpaceAfter: 10,
  });
  pageNum(s, 7);
}

// =============================================================
// Slide 8: Feature Engineering
// =============================================================
{
  const s = lightSlide();
  sectionHeader(s, "04 · Features", "Feature Engineering");
  const groups = [
    ["Calendar", "year, month, week, quarter, is_weekend, hemisphere-aware season"],
    ["Comfort Indices", "heat index, wind chill, temp-feels difference, humidity index"],
    ["Lag & Rolling", "1/2/3/7-day lags; 3/7/14-day rolling mean & std, per location"],
    ["Interactions", "humidity\u00d7temp, wind\u00d7pressure, polynomial temperature"],
  ];
  groups.forEach((g, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.6 + col * 6.15, y = 2.1 + row * 2.35;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 5.85, h: 2.05, rectRadius: 0.08, fill: { color: OFFWHITE },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 3, angle: 90, opacity: 0.08 } });
    s.addText(g[0], { x: x + 0.3, y: y + 0.2, w: 5.25, h: 0.5, fontFace: "Cambria", fontSize: 17, bold: true, color: DEEP });
    s.addText(g[1], { x: x + 0.3, y: y + 0.75, w: 5.25, h: 1.15, fontFace: "Calibri", fontSize: 13, color: MUTED, valign: "top" });
  });
  pageNum(s, 8);
}

// =============================================================
// Slide 9: ML Models Overview
// =============================================================
{
  const s = lightSlide();
  sectionHeader(s, "05 · Modeling", "Machine Learning Models");
  const models = ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "SVR", "XGBoost", "LightGBM", "Ensemble (avg)"];
  models.forEach((m, i) => {
    const col = i % 4, row = Math.floor(i / 4);
    const x = 0.6 + col * 3.05, y = 2.3 + row * 1.7;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 2.8, h: 1.4, rectRadius: 0.1, fill: { color: row === 0 ? TEAL : DEEP } });
    s.addText(m, { x, y, w: 2.8, h: 1.4, align: "center", valign: "middle", fontFace: "Calibri", fontSize: 14, bold: true, color: WHITE });
  });
  s.addText("Target: temperature_celsius  |  80/20 train-test split  |  Leakage-aware feature matrix (see notes)", {
    x: 0.6, y: 5.9, w: 11.5, h: 0.5, fontFace: "Calibri", fontSize: 13, italic: true, color: MUTED,
  });
  pageNum(s, 9);
}

// =============================================================
// Slide 10: Model Comparison
// =============================================================
{
  const s = lightSlide();
  sectionHeader(s, "06 · Results", "Model Comparison");
  const headerRow = modelComparison[0].map(h => ({ text: h, options: { fill: { color: DEEP }, color: WHITE, bold: true, fontSize: 12 } }));
  const bodyRows = modelComparison.slice(1).map((r, ri) => r.map((v, ci) => ({
    text: ci >= 2 ? Number(v).toFixed(3) : v,
    options: { fontSize: 12, fill: { color: ri % 2 === 0 ? WHITE : OFFWHITE }, bold: ri === 0 && ci === 1 },
  })));
  s.addTable([headerRow, ...bodyRows], {
    x: 0.6, y: 2.0, w: 7.0, colW: [0.7, 2.1, 1.05, 1.05, 1.05, 1.05, 0.9],
    border: { pt: 0.5, color: "DDE3E7" }, autoPage: false,
  });
  fitImage(s, img("model_comparison.png"), 7.9, 2.0, 4.9, 4.9);
  pageNum(s, 10);
}

// =============================================================
// Slide 11: Best Model Deep Dive
// =============================================================
{
  const s = lightSlide();
  sectionHeader(s, "06 · Results", "Best Model: Gradient Boosting");
  fitImage(s, img("pred_vs_actual_GradientBoosting.png"), 0.6, 2.0, 4.0, 5.0);
  fitImage(s, img("residuals_GradientBoosting.png"), 4.85, 2.0, 4.0, 5.0);
  fitImage(s, img("shap_summary.png"), 9.0, 1.9, 3.9, 5.1);
  pageNum(s, 11);
}

// =============================================================
// Slide 12: Time-Series Forecasting
// =============================================================
{
  const s = lightSlide();
  sectionHeader(s, "07 · Forecasting", "Time-Series Forecasting");
  const methods = [
    ["ARIMA(2,1,2)", "Classical autoregressive-integrated-moving-average baseline"],
    ["SARIMA(1,1,1)(1,1,1,7)", "Adds weekly seasonality on top of ARIMA"],
    ["Prophet / Fallback", "Additive trend + seasonality; statsmodels fallback keeps the pipeline dependency-light"],
  ];
  methods.forEach((m, i) => {
    const y = 2.15 + i * 1.55;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 11.9, h: 1.3, rectRadius: 0.08, fill: { color: OFFWHITE } });
    s.addText(m[0], { x: 0.9, y: y + 0.15, w: 3.6, h: 1.0, fontFace: "Cambria", fontSize: 16, bold: true, color: DEEP, valign: "middle" });
    s.addText(m[1], { x: 4.6, y: y + 0.15, w: 7.6, h: 1.0, fontFace: "Calibri", fontSize: 13.5, color: INK, valign: "middle" });
  });
  pageNum(s, 12);
}

// =============================================================
// Slide 13: Advanced Analytics
// =============================================================
{
  const s = lightSlide();
  sectionHeader(s, "08 · Advanced Analytics", "Unsupervised Learning & Explainability");
  const items = [
    "Isolation Forest & LOF flag anomalous readings (2% contamination)",
    "KMeans (k=4) groups observations into weather regimes",
    "DBSCAN finds density-based clusters without pre-set k",
    "PCA reduces 4 core variables to 2 components",
    "Permutation importance + SHAP explain the best model",
  ];
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: { code: "25CF" }, breakLine: i < items.length - 1 } })), {
    x: 0.6, y: 2.1, w: 6.3, h: 4.6, fontFace: "Calibri", fontSize: 15.5, color: INK, valign: "top", paraSpaceAfter: 16,
  });
  fitImage(s, img("feature_importance_RandomForest.png"), 7.1, 2.0, 5.6, 4.9);
  pageNum(s, 13);
}

// =============================================================
// Slide 14: Climate Trend & Heatwaves
// =============================================================
{
  const s = lightSlide();
  sectionHeader(s, "08 · Advanced Analytics", "Climate Trend & Heatwave Detection");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 2.1, w: 5.6, h: 2.1, rectRadius: 0.08, fill: { color: DEEP } });
  s.addText("+20.4\u00b0C / yr", { x: 0.6, y: 2.3, w: 5.6, h: 0.8, align: "center", fontFace: "Cambria", fontSize: 34, bold: true, color: WHITE });
  s.addText("Annualized linear trend over the sample window (reflects seasonal transition within the observed period, not a multi-year climate signal)", {
    x: 0.9, y: 3.05, w: 5.0, h: 1.0, align: "center", fontFace: "Calibri", fontSize: 11.5, italic: true, color: "BFE0EC",
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 4.4, w: 5.6, h: 2.1, rectRadius: 0.08, fill: { color: TEAL } });
  s.addText("20 Heatwave Events", { x: 0.6, y: 4.6, w: 5.6, h: 0.8, align: "center", fontFace: "Cambria", fontSize: 30, bold: true, color: WHITE });
  s.addText("Detected across locations: \u22653 consecutive days above each location's 95th-percentile temperature", {
    x: 0.9, y: 5.35, w: 5.0, h: 1.0, align: "center", fontFace: "Calibri", fontSize: 11.5, italic: true, color: "E4F3F7",
  });
  fitImage(s, img("trend_temp_monthly.png"), 6.5, 2.1, 6.2, 4.4);
  pageNum(s, 14);
}

// =============================================================
// Slide 15: Geospatial Analysis
// =============================================================
{
  const s = darkSlide();
  sectionHeader(s, "09 · Geospatial", "Global Weather Maps", true);
  fitImage(s, img("world_temperature_map_static.png"), 0.6, 2.0, 7.2, 5.1);
  const maps = ["Temperature", "Humidity", "Rainfall", "Wind Speed", "Air Quality (PM2.5)", "Density Heatmap"];
  maps.forEach((m, i) => {
    const x = 8.1, y = 2.2 + i * 0.78;
    s.addShape(pres.shapes.OVAL, { x, y, w: 0.35, h: 0.35, fill: { color: TEAL } });
    s.addText(m, { x: x + 0.55, y: y - 0.08, w: 4.3, h: 0.5, fontFace: "Calibri", fontSize: 15, color: WHITE, valign: "middle" });
  });
  s.addText("6 interactive Folium maps \u2014 outputs/figures/*.html", {
    x: 8.1, y: 6.8, w: 4.5, h: 0.4, fontFace: "Calibri", fontSize: 11, italic: true, color: "8FA9BC",
  });
  pageNum(s, 15);
}

// =============================================================
// Slide 16: Dashboard
// =============================================================
{
  const s = lightSlide();
  sectionHeader(s, "10 · Product", "Interactive Dashboard");
  const pages = ["Home", "Dataset", "EDA", "Climate Trends", "Forecast", "Model Comparison", "Maps", "About"];
  pages.forEach((pg, i) => {
    const col = i % 4, row = Math.floor(i / 4);
    const x = 0.6 + col * 3.05, y = 2.1 + row * 1.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 2.8, h: 0.9, rectRadius: 0.08, fill: { color: OFFWHITE } });
    s.addText(pg, { x, y, w: 2.8, h: 0.9, align: "center", valign: "middle", fontFace: "Calibri", fontSize: 14, bold: true, color: DEEP });
  });
  const features = ["Country / date-range filters", "Dark-mode toggle", "CSV download", "Manual prediction input against trained models"];
  s.addText(features.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, breakLine: i < features.length - 1 } })), {
    x: 0.6, y: 4.6, w: 11.5, h: 2.0, fontFace: "Calibri", fontSize: 15, color: INK, valign: "top", paraSpaceAfter: 8,
  });
  s.addText("streamlit run dashboard/app.py", {
    x: 0.6, y: 6.6, w: 6, h: 0.4, fontFace: "Courier New", fontSize: 13, color: TEAL,
  });
  pageNum(s, 16);
}

// =============================================================
// Slide 17: Results Summary
// =============================================================
{
  const s = lightSlide();
  sectionHeader(s, "11 · Summary", "Results at a Glance");
  const stats = [["0.933", "Best R\u00b2"], ["2.83\u00b0C", "Best RMSE"], ["7 + 1", "Models Trained"], ["44+", "EDA Charts"]];
  stats.forEach((st, i) => {
    const x = 0.6 + i * 3.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.2, w: 2.8, h: 1.8, rectRadius: 0.1, fill: { color: i % 2 === 0 ? DEEP : TEAL } });
    s.addText(st[0], { x, y: 2.35, w: 2.8, h: 0.9, align: "center", fontFace: "Cambria", fontSize: 30, bold: true, color: WHITE });
    s.addText(st[1], { x, y: 3.2, w: 2.8, h: 0.6, align: "center", fontFace: "Calibri", fontSize: 13, color: "E4F3F7" });
  });
  const insights = [
    "Gradient Boosting, LightGBM, and XGBoost cluster tightly at the top \u2014 model choice can favor operational concerns",
    "Rolling temperature statistics dominate feature importance (strong day-to-day autocorrelation)",
    "SVR underperforms without heavy tuning, showing preprocessing sensitivity across algorithm families",
  ];
  s.addText(insights.map((t, i) => ({ text: t, options: { bullet: { code: "25CF" }, breakLine: i < insights.length - 1 } })), {
    x: 0.6, y: 4.5, w: 12.0, h: 2.4, fontFace: "Calibri", fontSize: 15, color: INK, valign: "top", paraSpaceAfter: 14,
  });
  pageNum(s, 17);
}

// =============================================================
// Slide 18: Business Recommendations
// =============================================================
{
  const s = lightSlide();
  sectionHeader(s, "12 · Impact", "Business Recommendations");
  const recs = [
    ["Short-Horizon Forecasting", "Use SARIMA when only historical temperature is available; use gradient-boosted ML when auxiliary weather variables are already collected"],
    ["Early Warning Systems", "Heatwave-detection logic can plug into agriculture & public-health alerting with minimal added infrastructure"],
    ["Model Selection", "With top models within ~0.1\u00b0C RMSE of each other, prioritize inference latency, interpretability, and maintainability over marginal accuracy"],
  ];
  recs.forEach((r, i) => {
    const y = 2.1 + i * 1.65;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 12.1, h: 1.4, rectRadius: 0.08, fill: { color: OFFWHITE },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 3, angle: 90, opacity: 0.08 } });
    s.addText(r[0], { x: 0.9, y: y + 0.12, w: 3.4, h: 1.15, fontFace: "Cambria", fontSize: 15.5, bold: true, color: DEEP, valign: "middle" });
    s.addText(r[1], { x: 4.5, y: y + 0.12, w: 7.9, h: 1.15, fontFace: "Calibri", fontSize: 13, color: INK, valign: "middle" });
  });
  pageNum(s, 18);
}

// =============================================================
// Slide 19: Future Scope
// =============================================================
{
  const s = lightSlide();
  sectionHeader(s, "13 · Roadmap", "Future Scope");
  const items = [
    "Validate every finding against the real Kaggle dataset",
    "Add an LSTM / Temporal Fusion Transformer forecaster",
    "Deploy the best model behind a FastAPI microservice",
    "Add Docker packaging & GitHub Actions CI/CD",
    "Incorporate exogenous climate indices (ENSO, NAO)",
  ];
  items.forEach((t, i) => {
    const y = 2.15 + i * 0.92;
    s.addShape(pres.shapes.OVAL, { x: 0.6, y, w: 0.45, h: 0.45, fill: { color: TEAL } });
    s.addText(String(i + 1), { x: 0.6, y, w: 0.45, h: 0.45, align: "center", valign: "middle", fontFace: "Cambria", fontSize: 15, bold: true, color: WHITE });
    s.addText(t, { x: 1.25, y: y - 0.02, w: 10.8, h: 0.5, fontFace: "Calibri", fontSize: 16, color: INK, valign: "middle" });
  });
  pageNum(s, 19);
}

// =============================================================
// Slide 20: Conclusion / Questions
// =============================================================
{
  const s = darkSlide();
  s.addShape(pres.shapes.OVAL, { x: -2, y: -2.5, w: 6.5, h: 6.5, fill: { color: DEEP, transparency: 55 }, line: { type: "none" } });
  s.addShape(pres.shapes.OVAL, { x: 9.5, y: 4, w: 6, h: 6, fill: { color: TEAL, transparency: 62 }, line: { type: "none" } });
  s.addText("Conclusion & Questions", {
    x: 0.9, y: 2.7, w: 11.5, h: 1.0, fontFace: "Cambria", fontSize: 38, bold: true, color: WHITE,
  });
  s.addText("A complete, reproducible pipeline \u2014 from raw weather data to an interactive forecasting dashboard \u2014 with disclosed assumptions and honest metrics.", {
    x: 0.9, y: 3.75, w: 9.5, h: 1.0, fontFace: "Calibri", fontSize: 17, color: "C7DCE6",
  });
  s.addText("github.com/<your-username>/Weather-Trend-Forecasting", {
    x: 0.9, y: 6.6, w: 8, h: 0.4, fontFace: "Courier New", fontSize: 13, color: "8FD3E8",
  });
}

pres.writeFile({ fileName: path.join(__dirname, "Weather_Trend_Forecasting.pptx") }).then(() => {
  console.log("Presentation written.");
});
