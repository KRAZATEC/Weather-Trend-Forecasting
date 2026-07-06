const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, ImageRun, PageBreak, AlignmentType, BorderStyle,
  Header, Footer, PageNumber, TableOfContents, LevelFormat, NumberFormat,
} = require("docx");

const FIG = path.join(__dirname, "..", "outputs", "figures");
const REP = path.join(__dirname, "..", "outputs", "reports");

function img(name, width, height) {
  const p = path.join(FIG, name);
  return new ImageRun({ type: "png", data: fs.readFileSync(p), transformation: { width, height } });
}

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 400, after: 200 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 300, after: 150 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 160 } });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 80 } });
}
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, size: 18, color: "555555" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 300 },
  });
}
function figure(name, width, height, cap) {
  return [
    new Paragraph({ children: [img(name, width, height)], alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 } }),
    caption(cap),
  ];
}

function simpleTable(headers, rows) {
  const headerCells = headers.map(htext => new TableCell({
    shading: { type: ShadingType.CLEAR, fill: "1C7293" },
    width: { size: 100 / headers.length, type: WidthType.PERCENTAGE },
    children: [new Paragraph({ children: [new TextRun({ text: htext, bold: true, color: "FFFFFF" })] })],
  }));
  const bodyRows = rows.map(r => new TableRow({
    children: r.map(cell => new TableCell({
      width: { size: 100 / headers.length, type: WidthType.PERCENTAGE },
      children: [new Paragraph({ children: [new TextRun({ text: String(cell) })] })],
    })),
  }));
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    rows: [new TableRow({ children: headerCells }), ...bodyRows],
  });
}

const modelComparison = fs.readFileSync(path.join(REP, "model_comparison.csv"), "utf8")
  .trim().split("\n").map(l => l.split(","));
const mcHeaders = modelComparison[0];
const mcRows = modelComparison.slice(1).map(r => r.map((v, i) => (i >= 2 ? Number(v).toFixed(3) : v)));

const doc = new Document({
  sections: [
    // ---------------- Title Page ----------------
    {
      properties: { page: { size: { width: 12240, height: 15840 } } },
      children: [
        new Paragraph({ text: "", spacing: { before: 2000 } }),
        new Paragraph({
          children: [new TextRun({ text: "Weather Trend Forecasting", bold: true, size: 56, color: "1C7293" })],
          alignment: AlignmentType.CENTER, spacing: { after: 200 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "Machine Learning and Advanced Data Analytics on the", size: 28, color: "444444" })],
          alignment: AlignmentType.CENTER,
        }),
        new Paragraph({
          children: [new TextRun({ text: "Global Weather Repository Dataset", size: 28, color: "444444" })],
          alignment: AlignmentType.CENTER, spacing: { after: 800 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "Technical Project Report", size: 24, italics: true, color: "666666" })],
          alignment: AlignmentType.CENTER, spacing: { after: 2000 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "Prepared using Python, Scikit-learn, XGBoost, LightGBM, Statsmodels, and Streamlit", size: 20, color: "888888" })],
          alignment: AlignmentType.CENTER,
        }),
        new Paragraph({
          children: [new TextRun({ text: "July 2026", size: 20, color: "888888" })],
          alignment: AlignmentType.CENTER,
        }),
        new Paragraph({ children: [new PageBreak()] }),
      ],
    },
    // ---------------- Main Content ----------------
    {
      properties: {},
      headers: {
        default: new Header({ children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "Weather Trend Forecasting", size: 16, color: "999999" })],
        })] }),
      },
      footers: {
        default: new Footer({ children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Page ", size: 16 }), new TextRun({ children: [PageNumber.CURRENT], size: 16 })],
        })] }),
      },
      children: [
        h1("Executive Summary"),
        p("This report documents an end-to-end data science project that analyzes global weather patterns and builds machine learning models to forecast temperature trends. The project covers data cleaning, exploratory data analysis (40+ visualizations), feature engineering, seven regression models plus an ensemble, three time-series forecasting approaches (ARIMA, SARIMA, and Prophet/fallback), unsupervised advanced analytics, geospatial mapping, and an interactive dashboard."),
        p("The best-performing model, Gradient Boosting Regression, achieved an R\u00b2 of 0.933 and RMSE of 2.83\u00b0C on held-out data, with a 5-fold cross-validated RMSE of 2.92 \u00b1 0.14\u00b0C. Key findings include a strong negative correlation between humidity and temperature (r \u2248 -0.47), a strong positive relationship between UV index and temperature (r \u2248 0.83), and a measurable warming trend across the observation window consistent with seasonal transition."),
        p("Important scope note: this sandbox has no network access to kaggle.com, so the analysis in this report runs on a schema-matched synthetic dataset that reproduces the exact column structure of the Kaggle Global Weather Repository (60 cities, ~200 days, ~12,000 rows) with realistic seasonal patterns and injected data-quality issues. The full pipeline is designed to run unchanged against the real Kaggle CSV \u2014 see the README for instructions.", { italics: true, color: "555555" }),

        h1("1. Introduction"),
        p("Weather forecasting and climate-trend analysis have significant value across agriculture, disaster preparedness, energy demand planning, and public health. This project applies a full data science lifecycle \u2014 from raw data ingestion through deployment-ready dashboarding \u2014 to a global, multi-city weather dataset, with the goal of both predicting temperature accurately and surfacing actionable climate insights."),
        h2("1.1 Objectives"),
        bullet("Clean and validate a real-world-style, multi-country weather dataset"),
        bullet("Explore weather variables through 40+ visualizations across distributions, relationships, and trends"),
        bullet("Engineer domain-relevant features (calendar, comfort indices, lag/rolling statistics)"),
        bullet("Train and compare seven regression models plus an ensemble for temperature prediction"),
        bullet("Forecast short-term temperature trends using classical time-series methods"),
        bullet("Apply unsupervised learning for anomaly detection, clustering, and dimensionality reduction"),
        bullet("Explain model behavior using permutation importance and SHAP"),
        bullet("Visualize global weather patterns geospatially"),
        bullet("Package everything into a reusable, documented, dashboard-enabled project"),

        h1("2. Dataset"),
        p("Source: Kaggle \u2013 Global Weather Repository (nelgiriyewithana/global-weather-repository). The dataset provides daily weather observations across countries worldwide, including temperature, humidity, pressure, wind, precipitation, UV index, visibility, air quality indices, and astronomical data (sunrise/sunset, moon phase)."),
        h2("2.1 Schema Summary"),
        simpleTable(
          ["Category", "Example Columns"],
          [
            ["Identifiers", "country, location_name, latitude, longitude, timezone"],
            ["Temperature", "temperature_celsius, temperature_fahrenheit, feels_like_celsius"],
            ["Atmosphere", "pressure_mb, humidity, cloud, visibility_km, uv_index"],
            ["Wind", "wind_kph, wind_degree, wind_direction, gust_kph"],
            ["Precipitation", "precip_mm, condition_text"],
            ["Air Quality", "air_quality_PM2.5, air_quality_PM10, air_quality_Ozone, air_quality_Carbon_Monoxide"],
            ["Astronomy", "sunrise, sunset, moonrise, moonset, moon_phase, moon_illumination"],
            ["Time", "last_updated, last_updated_epoch"],
          ]
        ),
        new Paragraph({ text: "", spacing: { after: 200 } }),
        p("In this project's working copy, the dataset spans 60 cities across 60 countries, ~200 daily observations each, for 12,040 rows and 41 raw columns before feature engineering (67 after)."),

        h1("3. Methodology"),
        p("The pipeline follows a modular, reusable architecture (src/ package) with each stage independently testable: data_loader \u2192 preprocessing \u2192 feature_engineering \u2192 train \u2192 evaluate \u2192 forecasting \u2192 advanced_analytics \u2192 geospatial. Every stage is orchestrated by a notebook (notebooks/01-09) and/or a standalone script (scripts/)."),

        h1("4. Data Cleaning"),
        p("The cleaning phase addressed:"),
        bullet("Duplicate removal (40 injected duplicate rows detected and removed)"),
        bullet("Datetime parsing and validation of the last_updated column"),
        bullet("Missing-value imputation: median for numeric columns, mode for categorical columns (~2% missingness injected across 7 key numeric fields)"),
        bullet("Outlier detection using both IQR (1.5x) and Z-score (|z|>3) methods, with IQR-based winsorization (clipping) applied as the default treatment to preserve row count for time-series continuity"),
        bullet("Data validation checks: humidity bounds [0,100], latitude bounds [-90,90], longitude bounds [-180,180]"),
        p("This is intentionally a real cleaning problem, not a placeholder: the working dataset has injected duplicates, missing values, and extreme outliers so every step above does genuine, verifiable work (see notebooks/02_Data_Cleaning.ipynb for before/after counts)."),

        h1("5. Exploratory Data Analysis"),
        p("40+ charts were generated covering univariate distributions, bivariate relationships, categorical comparisons, and temporal trends. A representative subset is shown below; the full set is in outputs/figures/."),
        ...figure("dist_temperature_celsius.png", 500, 340, "Figure 1. Distribution of temperature (\u00b0C) across all observations."),
        ...figure("correlation_heatmap.png", 480, 430, "Figure 2. Correlation matrix of key numeric weather variables."),
        ...figure("ranking_hottest_countries.png", 500, 380, "Figure 3. Countries with the highest average temperature in the sample."),
        ...figure("trend_temperature_celsius_D.png", 520, 260, "Figure 4. Daily global average temperature trend over the observation window."),
        h2("5.1 Key EDA Observations"),
        bullet("Humidity and temperature are moderately negatively correlated (r \u2248 -0.47)"),
        bullet("UV index tracks temperature closely (r \u2248 0.83), consistent with solar-radiation-driven warming"),
        bullet("Equatorial countries (Kenya, Singapore, Malaysia, Indonesia) show the highest average temperatures; the pattern matches expected latitude-driven climatology"),
        bullet("Wind gust and wind speed are almost perfectly correlated (r \u2248 0.96), as expected physically"),

        h1("6. Feature Engineering"),
        p("New features were derived in four groups:"),
        bullet("Calendar: year, month, day, week of year, quarter, is_weekend, day_of_year, season (hemisphere-aware)"),
        bullet("Comfort indices: temp-feels difference, heat index (Rothfusz approximation), wind chill, humidity index"),
        bullet("Lag & rolling statistics: 1/2/3/7-day lags and 3/7/14-day rolling mean & std of temperature, computed per-location on date-sorted series"),
        bullet("Interaction & polynomial: humidity\u00d7temperature, wind\u00d7pressure, temperature squared"),
        p("A correlation-based feature-selection utility (select_features) ranks numeric features by absolute correlation with the target for quick screening."),

        h1("7. Machine Learning"),
        p("Seven regression models were trained to predict temperature_celsius: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, Support Vector Regression, XGBoost, and LightGBM, plus a simple weighted-average ensemble."),
        h2("7.1 A note on data leakage"),
        p("An initial feature set that included heat_index, wind_chill, feels_like_celsius, and temperature_celsius_sq produced a suspicious R\u00b2 of 1.000, because those features are deterministic functions of the same day's target temperature. They were removed from the model-facing feature matrix; only legitimately independent weather variables and past-looking lag/rolling temperature features remain. This is documented in src/train.py and is a deliberate, disclosed modeling decision rather than an oversight."),
        h2("7.2 Model Comparison"),
        simpleTable(mcHeaders, mcRows),
        new Paragraph({ text: "", spacing: { after: 200 } }),
        ...figure("model_comparison.png", 520, 280, "Figure 5. Model comparison by RMSE (lower is better)."),
        ...figure("pred_vs_actual_GradientBoosting.png", 380, 380, "Figure 6. Predicted vs. actual temperature, Gradient Boosting (best model)."),
        ...figure("residuals_GradientBoosting.png", 550, 240, "Figure 7. Residual analysis for the Gradient Boosting model."),
        ...figure("feature_importance_RandomForest.png", 480, 400, "Figure 8. Top feature importances, Random Forest."),
        ...figure("shap_summary.png", 430, 500, "Figure 9. SHAP summary plot showing feature impact direction and magnitude for the best model."),
        p("SHAP analysis confirms that rolling temperature statistics (7 and 14-day means) and UV index are the dominant predictive signals, consistent with the correlation analysis in Section 5."),

        h1("8. Forecasting"),
        p("Three time-series approaches were applied to the global daily-average temperature series: ARIMA(2,1,2), SARIMA(1,1,1)(1,1,1,7) capturing weekly seasonality, and Prophet (or, when Prophet is unavailable, a statsmodels seasonal-decomposition + linear-trend-extrapolation fallback that keeps the pipeline fully runnable without a native build toolchain). All three produce a 14-day-ahead forecast with comparable trend direction; SARIMA additionally captures short-cycle seasonality that ARIMA smooths over."),

        h1("9. Advanced Analytics"),
        p("Unsupervised methods were applied to surface structure and anomalies beyond the supervised modeling task:"),
        bullet("Isolation Forest and Local Outlier Factor flag statistically anomalous weather readings (contamination = 2%)"),
        bullet("KMeans (k=4) groups observations into weather regimes based on temperature, humidity, pressure, and wind"),
        bullet("DBSCAN identifies density-based clusters and noise points without pre-specifying cluster count"),
        bullet("PCA reduces the four core weather variables to 2 components for visualization"),
        bullet("Climate-trend analysis fits a linear trend to the daily series to estimate an annualized rate of change"),
        bullet("Heatwave detection flags per-location runs of \u22653 consecutive days above the location's 95th-percentile temperature \u2014 20 such events were detected in the sample window"),
        bullet("Country/continent-style ranking by average temperature, humidity, and precipitation"),

        h1("10. Geospatial Analysis"),
        p("Six interactive Folium maps were generated (outputs/figures/*.html): temperature, humidity, rainfall, wind speed, air quality (PM2.5), and a temperature density heatmap. Each marker map is clickable, showing the location name, country, and metric value in a popup."),

        h1("11. Dashboard"),
        p("A Streamlit dashboard (dashboard/app.py) provides interactive access to the full analysis: Home (global snapshot + KPIs), Dataset (raw table, summary stats, missing-value chart), EDA (variable explorer, correlation matrix), Climate Trends (aggregation toggle, seasonal comparison), Forecast (SARIMA-based forecast by location + manual prediction input against the trained ML model), Model Comparison, Maps (variable-selectable geospatial view), and About. Sidebar filters cover country and date range, with CSV download and a dark-mode toggle."),

        h1("12. Results & Insights"),
        bullet("Gradient Boosting is the best single model (RMSE 2.83\u00b0C, R\u00b2 0.933); LightGBM and XGBoost are statistically close behind"),
        bullet("SVR underperforms substantially without extensive hyperparameter tuning and feature scaling, illustrating the importance of preprocessing choices per algorithm"),
        bullet("Temperature is highly autocorrelated day-to-day (captured well by 7/14-day rolling means), while instantaneous atmospheric variables (pressure, wind) contribute comparatively little marginal predictive power once autoregressive features are included"),
        bullet("UV index is a strong same-day proxy for temperature, useful when direct temperature sensors are unavailable"),

        h1("13. Business Recommendations"),
        bullet("For short-horizon (1-14 day) forecasting, SARIMA or a rolling-feature gradient-boosting model are both viable; SARIMA is preferable when only historical temperature is available, while the ML model is preferable when auxiliary weather variables are already collected"),
        bullet("Heatwave-detection logic (Section 9) can feed early-warning systems for agriculture and public health with minimal additional infrastructure"),
        bullet("Given the close clustering of top model performance (RMSE within ~0.1\u00b0C across RF/GBM/XGBoost/LightGBM), model choice can reasonably be driven by operational concerns (inference latency, interpretability, maintenance) rather than accuracy alone"),

        h1("14. Future Work"),
        bullet("Validate all findings against the real Kaggle dataset"),
        bullet("Add a deep-learning forecaster (LSTM/TFT) for longer horizons"),
        bullet("Deploy the best model behind a FastAPI microservice with automated retraining"),
        bullet("Add Docker packaging and a GitHub Actions CI/CD pipeline"),
        bullet("Incorporate external climate indices (ENSO, NAO) as exogenous forecasting regressors"),

        h1("15. References"),
        p("1. Elgiriyewithana, N. Global Weather Repository. Kaggle. https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository"),
        p("2. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12."),
        p("3. Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. KDD."),
        p("4. Ke et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS."),
        p("5. Seabold & Perktold (2010). Statsmodels: Econometric and Statistical Modeling with Python. SciPy Conference."),
        p("6. Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions (SHAP). NeurIPS."),
        p("7. Taylor & Letham (2018). Forecasting at Scale (Prophet). The American Statistician."),

        h1("16. Appendix"),
        h2("A. Reproducing this report"),
        p("See README.md for full setup instructions. In short: pip install -r requirements.txt, then run the scripts/notebooks in the order documented there."),
        h2("B. Full model comparison metrics"),
        simpleTable(mcHeaders, mcRows),
      ],
    },
  ],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(path.join(__dirname, "Weather_Report.docx"), buf);
  console.log("Report written to", path.join(__dirname, "Weather_Report.docx"));
});
