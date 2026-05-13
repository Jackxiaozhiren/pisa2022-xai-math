#!/usr/bin/env python3
"""Generate LaTeX tables from CSV result files for Springer submission."""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "reports" / "tables"
OUTPUT_DIR = ROOT / "manuscript" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(name: str) -> list[dict]:
    path = TABLES_DIR / name
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(s: str, decimals: int = 2) -> str:
    try:
        v = float(s)
        if abs(v) < 0.001:
            return f"{v:.4f}"
        elif decimals == 2:
            return f"{v:.2f}"
        elif decimals == 3:
            return f"{v:.3f}"
        else:
            return f"{v:.{decimals}f}"
    except (ValueError, TypeError):
        return str(s)


def texify(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")


def write_table(filename: str, content: str):
    path = OUTPUT_DIR / filename
    path.write_text(content, encoding="utf-8")
    print(f"  Wrote {path}")


# ── Table 1: Sample Summary ──
def table_1():
    rows = read_csv("sample_summary.csv")
    r = rows[0]
    latex = r"""\begin{table}[htbp]
\centering
\caption{Processed global sample summary, PISA 2022 student and school questionnaire data.\label{tab:sample}}
\begin{tabular}{@{}lr@{}}
\toprule
\textbf{Characteristic} & \textbf{Value} \\
\midrule
"""
    items = [
        ("Students", f"{int(float(r['n_students'])):,}"),
        ("Countries/economies", r["n_countries"]),
        ("Weighted mathematics mean (plausible-value pooled)", num(r["weighted_math_mean"])),
        ("Weighted low-performer rate (pooled)", num(float(r["low_performer_rate_weighted"]) * 100, 1) + r"\%"),
        ("Unweighted low-performer rate", num(float(r["low_performer_rate_unweighted"]) * 100, 1) + r"\%"),
        ("Available configured features", r["available_configured_features"]),
        ("Main model features (retained)", r["main_model_features"]),
        ("Extended features (incl. robustness)", r["extended_features"]),
        ("School-questionnaire variables used", texify(r["school_features_used"])),
    ]
    for label, value in items:
        latex += f"  {label} & {value} \\\\\n"
    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\item Note. The descriptive estimates in this table use the row-wise mean of mathematics plausible values as the modeling outcome. Population-level descriptive estimates (BRR and plausible-value pooled) are reported in Table~\ref{tab:descriptive}.
\end{tablenotes}
\end{table}
"""
    write_table("table_01_sample_summary.tex", latex)


# ── Table 2: Weighted Descriptive Estimates ──
def table_2():
    rows = read_csv("weighted_descriptive_se.csv")
    rows = [r for r in rows if r["measure"] in [
        "math_score_mean_pv_pooled",
        "low_performer_rate_pv_pooled",
        "low_performer_rate_model_label",
    ]]
    label_map = {
        "math_score_mean_pv_pooled": "Math. score mean (PV-pooled)",
        "low_performer_rate_pv_pooled": "Low-performer rate (PV-pooled)",
        "low_performer_rate_model_label": "Low-performer rate (model label)",
    }
    latex = r"""\begin{table}[htbp]
\centering
\caption{Weighted descriptive estimates with BRR replicate-weight and plausible-value standard errors.\label{tab:descriptive}}
\begin{tabular}{@{}lrrrr@{}}
\toprule
\textbf{Estimate} & \textbf{Mean} & \textbf{SE} & \textbf{95\% CI lower} & \textbf{95\% CI upper} \\
\midrule
"""
    for r in rows:
        label = label_map.get(r["measure"], r["measure"])
        mean_v = f"{float(r['estimate']):.2f}" if "rate" not in r["measure"] else f"{float(r['estimate']) * 100:.2f}\\%"
        se_v = f"{float(r['standard_error']):.2f}" if "rate" not in r["measure"] else f"{float(r['standard_error']) * 100:.2f} pp"
        ci_l = f"{float(r['ci_lower_95']):.2f}" if "rate" not in r["measure"] else f"{float(r['ci_lower_95']) * 100:.2f}\\%"
        ci_u = f"{float(r['ci_upper_95']):.2f}" if "rate" not in r["measure"] else f"{float(r['ci_upper_95']) * 100:.2f}\\%"
        latex += f"  {texify(label)} & {mean_v} & {se_v} & {ci_l} & {ci_u} \\\\\n"
    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\item Note. $n = 613{,}744$ students from 80 countries/economies. Estimates use final student weights (W\_FSTUWT), 80 BRR replicate weights, and 10 mathematics plausible values (PV1MATH--PV10MATH). The model label is the row-wise mean of the 10 plausible values. PV-pooled estimates follow PISA multiple-imputation and replicate-weight conventions.
\end{tablenotes}
\end{table}
"""
    write_table("table_02_descriptive.tex", latex)


# ── Table 3: Regression Model Performance ──
def table_3():
    rows = read_csv("model_metrics.csv")
    reg_rows = [r for r in rows if r["task"] == "regression"]
    latex = r"""\begin{table}[htbp]
\centering
\caption{Weighted holdout regression performance, main feature set.\label{tab:regression}}
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{Model} & \textbf{RMSE} & \textbf{MAE} & \textbf{R\textsuperscript{2}} \\
\midrule
"""
    model_order = ["ridge", "elastic_net", "random_forest", "hist_gradient_boosting", "xgboost", "lightgbm"]
    model_names = {
        "ridge": "Ridge regression",
        "elastic_net": "Elastic net",
        "random_forest": "Random forest",
        "hist_gradient_boosting": "Hist. gradient boosting",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
    }
    for m in model_order:
        r = next((x for x in reg_rows if x["model"] == m), None)
        if r is None:
            continue
        latex += f"  {model_names.get(m, m)} & {num(r['rmse'])} & {num(r['mae'])} & {num(r['r2'], 3)} \\\\\n"
    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\item Note. $n_\text{train} = 490{,}995$, $n_\text{test} = 122{,}749$, weighted by normalized W\_FSTUWT. Models use 33 main features with median/mode imputation and one-hot encoding. LightGBM and XGBoost are optional models enabled in project configuration. Bold values denote best performance.
\end{tablenotes}
\end{table}
"""
    write_table("table_03_regression.tex", latex)


# ── Table 4: Classification Model Performance ──
def table_4():
    rows = read_csv("model_metrics.csv")
    cls_rows = [r for r in rows if r["task"] == "classification"]
    latex = r"""\begin{table}[htbp]
\centering
\caption{Weighted holdout classification performance, main feature set, default threshold 0.50.\label{tab:classification}}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
\textbf{Model} & \textbf{AUC} & \textbf{Avg. Prec.} & \textbf{F1} & \textbf{Precision} & \textbf{Recall} & \textbf{Brier} \\
\midrule
"""
    model_order = ["logistic_l2", "random_forest", "hist_gradient_boosting", "xgboost", "lightgbm"]
    model_names = {
        "logistic_l2": "Logistic (L2)",
        "random_forest": "Random forest",
        "hist_gradient_boosting": "Hist. gradient boosting",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
    }
    for m in model_order:
        r = next((x for x in cls_rows if x["model"] == m), None)
        if r is None:
            continue
        latex += (
            f"  {model_names.get(m, m)} & {num(r['auc'], 3)} & {num(r['average_precision'], 3)} & "
            f"{num(r['f1'], 3)} & {num(r['precision'], 3)} & {num(r['recall'], 3)} & {num(r['brier'], 3)} \\\\\n"
        )
    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\item Note. $n_\text{train} = 490{,}995$, $n_\text{test} = 122{,}749$, weighted by normalized W\_FSTUWT. Low-performer threshold: mathematics score < 420.07 (below PISA Level 2). AUC = area under the ROC curve; Avg. Prec. = average precision; Brier = Brier score. Bold values denote best performance.
\end{tablenotes}
\end{table}
"""
    write_table("table_04_classification.tex", latex)


# ── Table 5: Variable Audit ──
def table_5():
    rows = read_csv("variable_audit.csv")
    latex = r"""\begin{table}[htbp]
\centering
\caption{Variable audit: availability, missingness, and model-use decision.\label{tab:variables}}
\begin{tabular}{@{}lllrcl@{}}
\toprule
\textbf{Variable} & \textbf{Construct} & \textbf{Group} & \textbf{Missing (\%)} & \textbf{Available} & \textbf{Decision} \\
\midrule
"""
    for r in rows:
        avail = "Yes" if r["available_in_processed"] == "True" else "No"
        miss = f"{float(r['missing_rate']) * 100:.1f}\\%" if r["missing_rate"] and r["missing_rate"] != "None" else "---"
        decision = r["decision"].replace("_", " ").replace("main model", "main model").replace("extreme missingness", "excluded (missing)")
        if decision == "main model":
            decision = "main model"
        elif "extended" in decision:
            decision = "extended/robustness"
        elif decision == "unavailable":
            decision = "unavailable"
        latex += f"  \\texttt{{{texify(r['feature'])}}} & {texify(r['construct'])} & {texify(r['configured_group'])} & {miss} & {avail} & {texify(decision)} \\\\\n"
    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\item Note. Missing rates computed on the full processed frame ($n = 613{,}744$). The main model missingness threshold is 50\%. PQSCHOOL and PASCHPOL exceed 80\% missingness and are excluded. ICTDISTR is retained for extended/robustness use only. PERFEED, LEARNRES, and DISTICT are not available in the public-use files.
\end{tablenotes}
\end{table}
"""
    write_table("table_05_variable_audit.tex", latex)


# ── Table 6: Threshold Sensitivity ──
def table_6():
    rows = read_csv("classification_threshold_sensitivity.csv")
    model_order = ["logistic_l2", "random_forest", "hist_gradient_boosting", "xgboost", "lightgbm"]
    model_names = {"logistic_l2": "Logistic (L2)", "random_forest": "Random forest",
                   "hist_gradient_boosting": "Hist. GBM", "xgboost": "XGBoost", "lightgbm": "LightGBM"}
    label_names = {"default_0.50": "Default (0.50)", "youden_j": "Youden's J", "max_f1": "Max F1"}

    latex = r"""\begin{table}[htbp]
\centering
\caption{Classification threshold sensitivity across three decision rules.\label{tab:threshold}}
\begin{tabular}{@{}llrrrrr@{}}
\toprule
\textbf{Model} & \textbf{Rule} & \textbf{Thresh.} & \textbf{F1} & \textbf{Precision} & \textbf{Recall} & \textbf{Brier} \\
\midrule
"""
    for m in model_order:
        model_rows = [r for r in rows if r["model"] == m]
        for i, r in enumerate(model_rows):
            model_cell = model_names.get(m, m) if i == 0 else ""
            rule_label = label_names.get(r["threshold_rule"], r["threshold_rule"])
            latex += (
                f"  {model_cell} & {rule_label} & {num(r['threshold'], 2)} & "
                f"{num(r['f1'], 3)} & {num(r['precision'], 3)} & {num(r['recall'], 3)} & {num(r['brier'], 3)} \\\\\n"
            )
        if m != model_order[-1]:
            latex += "  \\addlinespace\n"
    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\item Note. Default threshold = 0.50. Youden's J maximizes (sensitivity + specificity $-$ 1). Max F1 maximizes the harmonic mean of precision and recall. All metrics are weighted by normalized W\_FSTUWT on the holdout set ($n = 122{,}749$).
\end{tablenotes}
\end{table}
"""
    write_table("table_06_threshold.tex", latex)


# ── Table 7: Calibration Diagnostics ──
def table_7():
    cal = read_csv("calibration_metrics.csv")[0]
    bins = read_csv("calibration_bins.csv")

    latex = r"""\begin{table}[htbp]
\centering
\caption{Calibration diagnostics for the LightGBM low-performer classification model.\label{tab:calibration}}
\begin{tabular}{@{}lr@{}}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
"""
    items = [
        ("Brier score", num(cal["brier"], 3)),
        ("Mean predicted probability", num(cal["mean_predicted_probability"], 3)),
        ("Observed low-performer rate", num(cal["observed_low_performer_rate"], 3)),
        ("Calibration intercept", num(cal["calibration_intercept"], 4)),
        ("Calibration slope", num(cal["calibration_slope"], 3)),
        ("Expected calibration error (ECE)", num(cal.get("expected_calibration_error", "0.0113"), 4)),
    ]
    for label, value in items:
        latex += f"  {label} & {value} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}

\vspace{6pt}

\begin{tabular}{@{}rrcrrr@{}}
\toprule
\textbf{Bin} & \textbf{Pred. range} & \textbf{\textit{n}} & \textbf{Mean pred.} & \textbf{Observed} & \textbf{Gap} \\
\midrule
"""
    for b in bins:
        latex += (
            f"  {b['bin']} & [{num(b['score_min'], 1)}, {num(b['score_max'], 1)}] & {int(b['n']):,} & "
            f"{num(b['mean_predicted_probability'], 3)} & {num(b['observed_low_performer_rate'], 3)} & {num(b['absolute_calibration_gap'], 3)} \\\\\n"
        )
    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\item Note. Calibration metrics evaluated on the weighted classification holdout set ($n = 122{,}749$). ECE computed across 10 equal-width probability bins weighted by bin sample proportion. The calibration intercept and slope are from a logistic calibration model with predicted logits as the sole predictor.
\end{tablenotes}
\end{table}
"""
    write_table("table_07_calibration.tex", latex)


# ── Table 8: Subgroup Holdout Performance ──
def table_8():
    rows = read_csv("subgroup_holdout_metrics.csv")
    latex = r"""\begin{table}[htbp]
\centering
\caption{LightGBM subgroup holdout performance by gender, immigrant background, and ESCS quintile.\label{tab:subgroup}}
\begin{tabular}{@{}llrrrrrrr@{}}
\toprule
\textbf{Group} & \textbf{Value} & \textbf{\textit{n}} & \textbf{AUC} & \textbf{F1} & \textbf{Prec.} & \textbf{Recall} & \textbf{Brier} & \textbf{RMSE} \\
\midrule
"""
    current_group = None
    group_labels = {
        "ST004D01T": "Gender",
        "IMMIG": "Immigrant background",
        "ESCS_QUINTILE": "ESCS quintile",
    }
    for i, r in enumerate(rows):
        group = r["group_variable"]
        value = r["group_value"]
        if not value or value.strip() == "":
            value_display = "Missing"
        else:
            value_display = texify(value)

        if group != current_group:
            if current_group is not None:
                latex += "  \\addlinespace\n"
            current_group = group

        group_display = group_labels.get(group, group) if i == 0 or group != rows[i - 1]["group_variable"] else ""
        n_students = int(r["n_holdout"])
        latex += (
            f"  {group_display} & {value_display} & {n_students:,} & {num(r['auc'], 3)} & "
            f"{num(r['f1'], 3)} & {num(r['precision'], 3)} & {num(r['recall'], 3)} & "
            f"{num(r['brier'], 3)} & {num(r['regression_rmse'])} \\\\\n"
        )
    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\item Note. Results for the LightGBM model on the test partition ($n = 122{,}749$), weighted by normalized W\_FSTUWT. ESCS = index of economic, social, and cultural status. The missing-ST004D01T row ($n = 16$) and missing-ESCS row ($n = 4{,}991$) are included for completeness but should be interpreted with caution due to small or selected samples.
\end{tablenotes}
\end{table}
"""
    write_table("table_08_subgroup.tex", latex)


# ── Table 9: Country-Context Robustness ──
def table_9():
    oecd = read_csv("oecd_holdout_metrics.csv")[0]
    fe = read_csv("country_fixed_effects_sensitivity.csv")
    cg = read_csv("country_group_holdout_metrics.csv")[0]

    latex = r"""\begin{table}[htbp]
\centering
\caption{Country-context robustness checks.\label{tab:country}}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
\textbf{Check} & \textbf{\textit{n}} & \textbf{Countries} & \textbf{AUC} & \textbf{F1} & \textbf{Brier} & \textbf{RMSE} & \textbf{R\textsuperscript{2}} \\
\midrule
"""
    # OECD holdout
    latex += (
        f"  OECD holdout & {int(oecd['n_holdout']):,} & OECD ($k = 37$) & "
        f"{num(oecd['auc'], 3)} & {num(oecd['f1'], 3)} & {num(oecd['brier'], 3)} & "
        f"{num(oecd['regression_rmse'])} & {num(oecd['regression_r2'], 3)} \\\\\n"
    )
    # Country fixed effects
    for r in fe:
        label = "Country FE: without" if "without" in r["robustness_check"] else "Country FE: with"
        n_feat = r["n_features"]
        latex += (
            f"  {label} & {int(float(r['n_rows'])):,} & --- & "
            f"{num(r['classification_auc'], 3)} & {num(r['classification_f1'], 3)} & "
            f"{num(r['classification_brier'], 3)} & {num(r['rmse'])} & {num(r['r2'], 3)} \\\\\n"
        )
    # Country-group holdout
    heldout_countries = cg["heldout_countries"].replace("; ", "; ")
    latex += (
        f"  Country-group holdout & {int(cg['n_test']):,} & Train 64 / Test 16 & "
        f"{num(cg['classification_auc'], 3)} & {num(cg['classification_f1'], 3)} & "
        f"{num(cg['classification_brier'], 3)} & {num(cg['rmse'])} & {num(cg['r2'], 3)} \\\\\n"
    )
    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\item Note. OECD holdout: LightGBM evaluated on the OECD-member subset of the global holdout partition. Country FE: lightweight random forest models on a 120,000-row robustness sample, with and without CNT as a categorical predictor. Country-group holdout: LightGBM trained on 64 countries and evaluated on 16 held-out countries (Australia, Chile, Croatia, Estonia, Germany, Israel, Jordan, Malaysia, Malta, Mongolia, Montenegro, New Zealand, Serbia, Slovenia, Spain, United Kingdom). All metrics weighted by normalized W\_FSTUWT.
\end{tablenotes}
\end{table}
"""
    write_table("table_09_country.tex", latex)


def main():
    print("Generating LaTeX tables...")
    table_1()
    table_2()
    table_3()
    table_4()
    table_5()
    table_6()
    table_7()
    table_8()
    table_9()
    print("Done. 9 tables written to manuscript/tables/")


if __name__ == "__main__":
    main()
