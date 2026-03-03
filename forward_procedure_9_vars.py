import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

INPUT_FILE  = "Data analytics group project.xlsx"
SHEET_NAME  = "Data-Sheet (2)"
OUTPUT_FILE = "forward_selection_results.xlsx"

DEPENDENT = "P/B Ratio"

PREDICTORS = [
    "P/E Ratio",
    "LN(Assets)",
    "ROE",
    "SGrowth",
    "Debt/EBITDA Ratio",
    "D2834",
    "D2835",
    "ROE^2",
    "Sgrowth^2",
]

XI_LABELS = {var: f"X{i+1}" for i, var in enumerate(PREDICTORS)}

IGNORE_SUFFIXES = (".1",)
IGNORE_EXACT    = {"Company", "Ticker", "SIC 3", "SIC 4"}


def load_data(filepath, sheet):
    df = pd.read_excel(filepath, sheet_name=sheet, header=1)
    drop_cols = []
    for col in df.columns:
        col_str = str(col)
        if col_str in IGNORE_EXACT:
            drop_cols.append(col)
        elif col_str.startswith("Unnamed"):
            drop_cols.append(col)
        elif any(col_str.endswith(sfx) for sfx in IGNORE_SUFFIXES):
            drop_cols.append(col)
    return df.drop(columns=drop_cols, errors="ignore")


def fit_ols(df, y_col, x_cols):
    needed = [y_col] + x_cols
    sub = df[needed].copy()
    for col in needed:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna()
    y = sub[y_col]
    X = sm.add_constant(sub[x_cols], has_constant="add")
    return sm.OLS(y, X).fit()


def format_variables_in_model(selected):
    parts = [f"{var} ({XI_LABELS[var]})" for var in selected]
    return "+".join(parts)


def format_added_variable(var, round_num):
    if round_num == 1:
        return var
    return f"{var} ({XI_LABELS[var]})"


def forward_selection(df):
    remaining  = list(PREDICTORS)
    selected   = []
    best_adjr2 = -np.inf
    steps_rows  = []
    tested_rows = []
    round_num   = 0

    while remaining:
        round_num += 1
        round_best = {"adjr2": -np.inf, "var": None, "result": None}

        for candidate in remaining:
            trial_vars = selected + [candidate]
            result = fit_ols(df, DEPENDENT, trial_vars)

            tested_rows.append({
                "round":            round_num,
                "variables_tested": " + ".join(trial_vars),
                "added_candidate":  candidate,
                "n_obs":            int(result.nobs),
                "R2":               result.rsquared,
                "Adjusted_R2":      result.rsquared_adj,
                "AIC":              result.aic,
                "BIC":              result.bic,
            })

            if result.rsquared_adj > round_best["adjr2"]:
                round_best = {
                    "adjr2":  result.rsquared_adj,
                    "var":    candidate,
                    "result": result,
                    "vars":   trial_vars,
                }

        if round_best["adjr2"] <= best_adjr2:
            print(f"  Round {round_num}: '{round_best['var']}' does not improve Adj R2 "
                  f"({round_best['adjr2']:.8f} <= {best_adjr2:.8f}). Stopping.")
            break

        best_adjr2 = round_best["adjr2"]
        chosen_var = round_best["var"]
        chosen_res = round_best["result"]
        selected.append(chosen_var)
        remaining.remove(chosen_var)

        print(f"  Round {round_num}: Added '{chosen_var}' -> "
              f"Adj R2 = {best_adjr2:.8f} | Model: {' + '.join(selected)}")

        steps_rows.append({
            "round":              round_num,
            "variables_in_model": format_variables_in_model(selected),
            "added_variable":     format_added_variable(chosen_var, round_num),
            "n_obs":              int(chosen_res.nobs),
            "R2":                 chosen_res.rsquared,
            "Adjusted_R2":        chosen_res.rsquared_adj,
            "AIC":                chosen_res.aic,
            "BIC":                chosen_res.bic,
        })

    if not selected:
        raise RuntimeError("No variables were selected.")

    final_result = fit_ols(df, DEPENDENT, selected)
    return steps_rows, tested_rows, final_result, selected


def summary_to_df(result):
    rows = []
    scalars = {
        "Dep. Variable":    result.model.endog_names,
        "R-squared":        result.rsquared,
        "Adj. R-squared":   result.rsquared_adj,
        "F-statistic":      result.fvalue,
        "Prob(F-stat)":     result.f_pvalue,
        "AIC":              result.aic,
        "BIC":              result.bic,
        "No. Observations": int(result.nobs),
        "Df Residuals":     int(result.df_resid),
        "Df Model":         int(result.df_model),
    }
    for k, v in scalars.items():
        rows.append({"section": "Model Info", "term": k, "value": v,
                     "std_err": "", "t": "", "p": "", "[0.025": "", "0.975]": ""})

    rows.append({k: "" for k in ["section", "term", "value", "std_err", "t", "p", "[0.025", "0.975]"]})

    conf = result.conf_int()
    for term in result.params.index:
        rows.append({
            "section": "Coefficients",
            "term":    term,
            "value":   result.params[term],
            "std_err": result.bse[term],
            "t":       result.tvalues[term],
            "p":       result.pvalues[term],
            "[0.025":  conf.loc[term, 0],
            "0.975]":  conf.loc[term, 1],
        })

    return pd.DataFrame(rows)


def set_number_format(ws, df, num_fmt="0.000000000000"):
    for col_idx, col_name in enumerate(df.columns, start=1):
        if pd.api.types.is_float_dtype(df[col_name]):
            col_letter = get_column_letter(col_idx)
            for cell in ws[col_letter][1:]:
                cell.number_format = num_fmt


def export_results(steps_rows, tested_rows, final_result, selected, output_path):
    steps_df  = pd.DataFrame(steps_rows)
    tested_df = pd.DataFrame(tested_rows)
    final_df  = summary_to_df(final_result)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        steps_df.to_excel( writer, sheet_name="Forward Steps", index=False)
        tested_df.to_excel(writer, sheet_name="Tested Models", index=False)
        final_df.to_excel( writer, sheet_name="Final Model",   index=False)

    wb = load_workbook(output_path)
    set_number_format(wb["Forward Steps"], steps_df)
    set_number_format(wb["Tested Models"], tested_df)

    ws_final = wb["Final Model"]
    float_cols = [c for c in final_df.columns
                  if final_df[c].apply(lambda x: isinstance(x, float)).any()]
    for col_idx, col_name in enumerate(final_df.columns, start=1):
        if col_name in float_cols:
            col_letter = get_column_letter(col_idx)
            for cell in ws_final[col_letter][1:]:
                if isinstance(cell.value, float):
                    cell.number_format = "0.000000000000"

    wb.save(output_path)


def main():
    print(f"Loading '{INPUT_FILE}', sheet '{SHEET_NAME}' ...")
    df = load_data(INPUT_FILE, SHEET_NAME)
    print(f"  Rows loaded: {len(df)}\n")

    print("Running Forward Selection ...")
    steps_rows, tested_rows, final_result, selected = forward_selection(df)

    print(f"\nExporting results to '{OUTPUT_FILE}' ...")
    export_results(steps_rows, tested_rows, final_result, selected, OUTPUT_FILE)

    print("\n" + "=" * 54)
    print(f"Output file : {os.path.abspath(OUTPUT_FILE)}")
    print(f"Final model : {DEPENDENT} ~ {' + '.join(selected)}")
    print(f"  n_obs     : {int(final_result.nobs)}")
    print(f"  R2        : {final_result.rsquared:.8f}")
    print(f"  Adj R2    : {final_result.rsquared_adj:.8f}")
    print(f"  AIC       : {final_result.aic:.6f}")
    print(f"  BIC       : {final_result.bic:.6f}")
    print("=" * 54 + "\n")
    print(final_result.summary())


if __name__ == "__main__":
    main()
