import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

INPUT_FILE = "Data analytics group project.xlsx"
SHEET_NAME = "Data-Sheet"
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
]

IGNORE_SUFFIXES = (".1",)
IGNORE_EXACT = {"Company", "Ticker", "SIC 3", "SIC 4"}

def load_data(filepath, sheet):
    df = pd.read_excel(filepath, sheet_name=sheet)

    drop_cols = []
    for col in df.columns:
        if col in IGNORE_EXACT:
            drop_cols.append(col)
        elif col.startswith("Unnamed"):
            drop_cols.append(col)
        elif any(col.endswith(s) for s in IGNORE_SUFFIXES):
            drop_cols.append(col)

    return df.drop(columns=drop_cols, errors="ignore")

def fit_model(df, y_col, x_cols):
    needed = [y_col] + x_cols
    sub = df[needed].copy()

    for col in needed:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")

    sub = sub.dropna()

    y = sub[y_col]
    X = sm.add_constant(sub[x_cols], has_constant="add")

    return sm.OLS(y, X).fit()

def forward_selection(df):
    remaining = list(PREDICTORS)
    selected = []
    best_adj_r2 = -np.inf

    steps = []
    tested = []
    round_num = 0

    while remaining:
        round_num += 1
        best = {"adjr2": -np.inf}

        for candidate in remaining:
            vars_test = selected + [candidate]
            result = fit_model(df, DEPENDENT, vars_test)

            tested.append({
                "round": round_num,
                "variables": " + ".join(vars_test),
                "candidate": candidate,
                "n_obs": int(result.nobs),
                "R2": result.rsquared,
                "Adjusted_R2": result.rsquared_adj,
                "AIC": result.aic,
                "BIC": result.bic
            })

            if result.rsquared_adj > best["adjr2"]:
                best = {
                    "adjr2": result.rsquared_adj,
                    "var": candidate,
                    "result": result,
                    "vars": vars_test
                }

        if best["adjr2"] <= best_adj_r2:
            break

        best_adj_r2 = best["adjr2"]
        selected.append(best["var"])
        remaining.remove(best["var"])

        r = best["result"]

        steps.append({
            "round": round_num,
            "variables_in_model": " + ".join(selected),
            "added_variable": best["var"],
            "n_obs": int(r.nobs),
            "R2": r.rsquared,
            "Adjusted_R2": r.rsquared_adj,
            "AIC": r.aic,
            "BIC": r.bic
        })

    final_model = fit_model(df, DEPENDENT, selected)

    return steps, tested, final_model, selected

def summary_to_df(result):
    rows = []

    stats = {
        "Dep. Variable": result.model.endog_names,
        "R-squared": result.rsquared,
        "Adj. R-squared": result.rsquared_adj,
        "F-statistic": result.fvalue,
        "Prob (F-stat)": result.f_pvalue,
        "AIC": result.aic,
        "BIC": result.bic,
        "No. Observations": int(result.nobs),
        "Df Residuals": int(result.df_resid),
        "Df Model": int(result.df_model)
    }

    for k, v in stats.items():
        rows.append({
            "section": "Model Info",
            "term": k,
            "value": v,
            "std_err": "",
            "t": "",
            "p": "",
            "[0.025": "",
            "0.975]": ""
        })

    rows.append({k: "" for k in ["section","term","value","std_err","t","p","[0.025","0.975]"]})

    conf = result.conf_int()

    for term in result.params.index:
        rows.append({
            "section": "Coefficients",
            "term": term,
            "value": result.params[term],
            "std_err": result.bse[term],
            "t": result.tvalues[term],
            "p": result.pvalues[term],
            "[0.025": conf.loc[term, 0],
            "0.975]": conf.loc[term, 1]
        })

    return pd.DataFrame(rows)

def set_number_format(ws, df):
    for col_idx, col_name in enumerate(df.columns, start=1):
        if pd.api.types.is_float_dtype(df[col_name]):
            col_letter = get_column_letter(col_idx)
            for cell in ws[col_letter][1:]:
                cell.number_format = "0.000000000000"

def export_results(steps, tested, final_model, output):
    steps_df = pd.DataFrame(steps)
    tested_df = pd.DataFrame(tested)
    final_df = summary_to_df(final_model)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        steps_df.to_excel(writer, sheet_name="Forward Steps", index=False)
        tested_df.to_excel(writer, sheet_name="Tested Models", index=False)
        final_df.to_excel(writer, sheet_name="Final Model", index=False)

    wb = load_workbook(output)

    set_number_format(wb["Forward Steps"], steps_df)
    set_number_format(wb["Tested Models"], tested_df)
  
    wb.save(output)

def main():
    df = load_data(INPUT_FILE, SHEET_NAME)
    steps, tested, final_model, selected = forward_selection(df)
    export_results(steps, tested, final_model, OUTPUT_FILE)

    print(os.path.abspath(OUTPUT_FILE))
    print("Final model:", DEPENDENT, "~", " + ".join(selected))

if __name__ == "__main__":
    main()
