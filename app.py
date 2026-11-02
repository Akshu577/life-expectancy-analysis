"""Streamlit app: explore life expectancy by country and try 'what if' changes on the 4-feature model."""
import numpy as np
import pandas as pd
import streamlit as st

from src.pipeline import TARGET, clean, fit_small_model, load_raw

st.set_page_config(page_title="Life Expectancy Explorer", page_icon="🌍", layout="wide")
st.title("What Affects Life Expectancy?")
st.caption("WHO data for 193 countries, 2000–2015. The model uses the 4 features that forward selection found "
           "predict as well as all 17. Source: github.com/Akshu577/life-expectancy-analysis")


@st.cache_resource
def setup():
    df, raw, features, log_cols, lower, upper = clean(load_raw())
    model, selected, rmse = fit_small_model(df, features)
    return raw, model, selected, rmse, log_cols, lower, upper


raw, model, selected, rmse, log_cols, lower, upper = setup()
countries = sorted(raw["Country"].unique())

tab1, tab2 = st.tabs(["Country trends", "What if?"])
with tab1:
    picked = st.multiselect("Countries", countries,
                            default=[c for c in ["Nepal", "India", "Bangladesh", "United States of America"]
                                     if c in countries])
    if picked:
        trend = raw[raw["Country"].isin(picked)].pivot_table(index="Year", columns="Country", values=TARGET)
        st.line_chart(trend, y_label="Life expectancy (years)")

with tab2:
    c1, c2 = st.columns(2)
    country = c1.selectbox("Start from a country", countries,
                           index=countries.index("Nepal") if "Nepal" in countries else 0)
    years = sorted(raw.loc[raw["Country"] == country, "Year"].unique())
    year = c2.selectbox("Year", years, index=len(years) - 1)
    row = raw[(raw["Country"] == country) & (raw["Year"] == year)].iloc[0]

    st.write("Change the values and see how the model's prediction moves.")
    limits = {"Income composition of resources": (0.0, 1.0, 0.01, "Human Development Index income part, 0 to 1"),
              "HIV/AIDS": (0.1, 51.0, 0.1, "Deaths per 1,000 live births, ages 0-4"),
              "Adult Mortality": (1.0, 725.0, 1.0, "Deaths per 1,000 adults aged 15-60"),
              "Diphtheria": (0.0, 99.0, 1.0, "DTP3 immunization coverage among 1-year-olds, %")}
    values = {}
    cols = st.columns(2)
    for i, f in enumerate(selected):
        lo, hi, step, help_text = limits[f]
        values[f] = cols[i % 2].slider(f, lo, hi, float(np.clip(row[f], lo, hi)), step, help=help_text,
                                       key=f"{f}-{country}-{year}")

    def predict(vals):
        x = []
        for f in selected:
            v = np.log1p(vals[f]) if f in log_cols else vals[f]
            x.append(np.clip(v, lower[f], upper[f]))  # same transform and capping as training
        return float(model.predict(pd.DataFrame([x], columns=selected))[0])

    base = predict({f: row[f] for f in selected})
    new = predict(values)
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Actual ({year})", f"{row[TARGET]:.1f} years")
    m2.metric("Model with real values", f"{base:.1f} years")
    m3.metric("Model with your values", f"{new:.1f} years", delta=f"{new - base:+.1f}")
    st.caption(f"Typical error of this model on test data: about {rmse:.1f} years. It shows associations in the "
               "data, not proof that changing one factor would cause the change.")
