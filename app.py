# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------
# Section 1: Gather Data
# --------------------------------------------
st.title("Monte Carlo Simulation - DCF")
st.header("Schritt 1 - Daten aus yfinance sammeln:")

# # Replace text_input with selectbox to choose exactly one ticker
# all_tickers = ["BMW.DE", "VOW3.DE", "MBG.DE"]
# ticker_symbol = st.selectbox("Wähle einen Ticker aus:", all_tickers, index=2)
ticker_symbol = st.text_input("Aktien Ticker-Symbol eingeben (z.B. AAPL, MSFT, MBG.DE, BMW.DE, VOW3.DE ):", "MBG.DE")


ticker = yf.Ticker(ticker_symbol)

income_statement = ticker.financials.T  # Yearly
balance_sheet = ticker.balance_sheet.T  # Yearly
cashflow_statement = ticker.cashflow.T  # Yearly

# Current share price
try:
    current_share_price = ticker.info["currentPrice"]
except:
    current_share_price = float("nan")

# Attempt to get 'number_of_shares'
if balance_sheet.empty:
    st.warning("Keine Bilanzdaten (Balance Sheet) vorhanden.")
    number_of_shares = 1
else:
    try:
        latest_date = balance_sheet.index[0]
        number_of_shares = balance_sheet.loc[latest_date, "Ordinary Shares Number"]
    except:
        number_of_shares = 1

# Extract columns needed for DCF
try:
    revenue = income_statement["Total Revenue"]
    ebit = income_statement["EBIT"]
    tax_rate = (
        income_statement["Tax Provision"] / income_statement["Pretax Income"]
    )
    depreciation = cashflow_statement["Depreciation And Amortization"]
    reinvestment = cashflow_statement["Capital Expenditure"]
    cash = balance_sheet["Cash And Cash Equivalents"]
    debt = balance_sheet["Total Debt"]
except KeyError:
    st.error("Für diesen Ticker fehlen wichtige Daten für die DCF-Berechnung.")
    st.stop()

# Build main data DataFrame
data = pd.DataFrame(
    {
        "Revenue": revenue,
        "EBIT": ebit,
        "Tax Rate": tax_rate,
        "Depreciation": depreciation,
        "Reinvestment": reinvestment,
        "Cash": cash,
        "Debt": debt,
    }
)
data = data.sort_index(ascending=True)  # Sort by date ascending

data["Revenue Growth Rate"] = (
    (data["Revenue"] - data["Revenue"].shift(1)) / data["Revenue"].shift(1)
)
data["EBIT Margin"] = data["EBIT"] / data["Revenue"]

st.write("Historische Daten:")
st.dataframe(data)

# --------------------------------------------
# Section 2: Define Variables
# --------------------------------------------
st.header("Schritt 2 - Definiere Annahmen:")

with st.expander("Annahmen anpassen"):
    wacc = st.number_input("WACC (Weighted Average Cost of Capital):", value=0.08)
    tgr = st.number_input("TGR (Terminal Growth Rate):", value=0.03)

    growth_rate_avg = st.number_input(
        "Umsatzwachstum:", value=data["Revenue Growth Rate"].mean()
    )
    margin_avg = st.number_input(
        "Marge:", value=data["EBIT Margin"].mean()
    )
    tax_rate_avg = st.number_input(
        "Steuersatz:", value=data["Tax Rate"].mean()
    )
    depreciation_avg = st.number_input(
        "Abschreibungs- und Abgrenzungsrate:",
        value=(data["Depreciation"].mean() / data["Revenue"].mean()),
    )
    reinvestment_rate = st.number_input(
        "Investitionsrate:",
        value=(data["Reinvestment"].mean() / data["Revenue"].mean()),
    )

cash_avg = data["Cash"].mean()
debt_avg = data["Debt"].mean()

# --------------------------------------------
# Section 3: Projection (Classic DCF)
# --------------------------------------------
st.header("Schritt 3 - Finanzielle Projektion:")

newest_revenue = data["Revenue"].iloc[-1]
newest_ebit = data["EBIT"].iloc[-1]
newest_tax_rate = data["Tax Rate"].iloc[-1]
newest_depreciation = data["Depreciation"].iloc[-1]
newest_reinvestment = data["Reinvestment"].iloc[-1]

projected_revenues = [newest_revenue]
projected_ebit = []
projected_net_surplus = []
projected_depreciation = []
projected_reinvestment = []

for year in range(1, 11):
    next_revenue = projected_revenues[-1] * (1 + growth_rate_avg)
    projected_revenues.append(next_revenue)

    next_ebit = next_revenue * margin_avg
    projected_ebit.append(next_ebit)

    projected_net_surplus.append(next_ebit * (1 - tax_rate_avg))
    projected_depreciation.append(next_revenue * depreciation_avg)
    projected_reinvestment.append(next_revenue * reinvestment_rate)

projection_df = pd.DataFrame(
    {
        "Year": range(0, 11),
        "Projected Revenue": projected_revenues,
        "Projected EBIT": [newest_ebit] + projected_ebit,
        "Projected Net Surplus": [newest_ebit * (1 - newest_tax_rate)]
        + projected_net_surplus,
        "Projected Depreciation": [newest_depreciation]
        + projected_depreciation,
        "Projected Reinvestment": [newest_reinvestment]
        + projected_reinvestment,
    }
)

projection_df["Free Cashflow"] = (
    projection_df["Projected Net Surplus"]
    + projection_df["Projected Depreciation"]
    + projection_df["Projected Reinvestment"]
)

projection_df["Present Value"] = (
    projection_df["Free Cashflow"] / ((1 + wacc) ** projection_df["Year"])
)

# Terminal (Rest) Value from last year's FCF
projection_df["Rest Value"] = projection_df["Free Cashflow"] / (wacc - tgr)
projection_df["Present Rest Value"] = projection_df["Rest Value"] / (
    (1 + wacc) ** projection_df["Year"]
)


def calculate_total_company_value(row):
    # Sum of discounted FCF up to 'this' year + discounted terminal value at 'this' year
    return projection_df.loc[: row.name, "Present Value"].sum() + row["Present Rest Value"]


projection_df["Total Company Value"] = projection_df.apply(
    calculate_total_company_value, axis=1
)

projection_df["Implied Share Price"] = (
    projection_df["Total Company Value"] + cash_avg - debt_avg
) / number_of_shares

st.write("Projektionsergebnisse:")
st.dataframe(projection_df)

# Compare Implied Share Price with Current Share Price
try:
    current_share_price = round(current_share_price, 2)
except:
    current_share_price = float("nan")

implied_price_year_5 = round(projection_df.loc[5, "Implied Share Price"], 2)
implied_price_year_10 = round(projection_df.loc[10, "Implied Share Price"], 2)

st.write(f"Heutiger Aktienkurs: {current_share_price}")
col1, col2 = st.columns(2)
with col1:
    st.write(f"Implizierter Aktienkurs in 5 Jahren: {implied_price_year_5}")
with col2:
    st.write(f"Implizierter Aktienkurs in 10 Jahren: {implied_price_year_10}")

with col1:
    if implied_price_year_5 < current_share_price:
        st.write(
            "Bewertung im 5. Jahr:",
            f'<span style="color:red;">***Überbewertet***</span>',
            unsafe_allow_html=True,
        )
    else:
        st.write(
            "Bewertung im 5. Jahr:",
            f'<span style="color:green;">***Unterbewertet***</span>',
            unsafe_allow_html=True,
        )

with col2:
    if implied_price_year_10 < current_share_price:
        st.write(
            "Bewertung im 10. Jahr:",
            f'<span style="color:red;">***Überbewertet***</span>',
            unsafe_allow_html=True,
        )
    else:
        st.write(
            "Bewertung im 10. Jahr:",
            f'<span style="color:green;">***Unterbewertet***</span>',
            unsafe_allow_html=True,
        )


# --------------------------------------------
# Section 4: Monte Carlo Simulation
# --------------------------------------------
st.header("Schritt 4 - Monte Carlo Simulation")

def get_distribution_params_lognormal(name, default_mean, default_sigma, default_max):
    """
    Instead of left, mode, right, we now specify:
       - mean (in log-space)
       - sigma (in log-space)
       - maximum (for clipping)
    """
    cols = st.columns(3)
    mean_val = cols[0].number_input(f"{name} Mean (log-space)", value=default_mean, key=f"{name}_mean")
    sigma_val = cols[1].number_input(f"{name} Sigma", value=default_sigma, key=f"{name}_sigma")
    max_val = cols[2].number_input(f"{name} Maximum", value=default_max, key=f"{name}_max")
    return mean_val, sigma_val, max_val

def get_distribution_params_triangular(name, default_left, default_mode, default_right):
    cols = st.columns(3)
    left = cols[0].number_input(f"{name} Min", value=default_left, key=f"{name}_left")
    mode = cols[1].number_input(f"{name} Mode", value=default_mode, key=f"{name}_mode")
    right = cols[2].number_input(f"{name} Max", value=default_right, key=f"{name}_right")
    left, mode, right = sorted([left, mode, right])
    return left, mode, right

cols = st.columns(2)
n_simulations = cols[0].number_input("Anzahl Simulationen:", value=1000, min_value=1)
valuation_year = cols[1].selectbox("Bewertungsjahr auswählen:", [5, 10])

with st.expander("Verteilungsparameter anpassen"):
    st.markdown("### Normal-Verteilung (Umsatzwachstum)")
    gmean_col, gstd_col = st.columns(2)
    growth_rate_mean = gmean_col.number_input(
        "Umsatzwachstum - Mean", value=growth_rate_avg, key="growth_rate_mean"
    )
    growth_rate_std = gstd_col.number_input(
        "Umsatzwachstum - Std. Abw.", value=abs(growth_rate_avg) * 0.1, key="growth_rate_std"
    )

    # Lognormal for TGR with (mean in log-space, sigma, max-clip)
    st.markdown("### Lognormal-Verteilung (TGR)")
    tgr_mean_log, tgr_sigma, tgr_max = get_distribution_params_lognormal(
        "TGR",
        default_mean=np.log(tgr),  # log-space mean (since tgr ~ 0.03)
        default_sigma=0.2,         # initial guess
        default_max=tgr * 1.2      # max cap for TGR
    )

    # Triangular for WACC, Margin, Tax Rate, Depreciation, Reinvestment
    st.markdown("### Dreieck-Verteilungen")
    wacc_left, wacc_mode, wacc_right = get_distribution_params_triangular(
        "WACC", wacc*0.8, wacc, wacc*1.2
    )
    margin_left, margin_mode, margin_right = get_distribution_params_triangular(
        "Marge", margin_avg*0.8, margin_avg, margin_avg*1.2
    )
    tax_rate_left, tax_rate_mode, tax_rate_right = get_distribution_params_triangular(
        "Steuersatz", tax_rate_avg*0.95, tax_rate_avg, tax_rate_avg*1.05
    )
    dep_left, dep_mode, dep_right = get_distribution_params_triangular(
        "A&A-Rate", depreciation_avg*0.8, depreciation_avg, depreciation_avg*1.2
    )
    reinv_left, reinv_mode, reinv_right = get_distribution_params_triangular(
        "Investitionsrate", reinvestment_rate*0.8, reinvestment_rate, reinvestment_rate*1.2
    )

# Generate distributions
wacc_dist = np.random.triangular(
    left=wacc_left, mode=wacc_mode, right=wacc_right, size=n_simulations
)
growth_rate_dist = np.random.normal(
    loc=growth_rate_mean, scale=growth_rate_std, size=n_simulations
)
margin_dist = np.random.triangular(
    left=margin_left, mode=margin_mode, right=margin_right, size=n_simulations
)
tax_rate_dist = np.random.triangular(
    left=tax_rate_left, mode=tax_rate_mode, right=tax_rate_right, size=n_simulations
)
dep_dist = np.random.triangular(
    left=dep_left, mode=dep_mode, right=dep_right, size=n_simulations
)
reinv_dist = np.random.triangular(
    left=reinv_left, mode=reinv_mode, right=reinv_right, size=n_simulations
)

# Now the lognormal TGR with mean = tgr_mean_log, sigma = tgr_sigma, clipped at tgr_max
raw_tgr_dist = np.random.lognormal(mean=tgr_mean_log, sigma=tgr_sigma, size=n_simulations)
# Clip TGR to not exceed tgr_max
tgr_dist = np.clip(raw_tgr_dist, 0, tgr_max)

implied_prices = []

for i in range(n_simulations):
    simulated_wacc = wacc_dist[i]
    simulated_tgr = tgr_dist[i]
    simulated_gr = growth_rate_dist[i]
    simulated_margin = margin_dist[i]
    simulated_tax = tax_rate_dist[i]
    simulated_dep = dep_dist[i]
    simulated_reinv = reinv_dist[i]

    # Project each year up to valuation_year
    rev = newest_revenue
    total_dcf = 0.0

    for yr in range(1, valuation_year + 1):
        rev = rev * (1 + simulated_gr)
        ebit = rev * simulated_margin
        net_surplus = ebit * (1 - simulated_tax)
        dep = rev * simulated_dep
        reinv = rev * simulated_reinv  # typically negative if CapEx is negative in statements

        fcf = net_surplus + dep + reinv
        discounted_fcf = fcf / ((1 + simulated_wacc) ** yr)
        total_dcf += discounted_fcf

    # Terminal value
    fcf_terminal = fcf * (1 + simulated_tgr)
    if (simulated_wacc - simulated_tgr) > 0:
        terminal_value = fcf_terminal / (simulated_wacc - simulated_tgr)
    else:
        terminal_value = fcf_terminal / max(0.01, simulated_wacc - simulated_tgr)

    discounted_terminal_value = terminal_value / ((1 + simulated_wacc) ** valuation_year)

    # Sum firm value
    total_firm_value = total_dcf + discounted_terminal_value + cash_avg - debt_avg
    implied_share_price = total_firm_value / number_of_shares
    implied_prices.append(implied_share_price)

mean_implied_price = np.mean(implied_prices)
std_implied_price = np.std(implied_prices)

st.subheader(f"Prognose für Jahr {valuation_year}")
if mean_implied_price > current_share_price:
    valuation = '<span style="color:green;">***Unterbewertet***</span>'
else:
    valuation = '<span style="color:red;">***Überbewertet***</span>'

cols = st.columns(2)
cols[0].write(f"Heutiger Aktienkurs: {current_share_price:.2f}")
cols[1].write(f"Bewertung: {valuation}", unsafe_allow_html=True)

cols = st.columns(2)
cols[0].write(f"⌀ impliziter Aktienkurs: {mean_implied_price:.2f}")
cols[1].write(f"Standardabweichung: {std_implied_price:.2f}")

plt.style.use("dark_background")
background_color = "#262730"
plt.figure(figsize=(6, 6), facecolor="#0E1117")
ax = plt.gca()
ax.set_facecolor(background_color)
plt.hist(implied_prices, bins=20, alpha=0.7, color="blue", edgecolor="white")
plt.title(f"Histogramm der impliziten Aktienkurse im Jahr {valuation_year}", color="white")
plt.xlabel("Implizierter Aktienkurs", color="white")
plt.ylabel("Häufigkeit", color="white")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
st.pyplot(plt)

# Sensitivity Pie Chart
st.subheader("Sensitivitätsanalyse")
variables = ["Umsatzwachstum", "Marge", "TGR", "A&A", "Investition", "WACC", "Steuersatz"]
variances = [
    np.var(growth_rate_dist),
    np.var(margin_dist),
    np.var(tgr_dist),
    np.var(dep_dist),
    np.var(reinv_dist),
    np.var(wacc_dist),
    np.var(tax_rate_dist),
]
total_variance = sum(variances)
percentages = [(var / total_variance) * 100 if total_variance != 0 else 0 for var in variances]

plt.figure(figsize=(6, 6), facecolor="#0E1117")
ax = plt.gca()
ax.set_facecolor(background_color)
custom_colors = ["#FF5733", "#33FF57", "#3357FF", "#FF33A1", "#A1FF33", "#33A1FF", "#FFC133"]
plt.pie(
    percentages,
    labels=variables,
    autopct="%1.0f%%",
    startangle=90,
    colors=custom_colors,
)
plt.title(f"Varianzbeitrag - {valuation_year}. Jahr", color="white")
st.pyplot(plt)

# --------------------------------------------
# Sidebar
# --------------------------------------------
st.sidebar.title(f"Aktie: {ticker_symbol}")
st.sidebar.write(f"Heutiger Aktienkurs: {current_share_price:.2f}")

# Sidebar Classic DCF
st.sidebar.header("Classic DCF")
st.sidebar.write(f"Implizierter Aktienkurs in 5 Jahren: {implied_price_year_5}")
st.sidebar.write(f"Implizierter Aktienkurs in 10 Jahren: {implied_price_year_10}")

# Sidebar Monte Carlo Simulation
st.sidebar.header("Monte Carlo Simulation")
st.sidebar.write(f"Prognose für Jahr {valuation_year}")
st.sidebar.write(f"⌀implizierter Aktienkurs: {mean_implied_price:.2f}")
st.sidebar.write(f"Standardabweichung: {std_implied_price:.2f}")
st.sidebar.write(f"{valuation}", unsafe_allow_html=True)