# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt

st.title("Monte Carlo Simulation - DCF")
with st.expander("Abstract"):
    st.write(
        """
        <div style="text-align: justify;">
        Das Ziel dieser Ausarbeitung besteht darin, Entscheidungsträgern eine bessere Informationsgrundlage für die Entscheidungsfindung im Zusammenhang mit Aktieninvestitionen zu bieten. Häufig liefern klassische Investitionsrechnungsverfahren nicht ausreichend Informationen, um das Risiko und die Rendite einer Investition angemessen zu bewerten. Das vorgestellte Modell soll daher helfen, fundierte Entscheidungen bei der Verwaltung von Finanzanlagen zu treffen.

        
        Zur Durchführung der Analyse kommen zwei Methoden zum Einsatz: Das Discounted Cash Flow (DCF)-Verfahren als eine fundamentale Analyse sowie die Monte-Carlo-Simulation als eine quantitative Analyse. Mithilfe des DCF-Verfahrens wird der innere Wert des Unternehmens ermittelt, woraufhin der innere Aktienwert berechnet wird.

        Die Ergebnisse des DCF-Verfahrens liefern Informationen über die erwartete Rendite ab dem jeweiligen Jahr. Die Monte-Carlo-Simulation hingegen liefert zusätzliche Erkenntnisse über die Wahrscheinlichkeit und Investitionsdauer, ab der die Aktie als über- bzw. unterbewertet angesehen werden kann, indem ganzheitlich Chancen und Risiken betrachtet werden. Darüber hinaus zeigen die Ergebnisse der simulationsbasierten Analyse, dass sich die relevanten Hebel je nach Zeitpunkt verändern. Diese Erkenntnisse helfen den Entscheidungsträgern dabei, fundierte Entscheidungen bei der Verwaltung ihrer Finanzanlagen zu treffen und die Risiken und Chancen im Zusammenhang mit Aktieninvestitionen besser zu verstehen.
        </div>
        """,
        unsafe_allow_html=True
    )
st.header("Schritt 1 - Daten aus yfinance sammeln:")

ticker_symbol = st.text_input("Aktien Ticker-Symbol eingeben (z.B. AAPL, MSFT, MBG.DE, BMW.DE, VOW3.DE ):", "MSFT")


ticker = yf.Ticker(ticker_symbol)

income_statement = ticker.financials.T  
balance_sheet = ticker.balance_sheet.T  
cashflow_statement = ticker.cashflow.T  


try:
    current_share_price = ticker.info["currentPrice"]
except:
    current_share_price = float("nan")

if balance_sheet.empty:
    st.warning("Keine Bilanzdaten (Balance Sheet) vorhanden.")
    number_of_shares = 1
else:
    try:
        latest_date = balance_sheet.index[0]
        number_of_shares = balance_sheet.loc[latest_date, "Ordinary Shares Number"]
    except:
        number_of_shares = 1

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
data = data.sort_index(ascending=True)  

data["Revenue Growth Rate"] = (
    (data["Revenue"] - data["Revenue"].shift(1)) / data["Revenue"].shift(1)
)
data["EBIT Margin"] = data["EBIT"] / data["Revenue"]

st.write("Historische Daten:")
st.dataframe(data)
with st.expander("Erklärung:"):
    st.write(
    """
    <div style="text-align: justify;">
    Im ersten Schritt wird die Grundlage für die Analyse durch das Sammeln historischer Finanzdaten eines Unternehmens geschaffen.
    Über die `yfinance`-Bibliothek werden wichtige finanzielle Kennzahlen, wie Umsatz, EBIT, Steueraufwand, Abschreibungen, 
    Investitionen und Kapitalstruktur, extrahiert. Diese Daten bilden die Basis für die Berechnung von Wachstumstrends und Margen, 
    welche entscheidend für die Discounted-Cashflow-(DCF)-Projektionen sind.

    Die Berechnung des freien Cashflows erfolgt gemäß folgender Formel:
    
    **Umsatz**  
    \* **Marge**  
    = **EBIT**  
    \- **Steueraufwand**  
    = **Nettoüberschuss für weitere Geschäftstätigkeiten**  
    \+ **Abschreibung und Amortisation**  
    \- **Investitionen in das Anlagevermögen**  
    \- **Änderungen des Working Capital**  
    = **Freier Cashflow**

    </div>
    """,
    unsafe_allow_html=True
)



st.header("Schritt 2 - Definiere Annahmen:")
st.write(
    """
    <div style="text-align: justify;">
    Treffe Annahmen, die für die Berechnungen der zukünftigen Cashflows erforderlich sind. 
    
    Hierzu gehören:

    - **Weighted Average Cost of Capital (WACC):** Ein entscheidender Diskontierungssatz, der sowohl das Risiko als auch die erwartete Rendite reflektiert.
    - **Terminal Growth Rate (TGR):** Die angenommene ewige Wachstumsrate für den Restwert des Unternehmens.
    - **Wachstumsraten und Margen:** Durchschnittswerte basierend auf historischen Daten des Unternehmens und der Branche.

    Benutzer haben die Möglichkeit, die Standardannahmen anzupassen, um Szenarien zu simulieren, die entweder optimistische oder konservative Bedingungen widerspiegeln. 
    Dies erlaubt eine flexible Bewertung des Unternehmenswertes.
    </div>
    """,
    unsafe_allow_html=True
)

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



try:
    current_share_price = round(current_share_price, 2)
except:
    current_share_price = float("nan")

implied_price_year_5 = round(projection_df.loc[5, "Implied Share Price"], 2)
implied_price_year_10 = round(projection_df.loc[10, "Implied Share Price"], 2)

st.write(f"Heutiger Aktienkurs: {current_share_price}")
col1, col2 = st.columns(2)
with col1:
    st.write(f"Innerer Aktienwert in 5 Jahren: {implied_price_year_5}")
with col2:
    st.write(f"Innerer Aktienwert in 10 Jahren: {implied_price_year_10}")

with col1:
    if implied_price_year_5 < current_share_price:
        st.write(
            "Bewertung 5-Jahres-Periode:",
            f'<span style="color:red;">***Überbewertet***</span>',
            unsafe_allow_html=True,
        )
    else:
        st.write(
            "Bewertung 5-Jahres-Periode:",
            f'<span style="color:green;">***Unterbewertet***</span>',
            unsafe_allow_html=True,
        )

with col2:
    if implied_price_year_10 < current_share_price:
        st.write(
            "Bewertung 10-Jahres-Periode:",
            f'<span style="color:red;">***Überbewertet***</span>',
            unsafe_allow_html=True,
        )
    else:
        st.write(
            "Bewertung 10-Jahres-Periode:",
            f'<span style="color:green;">***Unterbewertet***</span>',
            unsafe_allow_html=True,
        )

with st.expander("Erklärung:"):
    st.write(
    """
    <div style="text-align: justify;">
    Basierend auf den definierten Annahmen und den historischen Daten werden zukünftige Cashflows über einen Zeitraum von zehn Jahren projiziert. 
    Diese Projektionen umfassen:

    - **Umsatzwachstum:** Prognose des zukünftigen Unternehmensumsatzes.
    - **EBIT und Nettogewinn:** Berechnung unter Berücksichtigung der erwarteten Steuerlast.
    - **Abschreibungen und Investitionen:** Umrechnung auf Free Cashflows.

    Die Berechnung des gesamten Unternehmenswerts erfolgt gemäß der folgenden Formel:

     \[
    GUW = UW + RW = \sum_{t=1}^{n} \frac{FCF_t}{(1 + WACC)^t} + \frac{1}{(1 + WACC)^n} \cdot \left( \frac{FCF_{n+1}}{WACC - g} \right)
    \]

    Dabei steht:
    - \(GUW\): Gesamtunternehmenswert
    - \(UW\): Unternehmenswert durch diskontierte Free Cashflows
    - \(RW\): Restwert des Unternehmens
    - \(FCF_t\): Freier Cashflow im Jahr \(t\)
    - \(WACC\): Gewichtete durchschnittliche Kapitalkosten
    - \(g\): Ewige Wachstumsrate

    Aus dem Gesamtunternehmenswert wird der **innere Aktienwert** berechnet, indem folgende Schritte durchgeführt werden:

    **Gesamtunternehmenswert**  
    \+ **Cash und Beteiligungen**  
    \- **Nettoverschuldung**  
    = **Eigenkapitalwert**  
    \% **Anzahl der Aktien**  
    = **Innerer Aktienwert [€]**
    </div>
    """,
    unsafe_allow_html=True
)




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
valuation_year = cols[1].selectbox("Bewertungsjahr auswählen:", [5, 10], index=1)

with st.expander("Verteilungsparameter anpassen"):
    st.markdown("### Normal-Verteilung (Umsatzwachstum)")
    gmean_col, gstd_col = st.columns(2)
    growth_rate_mean = gmean_col.number_input(
        "Umsatzwachstum - Mean", value=growth_rate_avg, key="growth_rate_mean"
    )
    growth_rate_std = gstd_col.number_input(
        "Umsatzwachstum - Std. Abw.", value=abs(growth_rate_avg) * 0.1, key="growth_rate_std"
    )


    st.markdown("### Lognormal-Verteilung (TGR)")
    tgr_mean_log, tgr_sigma, tgr_max = get_distribution_params_lognormal(
        "TGR",
        default_mean=np.log(tgr),  # log-space mean (since tgr ~ 0.03)
        default_sigma=0.2,         # initial guess
        default_max=tgr * 1.2      # max cap for TGR
    )

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


raw_tgr_dist = np.random.lognormal(mean=tgr_mean_log, sigma=tgr_sigma, size=n_simulations)
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

    fcf_terminal = fcf * (1 + simulated_tgr)
    if (simulated_wacc - simulated_tgr) > 0:
        terminal_value = fcf_terminal / (simulated_wacc - simulated_tgr)
    else:
        terminal_value = fcf_terminal / max(0.01, simulated_wacc - simulated_tgr)

    discounted_terminal_value = terminal_value / ((1 + simulated_wacc) ** valuation_year)

    total_firm_value = total_dcf + discounted_terminal_value + cash_avg - debt_avg
    implied_share_price = total_firm_value / number_of_shares
    implied_prices.append(implied_share_price)

mean_implied_price = np.mean(implied_prices)
std_implied_price = np.std(implied_prices)

st.subheader(f"Prognose für Jahr {valuation_year}")
if mean_implied_price > current_share_price:
    valuation = '<span style="color:green;">***unterbewertet***</span>'
else:
    valuation = '<span style="color:red;">***überbewertet***</span>'

cols = st.columns(2)
cols[0].write(f"Heutiger Aktienkurs: {current_share_price:.2f}")
cols[1].write(f"Bewertung: {valuation}", unsafe_allow_html=True)

cols = st.columns(2)
cols[0].write(f"⌀ innerer Aktienwert: {mean_implied_price:.2f}")
cols[1].write(f"Standardabweichung: {std_implied_price:.2f}")

hist_data = pd.DataFrame({"Innerer Aktienwert": implied_prices})

percentile_10 = np.percentile(implied_prices, 10)
percentile_90 = np.percentile(implied_prices, 90)
current_price_percentile = (np.sum(np.array(implied_prices) < current_share_price) / len(implied_prices)) * 100


hist = alt.Chart(hist_data).mark_bar(opacity=0.7).encode(
    alt.X("Innerer Aktienwert:Q", bin=alt.Bin(maxbins=20), title="Innerer Aktienwert"),
    alt.Y("count():Q", title="Häufigkeit"),
    tooltip=["count()"]
).properties(
    title=f"Histogramm des inneren Aktienwerts für den Investitionszeitraum von {valuation_year} Jahren",
    width=600,
    height=400
)


line_data = pd.DataFrame({
    "Position": [percentile_10, percentile_90, current_share_price],
    "Label": ["10-Perzentile", "90-Perzentile", "Heutiger Preis"],
    "Color": ["yellow", "green", "red"]
})

vertical_lines = alt.Chart(line_data).mark_rule(strokeDash=[5, 5]).encode(
    x=alt.X("Position:Q", title="Innerer Aktienwert"),
    color=alt.Color("Color:N", scale=None, legend=None),  # Custom colors
    tooltip=["Label:N", "Position:Q"]  # Add tooltips for better interactivity
)

legend = alt.Chart(line_data).mark_point(size=100).encode(
    y=alt.Y("Label:N", axis=alt.Axis(title="Legend", labels=True)),
    color=alt.Color("Color:N", scale=None)  # Match custom colors
).properties(width=150)


st.altair_chart(hist + vertical_lines,use_container_width=True)
st.write(f"Der heutige Aktienkurs [in rot] liegt im {current_price_percentile:.2f}-Perzentil des Histogramms.")
with st.expander("Erklärung:"):
    st.write(
    """
    <div style="text-align: justify;">
    Dieser Schritt erweitert das klassische DCF-Verfahren um eine stochastische Analyse. Anstatt feste Werte für Wachstumsraten, Margen oder WACC zu verwenden, 
    werden Wahrscheinlichkeitsverteilungen implementiert, um die Unsicherheit dieser Parameter zu modellieren.

    Die Monte-Carlo-Simulation generiert eine Vielzahl von Szenarien durch zufällige Ziehungen aus den definierten Verteilungen. Hierdurch können die Wahrscheinlichkeiten für verschiedene Ergebnisse berechnet werden, wie:

    - **Unter- oder Überbewertung der Aktie:** Identifizierung von Preispunkten, bei denen eine Investition sinnvoll erscheint.
    - **Risikobewertung:** Ermittlung der Wahrscheinlichkeit von Verlusten oder Gewinnen unter unterschiedlichen Bedingungen.

    Die Kombination der Berechnung des freien Cashflows, des inneren Aktienwerts und der Monte-Carlo-Simulation ermöglicht eine realistischere und differenziertere Analyse, 
    da nicht nur Mittelwerte, sondern auch Extremwerte und Varianzen berücksichtigt werden.
    </div>
    """,
    unsafe_allow_html=True
)
    
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

pie_data = pd.DataFrame({
    "Variable": ["Umsatzwachstum", "Marge", "TGR", "A&A", "Investition", "WACC", "Steuersatz"],
    "Percentage": percentages
})

pie_chart = alt.Chart(pie_data).mark_arc(innerRadius=50).encode(
    theta="Percentage:Q",
    color="Variable:N",
    tooltip=["Variable:N", "Percentage:Q"]
).properties(
    title=f"Varianzbeitrag - {valuation_year}. Jahr",
    width=400,
    height=400
)

st.altair_chart(pie_chart, use_container_width=True)

with st.expander("Erklärung:"):
    st.write(
    """
    <div style="text-align: justify;">
    Die Sensitivitätsanalyse in der Monte-Carlo-Simulation analysiert den Einfluss der Varianz einzelner Eingabevariablen auf die Zielgrößen, wie den inneren Aktienwert. 
    Hierbei wird untersucht, welche Variablen den größten Beitrag zur Streuung der Ergebnisse leisten.

    **Ziele der Sensitivitätsanalyse:**
    - Bestimmung der Schlüsselvariablen, die die höchste Varianz im Ergebnis verursachen.
    - Quantifizierung der Unsicherheiten, die mit den Schwankungen dieser Variablen verbunden sind.
    - Unterstützung bei der Optimierung und Fokussierung auf kritische Parameter zur Risikominderung.

    Im simulationsbasierten Ansatz zeigt die Sensitivitätsanalyse, dass je nach Szenario und Zeitrahmen unterschiedliche Variablen, wie Umsatzwachstum oder WACC, 
    dominierend sein können. Die Varianzbeiträge der Variablen bieten dabei wertvolle Einblicke, um die wichtigsten Hebel für fundierte Entscheidungen zu identifizieren.

    Die Ergebnisse ermöglichen es Entscheidungsträgern, potenzielle Chancen besser zu bewerten und gleichzeitig Risiken gezielt zu steuern, 
    indem die Analyse sowohl kurzfristige als auch langfristige Auswirkungen berücksichtigt.
    </div>
    """,
    unsafe_allow_html=True
)



# Sidebar
st.sidebar.title(f"Aktie: {ticker_symbol}")
st.sidebar.write(f"Heutiger Aktienkurs: {current_share_price:.2f}")
if mean_implied_price < current_share_price:
    st.sidebar.write(
        f"Basierend auf den getroffenen Annahmen und Investitionszeitraum von {valuation_year} Jahren ist die Aktie mit einer Wahrscheinlichkeit von {current_price_percentile:.2f}% {valuation}.",
        unsafe_allow_html=True,
    )
else:
    st.sidebar.write(
        f"Basierend auf den getroffenen Annahmen und Investitionszeitraum von {valuation_year} Jahren ist die Aktie mit einer Wahrscheinlichkeit von {100 - current_price_percentile:.2f}% {valuation}.",
        unsafe_allow_html=True,
    )
st.sidebar.header("Monte Carlo Simulation")
st.sidebar.write(f"Prognose für den Investitionszeitraum von {valuation_year} Jahren:")
st.sidebar.write(f"⌀ Innerer Aktienwert: {mean_implied_price:.2f}")
st.sidebar.write(f"Standardabweichung: {std_implied_price:.2f}")

st.sidebar.header("Klassischer DCF-Verfahren")
st.sidebar.write(f"Innerer Aktienwert (5 Jahren): {implied_price_year_5}")
st.sidebar.write(f"Innerer Aktienwert (10 Jahren): {implied_price_year_10}")









#st.sidebar.write(f"{valuation}", unsafe_allow_html=True)
