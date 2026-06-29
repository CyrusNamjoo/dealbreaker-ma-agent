"""
Financial analysis tools for the FinancialAnalystAgent.

These tools perform calculations and analysis on financial data the agent has
already extracted from SEC EDGAR filings via load_web_page or the SEC EDGAR
MCP server. They do not fetch data themselves — the agent fetches raw filing
text and parses it into structured dicts before calling these tools.

CONTRACT (enforced in every return value):
- Every tool returns data_available: bool.
- If data_available=False, numeric output fields are absent and message
  explains exactly what data is missing and where to find it.
- No financial figure is estimated or inferred; calculations use only the
  values explicitly passed in the arguments.
"""

import math
from typing import Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def _pct(val: float | None, decimals: int = 4) -> float | None:
    return round(val, decimals) if val is not None else None


def _get(record: dict, key: str) -> float | None:
    val = record.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Tool 1 — calculate_financial_ratios
# ---------------------------------------------------------------------------

def calculate_financial_ratios(financials: dict) -> dict:
    """Compute 20+ financial ratios from structured income statement, balance
    sheet, and cash flow data extracted from public SEC EDGAR filings.

    Args:
        financials: dict with keys:
          periods (list[str]): period labels, most-recent first
            e.g. ["FY2023", "FY2022", "FY2021"]
          income_statements (list[dict]): one dict per period, keys:
            revenue, gross_profit, ebitda, ebit, interest_expense,
            net_income, cogs, sga_expense, depreciation (all float, USD)
          balance_sheets (list[dict]): one dict per period, keys:
            cash, current_assets, inventory, receivables, ppe,
            total_assets, current_liabilities, payables, total_debt,
            total_liabilities, equity (all float, USD)
          cash_flow_statements (list[dict]): one dict per period, keys:
            operating_cf, capex, free_cash_flow (float, USD;
            capex is negative by convention)

    Returns:
        dict with data_available (bool), message (str when False),
        periods_analysed, ratios {liquidity, leverage, profitability,
        efficiency, coverage}, cagr, source (str)
    """
    income = financials.get("income_statements", [])
    balance = financials.get("balance_sheets", [])
    cashflow = financials.get("cash_flow_statements", [])
    periods = financials.get("periods", [])

    if not income or not balance:
        return {
            "data_available": False,
            "message": (
                "Both income_statements and balance_sheets are required. "
                "Extract these from SEC 10-K or 10-Q filings using load_web_page "
                "with a URL from search_company_filings, then pass the structured "
                "figures here. Do not estimate any values."
            ),
            "source": "https://data.sec.gov",
        }

    i0 = income[0]
    b0 = balance[0]
    cf0 = cashflow[0] if cashflow else {}

    rev = _get(i0, "revenue")
    gross = _get(i0, "gross_profit")
    ebitda = _get(i0, "ebitda")
    ebit = _get(i0, "ebit")
    int_exp = _get(i0, "interest_expense")
    net_inc = _get(i0, "net_income")

    cash = _get(b0, "cash")
    curr_assets = _get(b0, "current_assets")
    inventory = _get(b0, "inventory") or 0.0
    receivables = _get(b0, "receivables")
    total_assets = _get(b0, "total_assets")
    curr_liab = _get(b0, "current_liabilities")
    payables = _get(b0, "payables")
    total_debt = _get(b0, "total_debt")
    total_liab = _get(b0, "total_liabilities")
    equity = _get(b0, "equity")

    op_cf = _get(cf0, "operating_cf")
    capex = _get(cf0, "capex")
    fcf = _get(cf0, "free_cash_flow")
    if fcf is None and op_cf is not None and capex is not None:
        fcf = op_cf + capex  # capex is negative in standard statements

    ratios: dict[str, Any] = {}

    # Liquidity
    liquidity: dict[str, Any] = {}
    cr = _safe_div(curr_assets, curr_liab)
    if cr is not None:
        liquidity["current_ratio"] = round(cr, 2)
    qr = _safe_div((curr_assets or 0) - inventory, curr_liab)
    if qr is not None:
        liquidity["quick_ratio"] = round(qr, 2)
    cashr = _safe_div(cash, curr_liab)
    if cashr is not None:
        liquidity["cash_ratio"] = round(cashr, 2)
    ratios["liquidity"] = liquidity

    # Leverage
    leverage: dict[str, Any] = {}
    de = _safe_div(total_debt, equity)
    if de is not None:
        leverage["debt_to_equity"] = round(de, 2)
    da = _safe_div(total_debt, total_assets)
    if da is not None:
        leverage["debt_to_assets"] = round(da, 2)
    ic = _safe_div(ebitda, int_exp)
    if ic is not None:
        leverage["interest_coverage"] = round(ic, 2)
    if total_debt is not None and cash is not None:
        net_debt = total_debt - cash
        leverage["net_debt_usd"] = net_debt
        nd_ebitda = _safe_div(net_debt, ebitda)
        if nd_ebitda is not None:
            leverage["net_debt_to_ebitda"] = round(nd_ebitda, 2)
    tle = _safe_div(total_liab, equity)
    if tle is not None:
        leverage["total_liabilities_to_equity"] = round(tle, 2)
    ratios["leverage"] = leverage

    # Profitability
    profitability: dict[str, Any] = {}
    for label, num in [
        ("gross_margin", gross),
        ("ebitda_margin", ebitda),
        ("ebit_margin", ebit),
        ("net_profit_margin", net_inc),
    ]:
        val = _safe_div(num, rev)
        if val is not None:
            profitability[label] = _pct(val)
    roa = _safe_div(net_inc, total_assets)
    if roa is not None:
        profitability["return_on_assets"] = _pct(roa)
    roe = _safe_div(net_inc, equity)
    if roe is not None:
        profitability["return_on_equity"] = _pct(roe)
    ocf_margin = _safe_div(op_cf, rev)
    if ocf_margin is not None:
        profitability["operating_cf_margin"] = _pct(ocf_margin)
    fcf_margin = _safe_div(fcf, rev)
    if fcf_margin is not None:
        profitability["fcf_margin"] = _pct(fcf_margin)
    ratios["profitability"] = profitability

    # Efficiency
    efficiency: dict[str, Any] = {}
    at = _safe_div(rev, total_assets)
    if at is not None:
        efficiency["asset_turnover"] = round(at, 2)
    rec_days = _safe_div(receivables, rev)
    if rec_days is not None:
        efficiency["receivables_days"] = round(rec_days * 365, 1)
    pay_days = _safe_div(payables, rev)
    if pay_days is not None:
        efficiency["payables_days"] = round(pay_days * 365, 1)
    inv_days = _safe_div(inventory, rev)
    if inv_days is not None:
        efficiency["inventory_days"] = round(inv_days * 365, 1)
    if all(k in efficiency for k in ("receivables_days", "payables_days", "inventory_days")):
        ccc = (
            efficiency["receivables_days"]
            + efficiency["inventory_days"]
            - efficiency["payables_days"]
        )
        efficiency["cash_conversion_cycle"] = round(ccc, 1)
    if capex is not None and rev:
        efficiency["capex_pct_revenue"] = _pct(abs(capex) / rev)
    ratios["efficiency"] = efficiency

    # Coverage
    coverage: dict[str, Any] = {}
    dscr = _safe_div(op_cf, total_debt)
    if dscr is not None:
        coverage["debt_service_cf_ratio"] = round(dscr, 2)
    fcf_td = _safe_div(fcf, total_debt)
    if fcf_td is not None:
        coverage["fcf_to_debt"] = round(fcf_td, 2)
    ratios["coverage"] = coverage

    # CAGR (revenue and EBITDA)
    cagr: dict[str, Any] = {}
    n = len(income) - 1
    if n >= 1:
        rev_now = _get(income[0], "revenue")
        rev_old = _get(income[-1], "revenue")
        if rev_now and rev_old and rev_old > 0:
            cagr["revenue"] = {
                "n_years": n,
                "value": round((rev_now / rev_old) ** (1 / n) - 1, 4),
            }
        eb_now = _get(income[0], "ebitda")
        eb_old = _get(income[-1], "ebitda")
        if eb_now and eb_old and eb_old > 0:
            cagr["ebitda"] = {
                "n_years": n,
                "value": round((eb_now / eb_old) ** (1 / n) - 1, 4),
            }

    return {
        "data_available": True,
        "periods_analysed": periods or [f"period_{i+1}" for i in range(len(income))],
        "ratios": ratios,
        "cagr": cagr,
        "notes": (
            "Ratios are calculated solely from the figures you provided. "
            "Verify every input figure against the SEC filing line items."
        ),
        "source": "https://data.sec.gov",
    }


# ---------------------------------------------------------------------------
# Tool 2 — build_dcf_model
# ---------------------------------------------------------------------------

def build_dcf_model(
    fcf_history: list[float],
    wacc: float,
    terminal_growth: float,
) -> dict:
    """Build a bear/base/bull DCF valuation from historical free cash flow.

    Projects FCF five years forward under three growth assumptions, then
    applies a Gordon Growth terminal value. All inputs must come from
    verified SEC filing data — never estimated.

    Args:
        fcf_history: list[float] — historical FCF in USD, most-recent first.
          Minimum 1 value required; 3+ preferred for reliable growth estimate.
          Example: [20_000_000, 18_000_000, 15_000_000]
        wacc: float — weighted average cost of capital as a decimal (e.g. 0.10).
          Must be > terminal_growth.
        terminal_growth: float — long-run perpetuity growth rate as a decimal
          (e.g. 0.025 for 2.5%). Must be < wacc and reasonable (< 0.05 for
          most companies).

    Returns:
        dict with data_available (bool), scenarios {bear, base, bull},
        sensitivity_table, projection_years, assumptions, source
    """
    if not fcf_history:
        return {
            "data_available": False,
            "message": (
                "fcf_history is empty. Extract free cash flow from the cash flow "
                "statement in SEC 10-K filings (Operating CF minus Capex). "
                "Provide at least 1 year; 3 years preferred."
            ),
            "source": "https://data.sec.gov",
        }
    if wacc <= 0:
        return {
            "data_available": False,
            "message": "wacc must be a positive decimal (e.g. 0.10 for 10%).",
            "source": "https://data.sec.gov",
        }
    if terminal_growth >= wacc:
        return {
            "data_available": False,
            "message": (
                f"terminal_growth ({terminal_growth:.3f}) must be less than "
                f"wacc ({wacc:.3f}) for the Gordon Growth model to be valid."
            ),
            "source": "https://data.sec.gov",
        }
    if terminal_growth < 0:
        return {
            "data_available": False,
            "message": "terminal_growth must be non-negative.",
            "source": "https://data.sec.gov",
        }

    base_fcf = fcf_history[0]

    # Derive historical growth rate if enough data is available.
    if len(fcf_history) >= 2 and fcf_history[-1] > 0:
        n = len(fcf_history) - 1
        hist_growth = (fcf_history[0] / fcf_history[-1]) ** (1 / n) - 1
    else:
        # Single data point — use terminal_growth as the base growth rate.
        hist_growth = terminal_growth

    # Scenario growth rates applied to FCF projections over 5 years.
    scenarios = {
        "bear": max(terminal_growth - 0.02, -0.10),   # floor at -10%
        "base": hist_growth,
        "bull": min(hist_growth * 1.5, 0.40),          # cap at 40%
    }

    projection_years = 5
    results: dict[str, Any] = {}

    for name, growth in scenarios.items():
        projected_fcfs: list[float] = []
        fcf = base_fcf
        for _ in range(projection_years):
            fcf = fcf * (1 + growth)
            projected_fcfs.append(round(fcf, 0))

        # Terminal value (Gordon Growth on year-5 FCF).
        terminal_fcf = projected_fcfs[-1] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (wacc - terminal_growth)

        # Discount each FCF and the terminal value to present.
        pv_fcfs: list[float] = []
        for yr, fcf_yr in enumerate(projected_fcfs, start=1):
            pv_fcfs.append(fcf_yr / (1 + wacc) ** yr)

        pv_terminal = terminal_value / (1 + wacc) ** projection_years
        enterprise_value = sum(pv_fcfs) + pv_terminal

        results[name] = {
            "growth_rate_applied": round(growth, 4),
            "projected_fcfs_usd": projected_fcfs,
            "pv_of_fcfs_usd": round(sum(pv_fcfs), 0),
            "terminal_value_usd": round(terminal_value, 0),
            "pv_of_terminal_value_usd": round(pv_terminal, 0),
            "enterprise_value_usd": round(enterprise_value, 0),
        }

    # Sensitivity table: EV under different WACC × terminal growth combinations.
    wacc_range = [wacc - 0.02, wacc, wacc + 0.02]
    tg_range = [terminal_growth - 0.005, terminal_growth, terminal_growth + 0.005]
    sensitivity: list[dict] = []
    for tg in tg_range:
        if tg < 0 or tg >= wacc - 0.02:
            continue
        row: dict[str, Any] = {"terminal_growth": round(tg, 4)}
        for w in wacc_range:
            if tg >= w:
                row[f"wacc_{round(w * 100, 1)}pct"] = "N/A (tg >= wacc)"
                continue
            tv = (base_fcf * (1 + hist_growth) ** projection_years * (1 + tg)) / (w - tg)
            pv_tv = tv / (1 + w) ** projection_years
            pv_f = sum(
                base_fcf * (1 + hist_growth) ** yr / (1 + w) ** yr
                for yr in range(1, projection_years + 1)
            )
            row[f"wacc_{round(w * 100, 1)}pct"] = round(pv_f + pv_tv, 0)
        sensitivity.append(row)

    return {
        "data_available": True,
        "projection_years": projection_years,
        "base_fcf_usd": base_fcf,
        "historical_fcf_inputs_usd": fcf_history,
        "scenarios": results,
        "sensitivity_table": sensitivity,
        "assumptions": {
            "wacc": wacc,
            "terminal_growth_rate": terminal_growth,
            "model": "Gordon Growth terminal value; 5-year explicit projection",
            "note": (
                "Enterprise value only. Subtract net debt and add cash to derive "
                "equity value. All FCF inputs must be sourced from verified SEC filings."
            ),
        },
        "source": "https://data.sec.gov",
    }


# ---------------------------------------------------------------------------
# Tool 3 — analyze_cash_flow_quality
# ---------------------------------------------------------------------------

def analyze_cash_flow_quality(cash_flow_statements: list[dict]) -> dict:
    """Detect cash flow quality issues and working capital manipulation signals.

    Analyses the relationship between operating cash flow and net income,
    working capital movements, and year-over-year trends to surface
    earnings quality concerns. Signals are flagged — not confirmed — and
    require analyst interpretation of the underlying SEC filing footnotes.

    Args:
        cash_flow_statements: list[dict] — one dict per fiscal year,
          most-recent first. Each dict should contain:
            period (str): e.g. "FY2023"
            operating_cf (float): cash from operations
            net_income (float): net income from income statement
            capex (float): capital expenditure (negative)
            revenue (float): total revenue (for ratio analysis)
            change_in_receivables (float): negative = receivables increased
            change_in_inventory (float): negative = inventory increased
            change_in_payables (float): positive = payables increased
            depreciation (float): D&A added back
            stock_based_comp (float): SBC added back (optional)
            free_cash_flow (float): optional; computed from operating_cf + capex

    Returns:
        dict with data_available, quality_signals (list of labelled signals),
        period_analysis (per-year metrics), warnings (list[str]),
        interpretation_guidance, source
    """
    if not cash_flow_statements:
        return {
            "data_available": False,
            "message": (
                "cash_flow_statements is empty. Extract cash flow data from "
                "the Statement of Cash Flows in SEC 10-K or 10-Q filings."
            ),
            "source": "https://data.sec.gov",
        }

    period_analysis: list[dict] = []
    quality_signals: list[dict] = []
    warnings: list[str] = []

    for stmt in cash_flow_statements:
        period = stmt.get("period", "unknown")
        op_cf = _get(stmt, "operating_cf")
        net_inc = _get(stmt, "net_income")
        capex = _get(stmt, "capex")
        rev = _get(stmt, "revenue")
        d_rec = _get(stmt, "change_in_receivables")
        d_inv = _get(stmt, "change_in_inventory")
        d_pay = _get(stmt, "change_in_payables")
        depr = _get(stmt, "depreciation")
        sbc = _get(stmt, "stock_based_comp") or 0.0

        fcf = _get(stmt, "free_cash_flow")
        if fcf is None and op_cf is not None and capex is not None:
            fcf = op_cf + capex

        row: dict[str, Any] = {"period": period}

        # Earnings quality ratio: OCF / Net Income (>1.0 is healthy).
        eqr = _safe_div(op_cf, net_inc)
        if eqr is not None:
            row["earnings_quality_ratio"] = round(eqr, 2)
            if eqr < 0.8:
                quality_signals.append({
                    "period": period,
                    "signal": "LOW_EARNINGS_QUALITY",
                    "detail": (
                        f"OCF/Net Income = {eqr:.2f} (below 0.80). Net income materially "
                        "exceeds operating cash flow — review accruals and non-cash items."
                    ),
                    "severity": "HIGH" if eqr < 0.5 else "MEDIUM",
                })
            elif eqr < 1.0:
                quality_signals.append({
                    "period": period,
                    "signal": "EARNINGS_QUALITY_WATCH",
                    "detail": (
                        f"OCF/Net Income = {eqr:.2f}. Slightly below 1.0 — "
                        "monitor for trend."
                    ),
                    "severity": "LOW",
                })

        # Non-cash add-backs as % of OCF.
        if depr is not None and op_cf and op_cf > 0:
            non_cash_pct = (depr + sbc) / op_cf
            row["non_cash_pct_of_ocf"] = round(non_cash_pct, 4)
            if non_cash_pct > 0.5:
                quality_signals.append({
                    "period": period,
                    "signal": "HIGH_NON_CASH_ADDBACKS",
                    "detail": (
                        f"Non-cash items (D&A + SBC) = {non_cash_pct:.1%} of OCF. "
                        "High reliance on non-cash add-backs inflates OCF."
                    ),
                    "severity": "MEDIUM",
                })

        # Receivables growth vs revenue growth signal.
        if d_rec is not None and rev and rev > 0:
            rec_growth_pct = abs(d_rec) / rev if d_rec < 0 else 0.0
            row["receivables_increase_pct_revenue"] = round(rec_growth_pct, 4)
            if d_rec < 0 and rec_growth_pct > 0.10:
                quality_signals.append({
                    "period": period,
                    "signal": "RECEIVABLES_SPIKE",
                    "detail": (
                        f"Receivables increased by {rec_growth_pct:.1%} of revenue. "
                        "Rapid receivables growth relative to revenue can indicate "
                        "aggressive recognition or channel stuffing. Verify DSO trend."
                    ),
                    "severity": "HIGH" if rec_growth_pct > 0.20 else "MEDIUM",
                })

        # Inventory build.
        if d_inv is not None and rev and rev > 0 and d_inv < 0:
            inv_build_pct = abs(d_inv) / rev
            row["inventory_build_pct_revenue"] = round(inv_build_pct, 4)
            if inv_build_pct > 0.08:
                quality_signals.append({
                    "period": period,
                    "signal": "INVENTORY_BUILD",
                    "detail": (
                        f"Inventory increased by {inv_build_pct:.1%} of revenue. "
                        "Investigate for demand slowdown or write-off risk."
                    ),
                    "severity": "MEDIUM",
                })

        # Payables stretch.
        if d_pay is not None and rev and rev > 0 and d_pay > 0:
            pay_stretch_pct = d_pay / rev
            row["payables_stretch_pct_revenue"] = round(pay_stretch_pct, 4)
            if pay_stretch_pct > 0.05:
                quality_signals.append({
                    "period": period,
                    "signal": "PAYABLES_STRETCH",
                    "detail": (
                        f"Accounts payable grew by {pay_stretch_pct:.1%} of revenue. "
                        "Extended payables artificially inflate OCF in the short term."
                    ),
                    "severity": "LOW",
                })

        if fcf is not None:
            row["free_cash_flow_usd"] = round(fcf, 0)
        if op_cf is not None:
            row["operating_cf_usd"] = round(op_cf, 0)
        if capex is not None:
            row["capex_usd"] = round(capex, 0)

        period_analysis.append(row)

    # Multi-year: check if OCF/Net Income has been consistently < 1 for 2+ years.
    eqrs = [p.get("earnings_quality_ratio") for p in period_analysis if "earnings_quality_ratio" in p]
    if len(eqrs) >= 2 and all(r < 1.0 for r in eqrs):
        warnings.append(
            "OCF/Net Income has been below 1.0 for all analysed periods — "
            "sustained earnings quality concern. Review accruals and revenue "
            "recognition policies in filing footnotes."
        )

    return {
        "data_available": True,
        "periods_analysed": len(cash_flow_statements),
        "quality_signals": quality_signals,
        "period_analysis": period_analysis,
        "warnings": warnings,
        "interpretation_guidance": (
            "Signals indicate areas requiring deeper review of SEC filing footnotes. "
            "A single signal in one year is not conclusive — look for multi-year trends. "
            "Always cite the specific filing and line item when escalating a finding."
        ),
        "source": "https://data.sec.gov",
    }


# ---------------------------------------------------------------------------
# Tool 4 — detect_accounting_anomalies
# ---------------------------------------------------------------------------

def detect_accounting_anomalies(financials_3yr: list[dict]) -> dict:
    """Run Beneish M-Score and Sloan accruals analysis on multi-year financials.

    The Beneish M-Score (Beneish 1999) uses 8 financial ratios to detect
    earnings manipulation. A score above -1.78 suggests possible manipulation.
    The Sloan accruals ratio detects whether earnings are driven by accruals
    rather than cash (high accruals predict future earnings reversals).

    Neither model is definitive — treat signals as hypotheses to investigate
    in SEC filing footnotes, not as confirmed findings.

    Args:
        financials_3yr: list[dict] — 2 or 3 years of financials, most-recent
          first. Each dict requires:
            period (str), revenue (float), receivables (float), cogs (float),
            current_assets (float), ppe (float, net PP&E),
            total_assets (float), depreciation (float), sga_expense (float),
            total_debt (float), current_liabilities (float),
            net_income (float), operating_cf (float)
          Optional: gross_profit (float; derived as revenue - cogs if absent)

    Returns:
        dict with data_available, beneish_m_score (dict with score and
        component indices), sloan_accruals_ratio (per year), anomaly_signals
        (list), interpretation, source
    """
    if len(financials_3yr) < 2:
        return {
            "data_available": False,
            "message": (
                "At least 2 years of financial data are required to compute "
                "year-over-year indices (DSRI, GMI, SGI, etc.). "
                "Extract data from consecutive 10-K filings on SEC EDGAR."
            ),
            "source": "https://data.sec.gov",
        }

    def g(rec: dict, key: str) -> float | None:
        return _get(rec, key)

    t = financials_3yr[0]   # most recent year (t)
    t1 = financials_3yr[1]  # prior year (t-1)

    # Derived gross profit if not supplied.
    def gross_profit(rec: dict) -> float | None:
        gp = g(rec, "gross_profit")
        if gp is not None:
            return gp
        rev = g(rec, "revenue")
        cogs = g(rec, "cogs")
        if rev is not None and cogs is not None:
            return rev - cogs
        return None

    anomaly_signals: list[dict] = []
    beneish_components: dict[str, Any] = {}
    missing_vars: list[str] = []

    # ----- DSRI: Days' Sales in Receivables Index -----
    # (Rec_t / Rev_t) / (Rec_{t-1} / Rev_{t-1})
    rec_t, rev_t = g(t, "receivables"), g(t, "revenue")
    rec_t1, rev_t1 = g(t1, "receivables"), g(t1, "revenue")
    dsri = None
    if all(v is not None and v > 0 for v in [rec_t, rev_t, rec_t1, rev_t1]):
        dsri = (rec_t / rev_t) / (rec_t1 / rev_t1)
        beneish_components["DSRI"] = round(dsri, 4)
        if dsri > 1.465:
            anomaly_signals.append({
                "index": "DSRI",
                "value": round(dsri, 3),
                "threshold": 1.465,
                "signal": "Receivables growing faster than sales — possible channel stuffing or lenient credit terms.",
            })
    else:
        missing_vars.append("DSRI (needs receivables, revenue for t and t-1)")

    # ----- GMI: Gross Margin Index -----
    # GM_{t-1} / GM_t   (>1 means deteriorating margin)
    gm_t = _safe_div(gross_profit(t), rev_t)
    gm_t1 = _safe_div(gross_profit(t1), rev_t1)
    gmi = None
    if gm_t and gm_t1:
        gmi = gm_t1 / gm_t
        beneish_components["GMI"] = round(gmi, 4)
        if gmi > 1.193:
            anomaly_signals.append({
                "index": "GMI",
                "value": round(gmi, 3),
                "threshold": 1.193,
                "signal": "Gross margin deteriorating year-over-year.",
            })
    else:
        missing_vars.append("GMI (needs gross_profit or cogs + revenue for t and t-1)")

    # ----- AQI: Asset Quality Index -----
    # (1 - (CA_t + PPE_t) / TA_t) / (1 - (CA_{t-1} + PPE_{t-1}) / TA_{t-1})
    ca_t, ppe_t, ta_t = g(t, "current_assets"), g(t, "ppe"), g(t, "total_assets")
    ca_t1, ppe_t1, ta_t1 = g(t1, "current_assets"), g(t1, "ppe"), g(t1, "total_assets")
    aqi = None
    if all(v is not None and v > 0 for v in [ca_t, ppe_t, ta_t, ca_t1, ppe_t1, ta_t1]):
        aqi_t = 1 - (ca_t + ppe_t) / ta_t
        aqi_t1 = 1 - (ca_t1 + ppe_t1) / ta_t1
        if aqi_t1 != 0:
            aqi = aqi_t / aqi_t1
            beneish_components["AQI"] = round(aqi, 4)
            if aqi > 1.254:
                anomaly_signals.append({
                    "index": "AQI",
                    "value": round(aqi, 3),
                    "threshold": 1.254,
                    "signal": "Increasing proportion of intangible or off-balance-sheet assets.",
                })
    else:
        missing_vars.append("AQI (needs current_assets, ppe, total_assets for t and t-1)")

    # ----- SGI: Sales Growth Index -----
    # Rev_t / Rev_{t-1}
    sgi = None
    if rev_t and rev_t1 and rev_t1 > 0:
        sgi = rev_t / rev_t1
        beneish_components["SGI"] = round(sgi, 4)
        if sgi > 1.607:
            anomaly_signals.append({
                "index": "SGI",
                "value": round(sgi, 3),
                "threshold": 1.607,
                "signal": "Very high sales growth — elevated manipulation risk in high-growth firms.",
            })

    # ----- DEPI: Depreciation Index -----
    # (Depr_{t-1} / (PPE_{t-1} + Depr_{t-1})) / (Depr_t / (PPE_t + Depr_t))
    depr_t, depr_t1 = g(t, "depreciation"), g(t1, "depreciation")
    depi = None
    if all(v is not None and v > 0 for v in [depr_t, ppe_t, depr_t1, ppe_t1]):
        rate_t = depr_t / (ppe_t + depr_t)
        rate_t1 = depr_t1 / (ppe_t1 + depr_t1)
        if rate_t > 0:
            depi = rate_t1 / rate_t
            beneish_components["DEPI"] = round(depi, 4)
            if depi > 1.077:
                anomaly_signals.append({
                    "index": "DEPI",
                    "value": round(depi, 3),
                    "threshold": 1.077,
                    "signal": "Declining depreciation rate — assets may be understated or useful-life assumptions extended.",
                })
    else:
        missing_vars.append("DEPI (needs depreciation, ppe for t and t-1)")

    # ----- SGAI: SG&A Index -----
    # (SGA_t / Rev_t) / (SGA_{t-1} / Rev_{t-1})
    sga_t, sga_t1 = g(t, "sga_expense"), g(t1, "sga_expense")
    sgai = None
    if all(v is not None for v in [sga_t, sga_t1]) and rev_t and rev_t1 and rev_t1 > 0:
        sgai = (sga_t / rev_t) / (sga_t1 / rev_t1)
        beneish_components["SGAI"] = round(sgai, 4)
        if sgai > 1.041:
            anomaly_signals.append({
                "index": "SGAI",
                "value": round(sgai, 3),
                "threshold": 1.041,
                "signal": "SG&A growing faster than revenue — operational leverage decreasing.",
            })
    else:
        missing_vars.append("SGAI (needs sga_expense, revenue for t and t-1)")

    # ----- LVGI: Leverage Index -----
    # ((LTD_t + CL_t) / TA_t) / ((LTD_{t-1} + CL_{t-1}) / TA_{t-1})
    td_t, cl_t = g(t, "total_debt"), g(t, "current_liabilities")
    td_t1, cl_t1 = g(t1, "total_debt"), g(t1, "current_liabilities")
    lvgi = None
    if all(v is not None for v in [td_t, cl_t, ta_t, td_t1, cl_t1, ta_t1]) and ta_t and ta_t1:
        lev_t = (td_t + cl_t) / ta_t
        lev_t1 = (td_t1 + cl_t1) / ta_t1
        if lev_t1 > 0:
            lvgi = lev_t / lev_t1
            beneish_components["LVGI"] = round(lvgi, 4)
            if lvgi > 1.111:
                anomaly_signals.append({
                    "index": "LVGI",
                    "value": round(lvgi, 3),
                    "threshold": 1.111,
                    "signal": "Leverage increasing — company taking on more debt relative to assets.",
                })
    else:
        missing_vars.append("LVGI (needs total_debt, current_liabilities, total_assets for t and t-1)")

    # ----- TATA: Total Accruals to Total Assets -----
    # (Net Income_t - OCF_t) / Total Assets_t
    ni_t, ocf_t = g(t, "net_income"), g(t, "operating_cf")
    tata = None
    if ni_t is not None and ocf_t is not None and ta_t:
        tata = (ni_t - ocf_t) / ta_t
        beneish_components["TATA"] = round(tata, 4)
        if tata > 0.031:
            anomaly_signals.append({
                "index": "TATA",
                "value": round(tata, 4),
                "threshold": 0.031,
                "signal": "High accruals relative to assets — earnings driven by accounting entries, not cash.",
            })
    else:
        missing_vars.append("TATA (needs net_income, operating_cf, total_assets for year t)")

    # ----- Beneish M-Score -----
    m_score_result: dict[str, Any] = {"components": beneish_components}
    if all(v is not None for v in [dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata]):
        m = (
            -4.84
            + 0.920 * dsri
            + 0.528 * gmi
            + 0.404 * aqi
            + 0.892 * sgi
            + 0.115 * depi
            - 0.172 * sgai
            + 4.679 * tata
            - 0.327 * lvgi
        )
        m_score_result["score"] = round(m, 3)
        if m > -1.78:
            m_score_result["interpretation"] = "LIKELY_MANIPULATOR (M > -1.78)"
        elif m > -2.22:
            m_score_result["interpretation"] = "GREY_ZONE (-2.22 < M <= -1.78)"
        else:
            m_score_result["interpretation"] = "UNLIKELY_MANIPULATOR (M <= -2.22)"
        m_score_result["threshold"] = {
            "manipulator": "-1.78",
            "grey_zone_lower": "-2.22",
            "citation": "Beneish (1999) — 'The Detection of Earnings Manipulation'",
        }
    else:
        m_score_result["score"] = None
        m_score_result["interpretation"] = (
            f"INCOMPLETE — M-Score not computed. Missing inputs: {'; '.join(missing_vars)}"
        )

    # ----- Sloan Accruals Ratio (per year available) -----
    sloan_ratios: list[dict] = []
    for yr in financials_3yr:
        period = yr.get("period", "unknown")
        ni = _get(yr, "net_income")
        ocf = _get(yr, "operating_cf")
        ta = _get(yr, "total_assets")
        inv_cf = _get(yr, "investing_cf")  # optional
        if ni is not None and ocf is not None and ta and ta > 0:
            if inv_cf is not None:
                sloan = (ni - ocf - inv_cf) / ta
            else:
                sloan = (ni - ocf) / ta
            sloan_ratios.append({
                "period": period,
                "sloan_accruals_ratio": round(sloan, 4),
                "flag": abs(sloan) > 0.10,
                "note": (
                    "Abs value > 0.10 is high. Positive = earnings exceed cash; "
                    "negative = cash exceeds reported earnings."
                ),
            })

    return {
        "data_available": True,
        "beneish_m_score": m_score_result,
        "sloan_accruals_ratios": sloan_ratios,
        "anomaly_signals": anomaly_signals,
        "missing_inputs": missing_vars if missing_vars else None,
        "interpretation": (
            "These are statistical indicators, not conclusions. Each signal requires "
            "forensic review of the specific SEC filing disclosures (footnotes, MD&A, "
            "revenue recognition policy). Do not cite these scores as proof of fraud."
        ),
        "source": "https://data.sec.gov",
    }


# ---------------------------------------------------------------------------
# Tool 5 — compare_industry_benchmarks
# ---------------------------------------------------------------------------

# Benchmark medians by sector — Damodaran Online (NYU Stern), January 2024.
# Source: https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/margin.html
# and related files (capital structure, profitability, efficiency by industry).
# Values represent sector medians across public US companies.
_DAMODARAN_BENCHMARKS: dict[str, dict[str, float]] = {
    "saas": {
        "ebitda_margin": 0.08,
        "gross_margin": 0.70,
        "net_profit_margin": 0.02,
        "revenue_cagr": 0.18,
        "return_on_equity": 0.07,
        "net_debt_to_ebitda": -1.0,   # typically net cash
        "current_ratio": 2.3,
        "debt_to_equity": 0.15,
        "fcf_margin": 0.07,
    },
    "software": {
        "ebitda_margin": 0.12,
        "gross_margin": 0.65,
        "net_profit_margin": 0.06,
        "revenue_cagr": 0.10,
        "return_on_equity": 0.12,
        "net_debt_to_ebitda": 0.5,
        "current_ratio": 2.0,
        "debt_to_equity": 0.25,
        "fcf_margin": 0.10,
    },
    "healthcare": {
        "ebitda_margin": 0.15,
        "gross_margin": 0.52,
        "net_profit_margin": 0.07,
        "revenue_cagr": 0.07,
        "return_on_equity": 0.10,
        "net_debt_to_ebitda": 1.5,
        "current_ratio": 1.7,
        "debt_to_equity": 0.45,
        "fcf_margin": 0.08,
    },
    "pharmaceuticals": {
        "ebitda_margin": 0.22,
        "gross_margin": 0.65,
        "net_profit_margin": 0.12,
        "revenue_cagr": 0.05,
        "return_on_equity": 0.16,
        "net_debt_to_ebitda": 1.0,
        "current_ratio": 2.0,
        "debt_to_equity": 0.35,
        "fcf_margin": 0.14,
    },
    "manufacturing": {
        "ebitda_margin": 0.12,
        "gross_margin": 0.28,
        "net_profit_margin": 0.06,
        "revenue_cagr": 0.04,
        "return_on_equity": 0.10,
        "net_debt_to_ebitda": 2.0,
        "current_ratio": 1.6,
        "debt_to_equity": 0.55,
        "fcf_margin": 0.05,
    },
    "retail": {
        "ebitda_margin": 0.08,
        "gross_margin": 0.32,
        "net_profit_margin": 0.03,
        "revenue_cagr": 0.05,
        "return_on_equity": 0.15,
        "net_debt_to_ebitda": 2.5,
        "current_ratio": 1.3,
        "debt_to_equity": 0.80,
        "fcf_margin": 0.03,
    },
    "financial_services": {
        "ebitda_margin": None,  # not meaningful for financials
        "gross_margin": None,
        "net_profit_margin": 0.18,
        "revenue_cagr": 0.06,
        "return_on_equity": 0.12,
        "net_debt_to_ebitda": None,
        "current_ratio": None,
        "debt_to_equity": 3.0,   # leverage is normal in banking
        "fcf_margin": 0.15,
    },
    "energy": {
        "ebitda_margin": 0.28,
        "gross_margin": 0.40,
        "net_profit_margin": 0.08,
        "revenue_cagr": 0.03,
        "return_on_equity": 0.08,
        "net_debt_to_ebitda": 2.5,
        "current_ratio": 1.2,
        "debt_to_equity": 0.60,
        "fcf_margin": 0.07,
    },
    "real_estate": {
        "ebitda_margin": 0.45,
        "gross_margin": 0.55,
        "net_profit_margin": 0.18,
        "revenue_cagr": 0.04,
        "return_on_equity": 0.07,
        "net_debt_to_ebitda": 6.0,
        "current_ratio": 0.9,
        "debt_to_equity": 1.20,
        "fcf_margin": 0.12,
    },
    "consumer_goods": {
        "ebitda_margin": 0.14,
        "gross_margin": 0.42,
        "net_profit_margin": 0.08,
        "revenue_cagr": 0.04,
        "return_on_equity": 0.14,
        "net_debt_to_ebitda": 1.8,
        "current_ratio": 1.4,
        "debt_to_equity": 0.50,
        "fcf_margin": 0.07,
    },
    "telecommunications": {
        "ebitda_margin": 0.32,
        "gross_margin": 0.55,
        "net_profit_margin": 0.05,
        "revenue_cagr": 0.02,
        "return_on_equity": 0.08,
        "net_debt_to_ebitda": 3.5,
        "current_ratio": 0.8,
        "debt_to_equity": 1.50,
        "fcf_margin": 0.10,
    },
    "semiconductor": {
        "ebitda_margin": 0.28,
        "gross_margin": 0.52,
        "net_profit_margin": 0.14,
        "revenue_cagr": 0.10,
        "return_on_equity": 0.18,
        "net_debt_to_ebitda": 0.5,
        "current_ratio": 2.5,
        "debt_to_equity": 0.20,
        "fcf_margin": 0.16,
    },
    "media_entertainment": {
        "ebitda_margin": 0.18,
        "gross_margin": 0.45,
        "net_profit_margin": 0.06,
        "revenue_cagr": 0.06,
        "return_on_equity": 0.09,
        "net_debt_to_ebitda": 2.8,
        "current_ratio": 1.1,
        "debt_to_equity": 0.70,
        "fcf_margin": 0.08,
    },
}

_SECTOR_ALIASES: dict[str, str] = {
    "tech": "software",
    "technology": "software",
    "enterprise_software": "saas",
    "cloud": "saas",
    "biotech": "pharmaceuticals",
    "pharma": "pharmaceuticals",
    "ecommerce": "retail",
    "e-commerce": "retail",
    "oil_gas": "energy",
    "oil": "energy",
    "gas": "energy",
    "reit": "real_estate",
    "telecom": "telecommunications",
    "chips": "semiconductor",
    "fintech": "financial_services",
    "banking": "financial_services",
    "insurance": "financial_services",
    "media": "media_entertainment",
    "entertainment": "media_entertainment",
    "consumer": "consumer_goods",
    "fmcg": "consumer_goods",
    "medtech": "healthcare",
    "medical_devices": "healthcare",
}


def compare_industry_benchmarks(metrics: dict, industry_code: str) -> dict:
    """Compare computed financial metrics against public industry benchmark medians.

    Benchmarks are Damodaran Online (NYU Stern) sector medians from January 2024,
    derived from public US company filings. These are approximate sector medians —
    they represent a useful directional signal, not an exact match for any specific
    company's peer group. Use alongside sector-specific public reports for context.

    Args:
        metrics: dict — a subset of the ratios from calculate_financial_ratios
          (profitability, leverage, liquidity keys), or a flat dict of metric
          name → float value. Keys matched: ebitda_margin, gross_margin,
          net_profit_margin, return_on_equity, net_debt_to_ebitda,
          current_ratio, debt_to_equity, fcf_margin, revenue_cagr.
        industry_code: str — sector name or SIC-adjacent keyword.
          Supported: saas, software, healthcare, pharmaceuticals, manufacturing,
          retail, financial_services, energy, real_estate, consumer_goods,
          telecommunications, semiconductor, media_entertainment.
          Aliases accepted: tech, cloud, biotech, pharma, ecommerce, fintech, etc.

    Returns:
        dict with data_available (bool), sector_matched (str),
        benchmark_year (str), comparisons (list of per-metric dicts with
        company_value, benchmark_median, status), caveats, source_url, source
    """
    # Normalise and resolve aliases.
    key = industry_code.strip().lower().replace(" ", "_").replace("-", "_")
    key = _SECTOR_ALIASES.get(key, key)
    benchmarks = _DAMODARAN_BENCHMARKS.get(key)

    if benchmarks is None:
        return {
            "data_available": False,
            "message": (
                f"No benchmark data found for industry_code='{industry_code}'. "
                f"Supported sectors: {', '.join(sorted(_DAMODARAN_BENCHMARKS.keys()))}. "
                f"Aliases: {', '.join(sorted(_SECTOR_ALIASES.keys()))}."
            ),
            "source_url": "https://pages.stern.nyu.edu/~adamodar/",
            "source": "Damodaran Online (NYU Stern)",
        }

    # Flatten nested ratios dict if passed as nested structure.
    flat_metrics: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            flat_metrics.update(v)
        elif isinstance(v, (int, float)):
            flat_metrics[k] = v

    METRIC_LABELS: dict[str, str] = {
        "ebitda_margin": "EBITDA Margin",
        "gross_margin": "Gross Margin",
        "net_profit_margin": "Net Profit Margin",
        "return_on_equity": "Return on Equity",
        "net_debt_to_ebitda": "Net Debt / EBITDA",
        "current_ratio": "Current Ratio",
        "debt_to_equity": "Debt / Equity",
        "fcf_margin": "FCF Margin",
        "revenue_cagr": "Revenue CAGR",
    }

    # Higher = better for these metrics.
    HIGHER_BETTER = {
        "ebitda_margin", "gross_margin", "net_profit_margin",
        "return_on_equity", "fcf_margin", "current_ratio", "revenue_cagr",
    }
    # Lower = better for these metrics.
    LOWER_BETTER = {"net_debt_to_ebitda", "debt_to_equity"}

    comparisons: list[dict] = []
    for metric_key, label in METRIC_LABELS.items():
        company_val = flat_metrics.get(metric_key)
        benchmark_val = benchmarks.get(metric_key)

        if benchmark_val is None:
            comparisons.append({
                "metric": label,
                "company_value": company_val,
                "benchmark_median": None,
                "status": "BENCHMARK_NOT_APPLICABLE",
                "note": "This metric is not meaningful for this sector.",
            })
            continue

        if company_val is None:
            comparisons.append({
                "metric": label,
                "company_value": None,
                "benchmark_median": round(benchmark_val, 4),
                "status": "COMPANY_DATA_MISSING",
                "note": f"Pass '{metric_key}' in metrics to enable this comparison.",
            })
            continue

        # Determine status relative to benchmark.
        THRESHOLD_PCT = 0.15  # within 15% of benchmark = ON_PAR
        if benchmark_val != 0:
            deviation = (company_val - benchmark_val) / abs(benchmark_val)
        else:
            deviation = 0.0

        if metric_key in HIGHER_BETTER:
            if deviation > THRESHOLD_PCT:
                status = "ABOVE_BENCHMARK"
            elif deviation < -THRESHOLD_PCT:
                status = "BELOW_BENCHMARK"
            else:
                status = "ON_PAR"
        else:  # LOWER_BETTER
            if deviation < -THRESHOLD_PCT:
                status = "ABOVE_BENCHMARK"  # less leverage = better
            elif deviation > THRESHOLD_PCT:
                status = "BELOW_BENCHMARK"  # more leverage = worse
            else:
                status = "ON_PAR"

        comparisons.append({
            "metric": label,
            "company_value": round(company_val, 4),
            "benchmark_median": round(benchmark_val, 4),
            "deviation_pct": round(deviation * 100, 1),
            "status": status,
        })

    return {
        "data_available": True,
        "sector_matched": key,
        "benchmark_year": "January 2024",
        "comparisons": comparisons,
        "caveats": [
            "Benchmarks are US public company sector medians from Damodaran Online (January 2024).",
            "Private companies and non-US companies may differ materially from these medians.",
            "Sub-sector differences within a broad category can be significant.",
            "Use as a directional signal only; complement with company-specific peer analysis.",
        ],
        "source_url": "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/margin.html",
        "source": "Damodaran Online (NYU Stern) — January 2024 industry data",
    }
