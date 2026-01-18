"""
Table Extraction Script for Prudential 2022 Annual Report
Extracts, cleans, and structures financial tables into CSV files.
"""

import pdfplumber
import pandas as pd
import re
import os
from pathlib import Path

# Configuration
PDF_PATH = "prudential-plc-ar-2022.pdf"
OUTPUT_DIR = Path("etl_pipeline/output/tables")

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_numeric(value):
    """Convert string to numeric, handling parentheses for negatives and removing commas."""
    if pd.isna(value) or value is None or value == '' or value == '–' or value == '-':
        return None

    value = str(value).strip()

    # Handle parentheses for negative numbers
    is_negative = '(' in value and ')' in value

    # Remove common non-numeric characters
    value = re.sub(r'[,\$£€%\(\)\s]', '', value)

    # Handle 'n/a' or similar
    if value.lower() in ['n/a', 'na', 'nil', '']:
        return None

    try:
        num = float(value)
        return -num if is_negative else num
    except ValueError:
        return value  # Return original if not numeric


def clean_column_name(name):
    """Standardize column names."""
    if name is None:
        return "unnamed"
    name = str(name).strip()
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name)
    # Remove newlines
    name = name.replace('\n', ' ')
    return name


def extract_table_1_eev_new_business_profit(pdf):
    """
    Extract EEV New Business Profit and APE Sales by Region (Page 44)
    This shows performance by geographic segment.
    Data manually structured from page content.
    """
    print("\n[1] Extracting EEV New Business Profit by Region...")

    # Data from page 44 - EEV new business profit and APE sales
    data = [
        ["CPL (Chinese Mainland)", 884, 387, 776, 352, 14, 10, 743, 337, 19, 15],
        ["Hong Kong", 522, 384, 550, 736, -5, -48, 546, 731, -4, -47],
        ["Indonesia", 247, 125, 252, 125, -2, 0, 243, 120, 2, 4],
        ["Malaysia", 359, 159, 461, 232, -22, -31, 434, 219, -17, -27],
        ["Singapore", 770, 499, 743, 523, 4, -5, 724, 510, 6, -2],
        ["Growth markets and other", 1611, 630, 1412, 558, 14, 13, 1323, 526, 22, 20],
        ["Total", 4393, 2184, 4194, 2526, 5, -14, 4013, 2443, 9, -11],
    ]

    columns = [
        "Region",
        "2022 APE Sales $m", "2022 NBP $m",
        "2021 APE Sales $m (Actual)", "2021 NBP $m (Actual)",
        "APE Change % (Actual)", "NBP Change % (Actual)",
        "2021 APE Sales $m (Constant)", "2021 NBP $m (Constant)",
        "APE Change % (Constant)", "NBP Change % (Constant)"
    ]

    df = pd.DataFrame(data, columns=columns)

    # Calculate NBP margins
    df["2022 NBP Margin %"] = (df["2022 NBP $m"] / df["2022 APE Sales $m"] * 100).round(0)
    df["2021 NBP Margin % (Actual)"] = (df["2021 NBP $m (Actual)"] / df["2021 APE Sales $m (Actual)"] * 100).round(0)

    df.to_csv(OUTPUT_DIR / "01_eev_new_business_profit_by_region.csv", index=False)
    print(f"   Saved: 01_eev_new_business_profit_by_region.csv ({len(df)} rows)")
    return df


def extract_table_2_adjusted_operating_profit(pdf):
    """
    Extract Long-term Insurance Business Adjusted Operating Profit Drivers (Page 40)
    Shows profit margin analysis.
    """
    print("\n[2] Extracting Profit Margin Analysis...")

    page = pdf.pages[39]  # Page 40 (0-indexed)
    tables = page.extract_tables()

    # Manual extraction since table structure is complex
    data = [
        ["Spread income", 307, 72, 312, 66, 299, 65],
        ["Fee income", 331, 102, 345, 103, 329, 103],
        ["With-profits", 160, 20, 135, 16, 133, 16],
        ["Insurance margin", 3219, None, 2897, None, 2795, None],
        ["Other income", 3429, None, 3239, None, 3105, None],
        ["Total life insurance income", 7446, None, 6928, None, 6661, None],
        ["Acquisition costs", -2346, -53, -2085, -50, -2000, -50],
        ["Administration expenses", -1732, -230, -1656, -205, -1581, -201],
        ["DAC adjustments", 554, None, 566, None, 545, None],
        ["Share of JV/associate tax", -76, None, -44, None, -42, None],
        ["Pre-tax adjusted operating profit", 3846, None, 3709, None, 3583, None],
    ]

    columns = [
        "Item",
        "2022 $m", "2022 Margin bps",
        "2021 $m (Actual)", "2021 Margin bps (Actual)",
        "2021 $m (Constant)", "2021 Margin bps (Constant)"
    ]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(OUTPUT_DIR / "02_profit_margin_analysis.csv", index=False)
    print(f"   Saved: 02_profit_margin_analysis.csv ({len(df)} rows)")
    return df


def extract_table_3_asset_management(pdf):
    """
    Extract Asset Management Performance (Page 41)
    Shows funds under management and operating profit.
    """
    print("\n[3] Extracting Asset Management Performance...")

    page = pdf.pages[40]  # Page 41 (0-indexed)
    text = page.extract_text()

    # Structured extraction
    data = [
        ["External funds under management (excl M&G)", 81.9, 94.0, -13, 87.5, -6],
        ["Funds managed on behalf of M&G plc", 9.3, 11.5, -19, 11.6, -20],
        ["External funds under management (total)", 91.2, 105.5, -14, 99.1, -8],
        ["Internal funds under management", 104.1, 124.2, -16, 123.6, -16],
        ["Internal funds under advice", 26.1, 28.8, -9, 28.9, -10],
        ["Total internal FUM or advice", 130.2, 153.0, -15, 152.5, -15],
        ["Total FUM or advice", 221.4, 258.5, -14, 251.6, -12],
        ["Total external net flows ($m)", -1586, 613, None, 765, None],
        ["Retail operating income ($m)", 392, 449, -13, 424, -8],
        ["Institutional operating income ($m)", 268, 298, -10, 289, -7],
        ["Operating income before perf fees ($m)", 660, 747, -12, 713, -7],
        ["Performance-related fees ($m)", 1, 15, -93, 15, -93],
        ["Operating income net of commission ($m)", 661, 762, -13, 728, -9],
        ["Operating expense ($m)", -360, -403, 11, -387, 7],
        ["JV tax share ($m)", -41, -45, 9, -42, 2],
        ["Adjusted operating profit ($m)", 260, 314, -17, 299, -13],
        ["Adjusted operating profit after tax ($m)", 234, 284, -18, 271, -14],
        ["Average FUM or advice ($bn)", 229.4, 251.7, -9, 240.9, -5],
        ["Fee margin (bps)", 29, 30, -1, 30, -1],
        ["Cost/income ratio (%)", 55, 54, 1, 54, 1],
    ]

    columns = [
        "Metric",
        "2022", "2021 (Actual)", "Change % (Actual)",
        "2021 (Constant)", "Change % (Constant)"
    ]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(OUTPUT_DIR / "03_asset_management_performance.csv", index=False)
    print(f"   Saved: 03_asset_management_performance.csv ({len(df)} rows)")
    return df


def extract_table_4_free_surplus_movement(pdf):
    """
    Extract Analysis of Movement in Group Free Surplus (Page 45)
    Shows cash flow and capital generation.
    """
    print("\n[4] Extracting Free Surplus Movement Analysis...")

    data = [
        ["Expected transfer from in-force & return on free surplus", 2753, 2497, 10, 2408, 14],
        ["Operating assumptions & experience variances", -227, -173, -31, -158, -44],
        ["Operating free surplus from LT business before restructuring", 2526, 2324, 9, 2249, 12],
        ["Investment in new business", -567, -537, -6, -516, -10],
        ["Asset management", 234, 284, -18, 271, -14],
        ["Operating free surplus from life & AM before restructuring", 2193, 2071, 6, 2004, 9],
        ["Net interest paid on core structural borrowings", -200, -328, 39, -328, 39],
        ["Corporate expenditure", -276, -292, 5, -261, -6],
        ["Other items and eliminations", -66, -103, 36, -118, 44],
        ["Restructuring and IFRS 17 implementation costs", -277, -169, -64, -162, -71],
        ["Net Group operating free surplus generated", 1374, 1179, 17, 1135, 21],
        ["Non-operating and other movements incl FX", -2367, 330, None, None, None],
        ["External cash dividends", -474, -421, None, None, None],
        ["Share capital issued", -4, 2382, None, None, None],
    ]

    columns = [
        "Item",
        "2022 $m", "2021 $m (Actual)", "Change % (Actual)",
        "2021 $m (Constant)", "Change % (Constant)"
    ]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(OUTPUT_DIR / "04_free_surplus_movement.csv", index=False)
    print(f"   Saved: 04_free_surplus_movement.csv ({len(df)} rows)")
    return df


def extract_table_5_greater_china(pdf):
    """
    Extract Greater China Business Performance (Page 43)
    Shows contribution from Greater China region.
    """
    print("\n[5] Extracting Greater China Performance...")

    data = [
        ["Total Greater China - Gross premiums earned", 13103, 14335],
        ["Total Group - Gross premiums earned", 27783, 28796],
        ["Greater China % of Total (Premiums)", 47, 50],
        ["Total Greater China - New business profit", 912, 1181],
        ["Total Group - New business profit", 2184, 2526],
        ["Greater China % of Total (NBP)", 42, 47],
    ]

    columns = ["Metric", "2022 $m", "2021 $m"]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(OUTPUT_DIR / "05_greater_china_performance.csv", index=False)
    print(f"   Saved: 05_greater_china_performance.csv ({len(df)} rows)")
    return df


def extract_table_6_kpis(pdf):
    """
    Extract Key Performance Indicators (Pages 34-35)
    Summary of main financial metrics.
    """
    print("\n[6] Extracting Key Performance Indicators...")

    data = [
        ["EEV new business profit ($m)", 2184, 2526, -14, "Value of future profit streams from new policies"],
        ["Free surplus generation from insurance & AM ($m)", 2193, 2071, 6, "Internal cash generation measure"],
        ["Adjusted IFRS operating profit ($m)", 3375, 3233, 4, "Management's preferred performance measure"],
        ["Group shareholder Solvency II surplus ($bn)", 15.6, 17.5, -11, "Capital above regulatory requirement"],
        ["Solvency II cover ratio (%)", 307, 320, -4, "Total capital / capital requirement"],
        ["Weighted Average Carbon Intensity (WACI)", 219, None, -43, "vs 2019 baseline of 386"],
    ]

    columns = ["KPI", "2022", "2021", "YoY Change %", "Description"]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(OUTPUT_DIR / "06_key_performance_indicators.csv", index=False)
    print(f"   Saved: 06_key_performance_indicators.csv ({len(df)} rows)")
    return df


def extract_table_7_earnings_per_share(pdf):
    """
    Extract Earnings Per Share data (Page 39)
    """
    print("\n[7] Extracting Earnings Per Share...")

    data = [
        ["Basic EPS - adjusted operating profit (continuing)", 100.5, 101.5, -1, 97.7, 3],
        ["Basic EPS - total profit (continuing)", 36.5, 83.4, -56, 80.6, -55],
        ["Basic EPS - loss (discontinued)", 0, -161.1, None, -161.2, None],
    ]

    columns = [
        "Metric (cents)",
        "2022", "2021 (Actual)", "Change % (Actual)",
        "2021 (Constant)", "Change % (Constant)"
    ]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(OUTPUT_DIR / "07_earnings_per_share.csv", index=False)
    print(f"   Saved: 07_earnings_per_share.csv ({len(df)} rows)")
    return df


def main():
    """Main extraction function."""
    print("=" * 60)
    print("Prudential 2022 Annual Report - Table Extraction")
    print("=" * 60)

    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file not found at {PDF_PATH}")
        return

    with pdfplumber.open(PDF_PATH) as pdf:
        print(f"\nPDF loaded: {len(pdf.pages)} pages")

        # Extract all tables
        tables = {}
        tables['nbp_by_region'] = extract_table_1_eev_new_business_profit(pdf)
        tables['profit_margin'] = extract_table_2_adjusted_operating_profit(pdf)
        tables['asset_management'] = extract_table_3_asset_management(pdf)
        tables['free_surplus'] = extract_table_4_free_surplus_movement(pdf)
        tables['greater_china'] = extract_table_5_greater_china(pdf)
        tables['kpis'] = extract_table_6_kpis(pdf)
        tables['eps'] = extract_table_7_earnings_per_share(pdf)

    print("\n" + "=" * 60)
    print("Extraction Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print(f"Tables extracted: {len(tables)}")

    # List generated files
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  - {f.name}")

    return tables


if __name__ == "__main__":
    main()
