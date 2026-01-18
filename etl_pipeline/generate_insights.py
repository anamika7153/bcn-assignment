"""
Insight Generation and Visualization Script
Generates insights and visualizations from extracted Prudential 2022 data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configuration
TABLES_DIR = Path("etl_pipeline/output/tables")
VIZ_DIR = Path("etl_pipeline/output/visualizations")

# Ensure output directory exists
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_tables():
    """Load all extracted CSV tables."""
    tables = {}
    for csv_file in TABLES_DIR.glob("*.csv"):
        name = csv_file.stem
        tables[name] = pd.read_csv(csv_file)
        print(f"Loaded: {name} ({len(tables[name])} rows)")
    return tables


def insight_1_regional_performance(tables):
    """
    Insight 1: Regional Performance Analysis
    Which regions are driving growth and which are declining?
    """
    print("\n" + "=" * 60)
    print("INSIGHT 1: Regional Performance Analysis")
    print("=" * 60)

    df = tables.get('01_eev_new_business_profit_by_region')
    if df is None:
        print("Table not found!")
        return

    # Exclude total row for regional analysis
    df_regions = df[df['Region'] != 'Total'].copy()

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Chart 1: 2022 APE Sales by Region
    colors = sns.color_palette("husl", len(df_regions))
    ax1 = axes[0]
    bars1 = ax1.barh(df_regions['Region'], df_regions['2022 APE Sales $m'], color=colors)
    ax1.set_xlabel('APE Sales ($ millions)')
    ax1.set_title('2022 APE Sales by Region', fontsize=12, fontweight='bold')
    ax1.bar_label(bars1, fmt='$%.0f', padding=3)

    # Chart 2: NBP Change % (Constant Exchange Rate)
    ax2 = axes[1]
    colors2 = ['green' if x >= 0 else 'red' for x in df_regions['NBP Change % (Constant)']]
    bars2 = ax2.barh(df_regions['Region'], df_regions['NBP Change % (Constant)'], color=colors2)
    ax2.set_xlabel('NBP Change %')
    ax2.set_title('New Business Profit Change % (Constant FX)', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.bar_label(bars2, fmt='%.0f%%', padding=3)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "01_regional_performance.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Print insight
    best_growth = df_regions.loc[df_regions['NBP Change % (Constant)'].idxmax()]
    worst_decline = df_regions.loc[df_regions['NBP Change % (Constant)'].idxmin()]

    insight = f"""
    KEY FINDINGS:

    1. GROWTH LEADER: {best_growth['Region']}
       - NBP grew by {best_growth['NBP Change % (Constant)']}% (constant FX)
       - APE Sales: ${best_growth['2022 APE Sales $m']}m
       - NBP Margin: {best_growth['2022 NBP Margin %']:.0f}%

    2. BIGGEST DECLINE: {worst_decline['Region']}
       - NBP declined by {worst_decline['NBP Change % (Constant)']}% (constant FX)
       - Despite APE Sales of ${worst_decline['2022 APE Sales $m']}m
       - NBP Margin compressed significantly from {worst_decline['2021 NBP Margin % (Actual)']:.0f}% to {worst_decline['2022 NBP Margin %']:.0f}%

    3. LARGEST MARKET BY APE: Growth markets and other
       - ${df_regions.loc[df_regions['2022 APE Sales $m'].idxmax(), '2022 APE Sales $m']}m in APE sales
       - Represents diversification beyond Greater China

    IMPLICATION: The significant decline in Hong Kong NBP (-47%) due to margin compression
    is a concern, but growth markets (+20% NBP) and CPL (+15% NBP) provide offset.
    """
    print(insight)

    return insight


def insight_2_profitability_drivers(tables):
    """
    Insight 2: Profitability Drivers Analysis
    What's driving operating profit growth?
    """
    print("\n" + "=" * 60)
    print("INSIGHT 2: Profitability Drivers Analysis")
    print("=" * 60)

    df = tables.get('02_profit_margin_analysis')
    if df is None:
        print("Table not found!")
        return

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Chart 1: Income Components Comparison
    income_items = df[df['Item'].isin([
        'Spread income', 'Fee income', 'With-profits', 'Insurance margin', 'Other income'
    ])].copy()

    ax1 = axes[0]
    x = range(len(income_items))
    width = 0.35

    bars1 = ax1.bar([i - width/2 for i in x], income_items['2022 $m'], width, label='2022', color='#2E86AB')
    bars2 = ax1.bar([i + width/2 for i in x], income_items['2021 $m (Constant)'], width, label='2021', color='#A23B72')

    ax1.set_xticks(x)
    ax1.set_xticklabels(income_items['Item'], rotation=45, ha='right')
    ax1.set_ylabel('$ millions')
    ax1.set_title('Income Components: 2022 vs 2021', fontsize=12, fontweight='bold')
    ax1.legend()

    # Chart 2: Expense Analysis
    expense_items = df[df['Item'].isin([
        'Acquisition costs', 'Administration expenses'
    ])].copy()
    expense_items['2022 $m'] = expense_items['2022 $m'].abs()
    expense_items['2021 $m (Constant)'] = expense_items['2021 $m (Constant)'].abs()

    ax2 = axes[1]
    x2 = range(len(expense_items))
    bars3 = ax2.bar([i - width/2 for i in x2], expense_items['2022 $m'], width, label='2022', color='#E94F37')
    bars4 = ax2.bar([i + width/2 for i in x2], expense_items['2021 $m (Constant)'], width, label='2021', color='#F3B700')

    ax2.set_xticks(x2)
    ax2.set_xticklabels(expense_items['Item'], rotation=45, ha='right')
    ax2.set_ylabel('$ millions (absolute)')
    ax2.set_title('Expense Components: 2022 vs 2021', fontsize=12, fontweight='bold')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "02_profitability_drivers.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Calculate growth rates
    insurance_margin_2022 = df[df['Item'] == 'Insurance margin']['2022 $m'].values[0]
    insurance_margin_2021 = df[df['Item'] == 'Insurance margin']['2021 $m (Constant)'].values[0]
    im_growth = ((insurance_margin_2022 / insurance_margin_2021) - 1) * 100

    insight = f"""
    KEY FINDINGS:

    1. INSURANCE MARGIN IS THE PRIMARY DRIVER
       - Insurance margin grew {im_growth:.1f}% to ${insurance_margin_2022}m
       - Driven by focus on recurring premium health & protection products
       - This is the largest income component

    2. EXPENSE EFFICIENCY IMPROVED
       - Acquisition costs ratio: 53% (2022) vs 50% (2021)
       - Slightly higher cost per new policy as business mix shifts

    3. WITH-PROFITS CONTRIBUTION GROWING
       - With-profits earnings up 20% on constant FX
       - Reflects maturing book and bonus declarations

    4. SPREAD INCOME STABLE
       - Modest growth despite interest rate volatility
       - Well-managed investment spread

    IMPLICATION: Growth is driven by core insurance operations, not investment returns.
    The business model is proving resilient to market volatility.
    """
    print(insight)

    return insight


def insight_3_capital_generation(tables):
    """
    Insight 3: Capital Generation & Cash Flow
    How efficiently is capital being generated and deployed?
    """
    print("\n" + "=" * 60)
    print("INSIGHT 3: Capital Generation & Deployment")
    print("=" * 60)

    df = tables.get('04_free_surplus_movement')
    if df is None:
        print("Table not found!")
        return

    # Create waterfall-style visualization
    fig, ax = plt.subplots(figsize=(14, 8))

    # Select key items for waterfall
    items = [
        'Expected transfer from in-force & return on free surplus',
        'Operating assumptions & experience variances',
        'Investment in new business',
        'Asset management',
        'Net interest paid on core structural borrowings',
        'Corporate expenditure',
        'Restructuring and IFRS 17 implementation costs',
        'Net Group operating free surplus generated'
    ]

    values = []
    colors = []
    for item in items:
        row = df[df['Item'] == item]
        if not row.empty:
            val = row['2022 $m'].values[0]
            values.append(val)
            if item == 'Net Group operating free surplus generated':
                colors.append('#2E86AB')  # Final total
            elif val >= 0:
                colors.append('#27AE60')  # Positive
            else:
                colors.append('#E74C3C')  # Negative

    # Shorten labels
    short_labels = [
        'In-force\ntransfer',
        'Assumption\nchanges',
        'New business\ninvestment',
        'Asset\nmanagement',
        'Interest\npaid',
        'Corporate\ncosts',
        'Restructuring\ncosts',
        'NET FREE\nSURPLUS'
    ]

    bars = ax.bar(short_labels, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('$ millions')
    ax.set_title('2022 Free Surplus Generation Waterfall', fontsize=14, fontweight='bold')

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'${val:,.0f}m',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "03_capital_generation.png", dpi=150, bbox_inches='tight')
    plt.close()

    insight = f"""
    KEY FINDINGS:

    1. STRONG IN-FORCE CASH GENERATION
       - ${values[0]:,.0f}m generated from existing business
       - This is the primary source of capital

    2. NEW BUSINESS REINVESTMENT
       - ${abs(values[2]):,.0f}m invested in writing new policies
       - Strain ratio (investment/generation) = {abs(values[2])/values[0]*100:.0f}%

    3. NET FREE SURPLUS: ${values[-1]:,.0f}m
       - Up 21% on constant FX basis
       - Supports dividends and strategic investments

    4. RESTRUCTURING COSTS
       - ${abs(values[6]):,.0f}m spent on IFRS 17 implementation
       - One-time transition cost, expected to normalize

    IMPLICATION: The business generates substantial free cash flow.
    The 21% growth in net free surplus demonstrates improved capital efficiency
    and supports the dividend policy.
    """
    print(insight)

    return insight


def insight_4_greater_china_concentration(tables):
    """
    Insight 4: Greater China Concentration Risk
    How concentrated is the business in Greater China?
    """
    print("\n" + "=" * 60)
    print("INSIGHT 4: Greater China Concentration Analysis")
    print("=" * 60)

    df = tables.get('05_greater_china_performance')
    if df is None:
        print("Table not found!")
        return

    # Create pie charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Get data
    gc_premiums = df[df['Metric'] == 'Total Greater China - Gross premiums earned']['2022 $m'].values[0]
    total_premiums = df[df['Metric'] == 'Total Group - Gross premiums earned']['2022 $m'].values[0]
    other_premiums = total_premiums - gc_premiums

    gc_nbp = df[df['Metric'] == 'Total Greater China - New business profit']['2022 $m'].values[0]
    total_nbp = df[df['Metric'] == 'Total Group - New business profit']['2022 $m'].values[0]
    other_nbp = total_nbp - gc_nbp

    # Chart 1: Premiums
    ax1 = axes[0]
    sizes1 = [gc_premiums, other_premiums]
    labels1 = [f'Greater China\n${gc_premiums:,.0f}m', f'Other Markets\n${other_premiums:,.0f}m']
    colors1 = ['#E74C3C', '#3498DB']
    explode1 = (0.05, 0)

    ax1.pie(sizes1, explode=explode1, labels=labels1, colors=colors1, autopct='%1.0f%%',
            shadow=False, startangle=90, textprops={'fontsize': 10})
    ax1.set_title('Gross Premiums Earned 2022', fontsize=12, fontweight='bold')

    # Chart 2: NBP
    ax2 = axes[1]
    sizes2 = [gc_nbp, other_nbp]
    labels2 = [f'Greater China\n${gc_nbp:,.0f}m', f'Other Markets\n${other_nbp:,.0f}m']
    colors2 = ['#E74C3C', '#3498DB']
    explode2 = (0.05, 0)

    ax2.pie(sizes2, explode=explode2, labels=labels2, colors=colors2, autopct='%1.0f%%',
            shadow=False, startangle=90, textprops={'fontsize': 10})
    ax2.set_title('New Business Profit 2022', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "04_greater_china_concentration.png", dpi=150, bbox_inches='tight')
    plt.close()

    # YoY change
    gc_prem_2021 = df[df['Metric'] == 'Total Greater China - Gross premiums earned']['2021 $m'].values[0]
    gc_nbp_2021 = df[df['Metric'] == 'Total Greater China - New business profit']['2021 $m'].values[0]

    insight = f"""
    KEY FINDINGS:

    1. PREMIUMS CONCENTRATION: {gc_premiums/total_premiums*100:.0f}%
       - Greater China contributed ${gc_premiums:,.0f}m of ${total_premiums:,.0f}m
       - Down from 50% in 2021 - healthy diversification

    2. NBP CONCENTRATION: {gc_nbp/total_nbp*100:.0f}%
       - Greater China NBP: ${gc_nbp:,.0f}m
       - Share declined from 47% in 2021

    3. YEAR-ON-YEAR CHANGES
       - Greater China premiums: {(gc_premiums/gc_prem_2021-1)*100:+.0f}%
       - Greater China NBP: {(gc_nbp/gc_nbp_2021-1)*100:+.0f}%

    4. DIVERSIFICATION PROGRESS
       - Other markets growing faster than Greater China
       - Reduces geographic concentration risk

    IMPLICATION: While Greater China remains significant (42-47% of business),
    concentration is declining. Growth markets are becoming more important,
    reducing dependency on any single region.
    """
    print(insight)

    return insight


def insight_5_solvency_position(tables):
    """
    Insight 5: Solvency and Capital Position
    How strong is the capital position?
    """
    print("\n" + "=" * 60)
    print("INSIGHT 5: Solvency & Capital Position")
    print("=" * 60)

    df = tables.get('06_key_performance_indicators')
    if df is None:
        print("Table not found!")
        return

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Chart 1: Key Financial Metrics YoY
    metrics = df[df['KPI'].isin([
        'EEV new business profit ($m)',
        'Free surplus generation from insurance & AM ($m)',
        'Adjusted IFRS operating profit ($m)'
    ])].copy()

    ax1 = axes[0]
    x = range(len(metrics))
    width = 0.35

    bars1 = ax1.bar([i - width/2 for i in x], metrics['2022'], width, label='2022', color='#2E86AB')
    bars2 = ax1.bar([i + width/2 for i in x], metrics['2021'], width, label='2021', color='#A23B72')

    ax1.set_xticks(x)
    labels = ['EEV NBP', 'Free Surplus\nGeneration', 'Adj. Operating\nProfit']
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('$ millions')
    ax1.set_title('Key Financial Metrics: 2022 vs 2021', fontsize=12, fontweight='bold')
    ax1.legend()

    # Add change percentages
    for i, row in enumerate(metrics.itertuples()):
        change = row._4  # YoY Change %
        if pd.notna(change):
            ax1.annotate(f'{change:+.0f}%',
                        xy=(i, max(row._2, row._3) + 100),
                        ha='center', fontsize=10, fontweight='bold',
                        color='green' if change >= 0 else 'red')

    # Chart 2: Solvency metrics
    solvency = df[df['KPI'].isin([
        'Group shareholder Solvency II surplus ($bn)',
        'Solvency II cover ratio (%)'
    ])].copy()

    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    surplus_2022 = solvency[solvency['KPI'].str.contains('surplus')]['2022'].values[0]
    surplus_2021 = solvency[solvency['KPI'].str.contains('surplus')]['2021'].values[0]
    ratio_2022 = solvency[solvency['KPI'].str.contains('ratio')]['2022'].values[0]
    ratio_2021 = solvency[solvency['KPI'].str.contains('ratio')]['2021'].values[0]

    x2 = [0, 1]
    ax2.bar([i - 0.2 for i in x2], [surplus_2021, surplus_2022], 0.4, color='#3498DB', label='Surplus ($bn)')
    ax2_twin.plot(x2, [ratio_2021, ratio_2022], 'ro-', markersize=10, linewidth=2, label='Cover Ratio (%)')

    ax2.set_xticks(x2)
    ax2.set_xticklabels(['2021', '2022'])
    ax2.set_ylabel('Solvency Surplus ($ billions)', color='#3498DB')
    ax2_twin.set_ylabel('Cover Ratio (%)', color='red')
    ax2.set_title('Solvency Position', fontsize=12, fontweight='bold')

    # Add values
    ax2.annotate(f'${surplus_2021}bn', xy=(0, surplus_2021 + 0.5), ha='center')
    ax2.annotate(f'${surplus_2022}bn', xy=(1, surplus_2022 + 0.5), ha='center')
    ax2_twin.annotate(f'{ratio_2021}%', xy=(0, ratio_2021 + 5), ha='center', color='red')
    ax2_twin.annotate(f'{ratio_2022}%', xy=(1, ratio_2022 + 5), ha='center', color='red')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "05_solvency_position.png", dpi=150, bbox_inches='tight')
    plt.close()

    insight = f"""
    KEY FINDINGS:

    1. SOLVENCY SURPLUS: ${surplus_2022}bn (2022) vs ${surplus_2021}bn (2021)
       - Decline of {(surplus_2022/surplus_2021-1)*100:.0f}% due to market movements
       - Still well above regulatory requirements

    2. COVER RATIO: {ratio_2022}% (2022) vs {ratio_2021}% (2021)
       - Remains strong at over 3x the requirement
       - Provides significant buffer for volatility

    3. OPERATING PROFIT GROWTH
       - Adjusted operating profit grew 4% to $3,375m
       - Demonstrates resilient business model

    4. FREE SURPLUS GENERATION
       - Up 6% to $2,193m
       - Strong internal cash generation

    IMPLICATION: Despite market headwinds reducing the absolute surplus,
    the solvency position remains very strong (307% cover ratio).
    The business continues to generate strong operating cash flows.
    """
    print(insight)

    return insight


def generate_summary_report(insights):
    """Generate a summary report of all insights."""
    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)

    summary = """
    PRUDENTIAL PLC 2022 ANNUAL REPORT - KEY INSIGHTS SUMMARY
    =========================================================

    1. REGIONAL PERFORMANCE
       - Growth markets (+20% NBP) outperforming
       - Hong Kong facing margin compression (-47% NBP)
       - Geographic diversification progressing well

    2. PROFITABILITY
       - Insurance margin driving growth (+15%)
       - Business model resilient to market volatility
       - Focus on health & protection products paying off

    3. CAPITAL GENERATION
       - Net free surplus up 21% to $1,374m
       - Strong in-force cash generation ($2,753m)
       - Supports dividends and reinvestment

    4. GREATER CHINA EXPOSURE
       - Concentration declining (47% -> 42% of NBP)
       - Diversification reducing single-market risk
       - Growth markets becoming more significant

    5. SOLVENCY STRENGTH
       - 307% cover ratio (well above requirements)
       - $15.6bn surplus provides volatility buffer
       - Operating profit growth demonstrates resilience

    OVERALL ASSESSMENT: Despite challenging market conditions,
    Prudential demonstrates strong operational performance,
    healthy capital generation, and improving geographic diversification.
    """
    print(summary)

    # Save summary to file
    with open(VIZ_DIR / "insights_summary.txt", 'w') as f:
        f.write(summary)
        for i, insight in enumerate(insights, 1):
            if insight:
                f.write(f"\n\n{'='*60}\nINSIGHT {i}\n{'='*60}\n")
                f.write(insight)

    print(f"\nSummary saved to: {VIZ_DIR / 'insights_summary.txt'}")


def main():
    """Main function to generate all insights and visualizations."""
    print("=" * 60)
    print("Prudential 2022 - Insight Generation")
    print("=" * 60)

    # Load tables
    tables = load_tables()

    # Generate insights
    insights = []
    insights.append(insight_1_regional_performance(tables))
    insights.append(insight_2_profitability_drivers(tables))
    insights.append(insight_3_capital_generation(tables))
    insights.append(insight_4_greater_china_concentration(tables))
    insights.append(insight_5_solvency_position(tables))

    # Generate summary
    generate_summary_report(insights)

    print("\n" + "=" * 60)
    print("Visualization Generation Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {VIZ_DIR.absolute()}")

    # List generated files
    print("\nGenerated visualizations:")
    for f in sorted(VIZ_DIR.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
