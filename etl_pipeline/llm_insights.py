"""
LLM-based Automated Insight Generation
Uses OpenAI GPT to analyze extracted tables and generate insights.
"""

import os
import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
TABLES_DIR = Path("etl_pipeline/output/tables")
OUTPUT_DIR = Path("etl_pipeline/output")

def load_all_tables():
    """Load all CSV tables and convert to text format."""
    tables_text = []

    for csv_file in sorted(TABLES_DIR.glob("*.csv")):
        df = pd.read_csv(csv_file)
        table_name = csv_file.stem.replace('_', ' ').title()

        # Convert to markdown table format
        md_table = df.to_markdown(index=False)
        tables_text.append(f"### {table_name}\n\n{md_table}")

    return "\n\n".join(tables_text)


def generate_llm_insights(tables_text: str) -> str:
    """Use OpenAI to generate insights from the tables."""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system_prompt = """You are a senior financial analyst specializing in insurance companies.
You analyze financial data and provide clear, actionable insights for executive audiences.
Focus on:
1. Key trends and year-over-year changes
2. Regional performance differences
3. Profitability drivers
4. Risk factors and concerns
5. Capital efficiency and solvency

Be specific with numbers and percentages. Highlight both positives and concerns."""

    user_prompt = f"""Analyze the following financial tables from Prudential's 2022 Annual Report
and provide 5 key insights. Each insight should:
- Have a clear headline
- Include specific numbers from the data
- Explain the business implication
- Be 3-4 sentences maximum

FINANCIAL DATA:

{tables_text}

Please provide exactly 5 insights in the following format:

**Insight 1: [Headline]**
[Analysis with specific numbers and implications]

**Insight 2: [Headline]**
...

End with a brief OVERALL ASSESSMENT (2-3 sentences)."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=1500
    )

    return response.choices[0].message.content


def main():
    """Main function for LLM insight generation."""
    print("=" * 60)
    print("LLM-Based Automated Insight Generation")
    print("=" * 60)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or environment.")
        print("\nTo create a .env file:")
        print("  cp .env.example .env")
        print("  # Then add your OpenAI API key")
        return

    print("\nLoading extracted tables...")
    tables_text = load_all_tables()
    print(f"Loaded {len(list(TABLES_DIR.glob('*.csv')))} tables")

    print("\nGenerating insights with GPT-4o-mini...")
    try:
        insights = generate_llm_insights(tables_text)

        print("\n" + "=" * 60)
        print("LLM-GENERATED INSIGHTS")
        print("=" * 60)
        print(insights)

        # Save to file
        output_file = OUTPUT_DIR / "llm_insights.md"
        with open(output_file, 'w') as f:
            f.write("# LLM-Generated Insights - Prudential 2022\n\n")
            f.write("*Generated using GPT-4o-mini from extracted financial tables*\n\n")
            f.write(insights)

        print(f"\n\nInsights saved to: {output_file}")

    except Exception as e:
        print(f"\nError generating insights: {e}")
        print("Please check your API key and try again.")


if __name__ == "__main__":
    main()
