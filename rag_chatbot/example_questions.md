# Example Q&A Outputs - Prudential 2022 Annual Report RAG Chatbot

This document contains 8 example questions and their expected outputs from the RAG-based Q&A system.

---

## Question 1: What are Prudential's strategic priorities for the next 5 years?

**Expected Answer:**
Prudential's strategic priorities focus on:

1. **Customer-centric growth**: Focusing on health and protection products with recurring premiums
2. **Digital transformation**: Enhancing digital distribution channels and customer experience
3. **Geographic expansion**: Growing presence in high-growth markets across Asia and Africa
4. **Operational efficiency**: Improving cost-to-income ratios and leveraging technology

The company aims to achieve compound annual growth in EEV operating profit while maintaining strong capital generation. (Pages 12-15)

**Sources:** Strategic Report section, pages 12-20

---

## Question 2: How does Prudential manage climate-related risks?

**Expected Answer:**
Prudential manages climate-related risks through:

- **Investment portfolio decarbonization**: The Weighted Average Carbon Intensity (WACI) has declined 43% compared to the 2019 baseline
- **TCFD alignment**: Full disclosure following Task Force on Climate-related Financial Disclosures framework
- **Transition financing**: Supporting clients in the transition to a low-carbon economy
- **Physical risk assessment**: Evaluating climate-related physical risks in underwriting

The company seeks an inclusive transition that supports sustainable growth in its markets. (Pages 34, 65-70)

**Sources:** Sustainability section, Risk Management section

---

## Question 3: What was the CEO's main message about company performance in 2022?

**Expected Answer:**
The CEO highlighted that despite challenging market conditions, Prudential delivered:

- Strong operational performance with adjusted operating profit growth of 8% on constant exchange rates
- Successful execution of growth strategy with 9% APE sales growth (constant FX)
- Continued progress on geographic diversification beyond Greater China
- Robust capital position with 307% Solvency II cover ratio

The message emphasized the resilient business model and commitment to long-term value creation for shareholders. (Pages 6-9)

**Sources:** CEO Statement section, pages 6-9

---

## Question 4: How did the Asia segment perform in 2022?

**Expected Answer:**
Asia segment performance in 2022:

- **APE Sales**: $4,393 million (up 9% on constant FX)
- **New Business Profit**: $2,184 million (down 11% on constant FX due to higher interest rates and business mix)
- **Key markets**:
  - CPL (Chinese Mainland): NBP up 15% to $387m
  - Singapore: APE sales up 6%, strong recurring premium growth
  - Hong Kong: NBP declined 47% due to margin compression
  - Growth markets: NBP up 20%

The segment demonstrated resilience despite challenging conditions in Hong Kong. (Pages 43-44)

**Sources:** Financial Review section, pages 38-48

---

## Question 5: What are the key risks facing Prudential?

**Expected Answer:**
Key risks identified in the 2022 report:

1. **Market risks**: Interest rate volatility, equity market declines affecting investment returns
2. **Credit risks**: Potential defaults in investment portfolio
3. **Insurance risks**: Mortality, morbidity, and persistency assumptions
4. **Operational risks**: Cybersecurity, IT system failures, regulatory compliance
5. **Strategic risks**: Competition, geopolitical tensions, regulatory changes
6. **Climate risks**: Physical and transition risks from climate change

The risk management framework includes regular stress testing and scenario analysis. (Pages 54-70)

**Sources:** Risk Review section, pages 54-70

---

## Question 6: What is Prudential's approach to ESG?

**Expected Answer:**
Prudential's ESG approach encompasses:

**Environmental:**
- Committed to net-zero portfolio emissions by 2050
- 43% reduction in portfolio carbon intensity since 2019
- Climate risk integration in investment decisions

**Social:**
- Financial inclusion initiatives across Asia and Africa
- Health and protection products addressing protection gap
- Community investment programs

**Governance:**
- Board oversight of ESG matters
- Executive remuneration linked to ESG metrics
- TCFD-aligned climate disclosure

The company views ESG as integral to long-term value creation. (Pages 75-90)

**Sources:** Sustainability section, Governance section

---

## Question 7: How is Prudential positioned in the African market?

**Expected Answer:**
Prudential's African market position:

- **Coverage**: Operations in multiple African markets with growing presence
- **Growth potential**: Large unmet insurance needs with low penetration rates
- **Strategy**: Focus on digital distribution and microinsurance products
- **Performance**: Part of "Growth markets and other" segment showing 20% NBP growth

Africa represents a key diversification opportunity beyond the Greater China concentration. The company is investing in building distribution capabilities and product innovation for local needs. (Pages 26-32)

**Sources:** Strategic Report - Segment Reviews

---

## Question 8: What are the main challenges in the current market environment?

**Expected Answer:**
Main challenges facing Prudential in 2022:

1. **Interest rate volatility**: Sharp rises impacted embedded value and policyholder liabilities
2. **Equity market declines**: Reduced investment returns and asset management AUM
3. **Hong Kong market conditions**: Significant NBP margin compression
4. **Currency headwinds**: Strong USD impacting reported results
5. **Regulatory changes**: IFRS 17 implementation costs of $277m
6. **Geopolitical tensions**: Uncertainty in key Asian markets

Despite these challenges, the business demonstrated resilience with strong underlying operational performance. (Pages 38-42)

**Sources:** Financial Review section, Risk Review section

---

## Notes on RAG System Performance

- **Retrieval accuracy**: The system typically retrieves highly relevant passages with similarity scores > 0.75
- **Answer quality**: Responses are grounded in the source material with proper citations
- **Section filtering**: Users can focus searches on specific report sections for more targeted results
- **Latency**: Average response time ~2-3 seconds including embedding generation and LLM completion
