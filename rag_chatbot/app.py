"""
Streamlit Chatbot Application
Interactive Q&A interface for Prudential 2022 Annual Report.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Prudential AR 2022 Q&A",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f8f9fa;
        border-left: 3px solid #1E3A5F;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .stButton>button {
        background-color: #1E3A5F;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def check_environment():
    """Check if required environment variables are set."""
    required = ["OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"]
    missing = [var for var in required if not os.getenv(var)]
    return missing


def init_qa_chain():
    """Initialize the QA chain."""
    from rag_chatbot.qa_chain import QAChain
    return QAChain()


def set_question(question: str):
    """Callback to set the question in session state."""
    st.session_state.question_input = question


def main():
    # Initialize session state for question input
    if 'question_input' not in st.session_state:
        st.session_state.question_input = ""
    # Header
    st.markdown('<p class="main-header">📊 Prudential 2022 Annual Report Q&A</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about Prudential\'s 2022 Annual Report using AI-powered search</p>', unsafe_allow_html=True)

    # Check environment
    missing_vars = check_environment()
    if missing_vars:
        st.error(f"Missing environment variables: {', '.join(missing_vars)}")
        st.info("Please set these in your .env file and restart the application.")
        st.code("""
# .env file example
OPENAI_API_KEY=your-openai-key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key
        """)
        return

    # Initialize QA chain
    if 'qa_chain' not in st.session_state:
        with st.spinner("Initializing AI assistant..."):
            try:
                st.session_state.qa_chain = init_qa_chain()
            except Exception as e:
                st.error(f"Error initializing: {e}")
                st.info("Make sure Supabase is set up correctly and the index is built.")
                return

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Number of sources
        top_k = st.slider("Number of sources to retrieve", 3, 10, 5)

        # Section filter
        sections = [
            "All Sections",
            "Leadership - Chair's Statement",
            "Leadership - CEO Statement",
            "Strategic Report - Strategy",
            "Financial Review",
            "Risk Management",
            "Sustainability & ESG",
            "Corporate Governance"
        ]
        section_filter = st.selectbox("Filter by section", sections)

        st.divider()

        # Example questions
        st.header("💡 Example Questions")
        example_questions = [
            "What are Prudential's strategic priorities?",
            "How does Prudential manage climate-related risks?",
            "What was the CEO's main message?",
            "How did the Asia segment perform?",
            "What are the key risks facing Prudential?",
            "What is Prudential's approach to ESG?",
            "How is Prudential positioned in Africa?",
            "What are the market challenges?"
        ]

        for q in example_questions:
            st.button(
                q,
                key=f"btn_{q[:20]}",
                use_container_width=True,
                on_click=set_question,
                args=(q,)
            )

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        # Question input - uses session state key directly
        question = st.text_input(
            "Ask a question about the report:",
            placeholder="e.g., What are Prudential's strategic priorities for growth?",
            key="question_input"
        )

        # Submit button
        submit = st.button("🔍 Search", type="primary", use_container_width=True)

        if submit and question:
            with st.spinner("Searching and generating answer..."):
                # Get section filter
                filter_section = None if section_filter == "All Sections" else section_filter

                # Get answer
                result = st.session_state.qa_chain.answer(
                    question=question,
                    top_k=top_k,
                    section_filter=filter_section
                )

                # Display answer
                st.markdown("### 💬 Answer")
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

                # Display sources
                st.markdown("### 📚 Sources")
                for i, source in enumerate(result['sources'], 1):
                    with st.expander(f"Source {i}: Page {source['page']} - {source['section']} (similarity: {source['similarity']})"):
                        st.markdown(source['preview'])

    with col2:
        st.markdown("### 📖 About")
        st.info("""
        This chatbot uses **Retrieval-Augmented Generation (RAG)** to answer questions about Prudential's 2022 Annual Report.

        **How it works:**
        1. Your question is converted to an embedding
        2. Similar passages are retrieved from the report
        3. GPT-4o-mini generates an answer based on the context

        **Tech Stack:**
        - OpenAI Embeddings & GPT-4o-mini
        - Supabase with pgvector
        - Streamlit
        """)

        st.markdown("### 📊 Quick Facts")
        st.metric("Report Year", "2022")
        st.metric("Total Pages", "456")
        st.metric("Embedding Model", "text-embedding-3-small")


if __name__ == "__main__":
    main()
