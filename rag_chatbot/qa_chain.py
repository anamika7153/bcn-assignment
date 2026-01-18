"""
Q&A Chain Module
Implements the RAG pipeline for question answering.
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

from rag_chatbot.supabase_store import SupabaseVectorStore

# Load environment variables
load_dotenv()


class QAChain:
    """RAG-based Question Answering Chain."""

    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize the QA chain."""
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.vector_store = SupabaseVectorStore()
        self.model = model

        self.system_prompt = """You are a helpful assistant that answers questions about Prudential's 2022 Annual Report.

IMPORTANT INSTRUCTIONS:
1. Only answer based on the provided context from the report
2. If the context doesn't contain enough information, say so
3. Always cite the page number when referencing specific information
4. Be concise but thorough
5. Use bullet points for lists
6. Include specific numbers and percentages when available

Format your response as:
- Direct answer to the question
- Supporting details from the context
- Page references in parentheses (e.g., "Page 34")"""

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        section_filter: Optional[str] = None
    ) -> List[Dict]:
        """Retrieve relevant documents for a query."""
        filter_metadata = {}
        if section_filter:
            filter_metadata["section"] = section_filter

        results = self.vector_store.search(
            query=query,
            match_count=top_k,
            filter_metadata=filter_metadata if filter_metadata else None
        )
        return results

    def format_context(self, documents: List[Dict]) -> str:
        """Format retrieved documents into context string."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            page = doc['metadata'].get('page', 'N/A')
            section = doc['metadata'].get('section', 'N/A')
            content = doc['content']

            context_parts.append(
                f"[Source {i}] (Page {page}, Section: {section})\n{content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def answer(
        self,
        question: str,
        top_k: int = 5,
        section_filter: Optional[str] = None
    ) -> Dict:
        """
        Answer a question using RAG.

        Args:
            question: The user's question
            top_k: Number of documents to retrieve
            section_filter: Optional section to filter by

        Returns:
            Dict with answer, sources, and metadata
        """
        # Retrieve relevant documents
        documents = self.retrieve(question, top_k, section_filter)

        if not documents:
            return {
                "answer": "I couldn't find relevant information in the report to answer this question.",
                "sources": [],
                "context_used": ""
            }

        # Format context
        context = self.format_context(documents)

        # Generate answer
        user_prompt = f"""Based on the following context from Prudential's 2022 Annual Report, please answer the question.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

        response = self.openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )

        answer = response.choices[0].message.content

        # Format sources
        sources = [
            {
                "page": doc['metadata'].get('page'),
                "section": doc['metadata'].get('section'),
                "similarity": round(doc['similarity'], 3),
                "preview": doc['content'][:150] + "..."
            }
            for doc in documents
        ]

        return {
            "answer": answer,
            "sources": sources,
            "context_used": context,
            "model": self.model
        }


def main():
    """Test the QA chain with example questions."""
    print("=" * 60)
    print("Q&A Chain Test")
    print("=" * 60)

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        return

    if not os.getenv("SUPABASE_URL"):
        print("Error: SUPABASE_URL not set")
        return

    # Initialize chain
    print("\nInitializing QA chain...")
    qa = QAChain()

    # Test questions
    test_questions = [
        "What are Prudential's strategic priorities?",
        "How did the Asia segment perform in 2022?",
        "What are the main risks facing Prudential?",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print("=" * 60)

        result = qa.answer(question)

        print(f"\nA: {result['answer']}")
        print(f"\nSources used: {len(result['sources'])}")
        for i, source in enumerate(result['sources'][:3], 1):
            print(f"  {i}. Page {source['page']} ({source['section']}) - sim: {source['similarity']}")


if __name__ == "__main__":
    main()
