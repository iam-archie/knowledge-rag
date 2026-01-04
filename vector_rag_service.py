"""
Vector RAG Service - FAISS + BM25 Hybrid Search with Corrective RAG
===================================================================
Features:
- PDF Document Loading
- FAISS Vector Store
- BM25 Keyword Search
- Hybrid Search (Vector + BM25)
- Corrective RAG (grade context, fallback to web)
- Web Search Fallback

Author: Sathish Suresh
"""

import os
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults

log = logging.getLogger(__name__)


@dataclass
class VectorRagResult:
    """Result from Vector RAG query"""
    answer: str
    sources: List[str]
    used_web: bool
    latency_ms: int


class VectorRAGService:
    """
    Vector DB RAG Service with Corrective RAG
    
    Flow:
    1. Load PDFs -> Create chunks
    2. Build FAISS Vector Store + BM25 Index
    3. On query: Hybrid search (Vector + BM25)
    4. Grade if context enough
    5. If NO -> Add web search results
    6. Generate answer from context
    """

    def __init__(
        self,
        data_dir: Path,
        k: int = 5,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_web_urls: int = 3,
    ):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")

        self.data_dir = data_dir
        self.k = k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_web_urls = max_web_urls

        # Text Splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

        # Embeddings & LLM
        self.emb = OpenAIEmbeddings(model=embedding_model)
        self.llm = ChatOpenAI(model=chat_model, temperature=temperature)

        # Prompts
        self.grade_prompt = ChatPromptTemplate.from_template(
            """You are checking if the context contains enough information to answer the question.
            
Question: {q}

Context:
{context}

Reply with ONLY one word: YES or NO
- YES = context has relevant information to answer
- NO = context is missing or irrelevant"""
        )

        self.answer_prompt = ChatPromptTemplate.from_template(
            """Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {q}

Answer:"""
        )

        # Web Search Tool
        self.search = DuckDuckGoSearchResults(output_format="list")

        # Build Vector Store and BM25 at startup
        self._chunks = self._load_documents()
        self._vs = self._build_vector_store()
        self._bm25 = self._build_bm25_retriever()

        log.info(f"VectorRAGService initialized with {len(self._chunks)} chunks")

    def _load_documents(self) -> List[Any]:
        """Load and chunk all PDFs from data directory"""
        if not self.data_dir.exists():
            raise RuntimeError(f"DATA_DIR not found: {self.data_dir}")

        # Load PDFs
        docs = DirectoryLoader(
            str(self.data_dir),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        ).load()

        if not docs:
            raise RuntimeError(f"No PDFs found in {self.data_dir}")

        # Split into chunks
        chunks = self.splitter.split_documents(docs)
        log.info(f"Loaded {len(docs)} PDF pages -> {len(chunks)} chunks")
        
        return chunks

    def _build_vector_store(self) -> FAISS:
        """Build FAISS vector store from chunks"""
        log.info("Building FAISS vector store...")
        vs = FAISS.from_documents(self._chunks, self.emb)
        log.info("FAISS vector store built")
        return vs

    def _build_bm25_retriever(self) -> BM25Retriever:
        """Build BM25 retriever for keyword search"""
        log.info("Building BM25 retriever...")
        bm25 = BM25Retriever.from_documents(self._chunks)
        bm25.k = self.k
        log.info("BM25 retriever built")
        return bm25

    def _hybrid_search(self, query: str) -> List[Any]:
        """
        Hybrid search: combine Vector + BM25 results
        Returns deduplicated documents
        """
        # Vector search
        vector_docs = self._vs.similarity_search(query, k=self.k)
        
        # BM25 search
        bm25_docs = self._bm25.invoke(query)
        
        # Combine and deduplicate
        all_docs = vector_docs + bm25_docs
        seen = set()
        unique_docs = []
        
        for doc in all_docs:
            content_key = doc.page_content[:100]
            if content_key not in seen:
                seen.add(content_key)
                unique_docs.append(doc)
        
        return unique_docs[:self.k * 2]  # Return more for better coverage

    def _is_context_enough(self, q: str, docs: List[Any]) -> bool:
        """Grade if retrieved context is sufficient to answer"""
        if not docs:
            return False
            
        context = "\n\n".join(d.page_content[:800] for d in docs[:5])
        
        try:
            verdict = (
                self.llm.invoke(self.grade_prompt.format_messages(q=q, context=context))
                .content.strip().upper()
            )
            return verdict.startswith("YES")
        except Exception as e:
            log.warning(f"Grading failed: {e}")
            return True  # Assume enough if grading fails

    def _retrieve_web(self, q: str) -> Tuple[List[Any], List[str]]:
        """
        Retrieve web documents for additional context
        Returns: (web_chunks, urls_used)
        """
        try:
            results = self.search.invoke(f"10th standard science {q}") or []
            urls = []
            
            for r in results:
                link = r.get("link")
                if link and link not in urls:
                    urls.append(link)
                if len(urls) >= self.max_web_urls:
                    break

            if not urls:
                return [], []

            # Load web pages
            web_docs = WebBaseLoader(urls).load()
            web_chunks = self.splitter.split_documents(web_docs)
            
            log.info(f"Retrieved {len(web_chunks)} chunks from {len(urls)} web URLs")
            return web_chunks, urls
            
        except Exception as e:
            log.warning(f"Web retrieval failed: {e}")
            return [], []

    @staticmethod
    def _format_context(docs: List[Any]) -> str:
        """Format documents into context string"""
        return "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in docs
        )

    @staticmethod
    def _extract_sources(docs: List[Any]) -> List[str]:
        """Extract unique source names from documents"""
        seen = set()
        sources = []
        for d in docs:
            s = d.metadata.get("source", "unknown")
            # Simplify source name
            if "/" in s:
                s = s.split("/")[-1]
            if s not in seen:
                seen.add(s)
                sources.append(s)
        return sources

    def ask(self, q: str, allow_web: bool = True) -> VectorRagResult:
        """
        Main query method
        
        Args:
            q: Question to answer
            allow_web: Whether to allow web search fallback
            
        Returns:
            VectorRagResult with answer, sources, etc.
        """
        t0 = time.time()
        q = (q or "").strip()
        
        if not q:
            raise ValueError("Query is empty")

        # Step 1: Hybrid search (Vector + BM25)
        local_hits = self._hybrid_search(q)
        log.info(f"Retrieved {len(local_hits)} local documents")

        used_web = False
        hits = local_hits

        # Step 2: Corrective RAG - Check if context is enough
        if allow_web and not self._is_context_enough(q, local_hits):
            log.info("Context insufficient, fetching from web...")
            web_hits, urls = self._retrieve_web(q)
            
            if web_hits:
                used_web = True
                # Combine local + web results
                combined_docs = list(local_hits) + list(web_hits)
                
                # Re-rank combined results
                combined_vs = FAISS.from_documents(combined_docs, self.emb)
                hits = combined_vs.similarity_search(q, k=self.k)
                log.info(f"Combined search returned {len(hits)} documents")

        # Step 3: Generate answer
        context = self._format_context(hits)
        answer = self.llm.invoke(
            self.answer_prompt.format_messages(q=q, context=context)
        ).content

        latency_ms = int((time.time() - t0) * 1000)
        
        return VectorRagResult(
            answer=answer,
            sources=self._extract_sources(hits),
            used_web=used_web,
            latency_ms=latency_ms,
        )
