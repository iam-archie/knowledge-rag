"""
Knowledge Graph Service - Entity-Relation based Search
======================================================
Features:
- Extract entities and relations from PDF documents
- Build in-memory Knowledge Graph (NetworkX)
- Query graph for related information
- Visualize graph structure
- Science Topic Guardrails

Author: Sathish Suresh
"""

import os
import re
import time
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

log = logging.getLogger(__name__)

# Science topics for guardrails
SCIENCE_TOPICS = [
    # Core subjects
    "physics", "chemistry", "biology", "science", "scientist",
    # Physics concepts
    "gravity", "force", "motion", "energy", "work", "power",
    "light", "sound", "wave", "electricity", "current", "voltage", "resistance",
    "ohm", "newton", "kepler", "law", "equation", "formula",
    "lens", "mirror", "refraction", "reflection", "spectrum",
    "heat", "temperature", "thermometer", "conduction", "convection",
    "magnetism", "magnetic", "field", "electromagnetic",
    "lenz", "faraday", "induction", "emf", "flux", "solenoid", "coil",
    "transformer", "generator", "motor", "alternating", "direct",
    "ampere", "coulomb", "joule", "watt", "hertz",
    # Chemistry concepts
    "atom", "molecule", "element", "compound", "reaction", "acid", "base", "salt",
    "metal", "carbon", "periodic", "table", "electron", "proton", "neutron",
    "oxidation", "reduction", "redox", "electrolysis", "corrosion",
    "polymer", "hydrocarbon", "alcohol", "ester", "soap", "detergent",
    # Biology concepts
    "cell", "organism", "plant", "animal", "photosynthesis", "respiration",
    "chromosome", "dna", "gene", "heredity", "evolution", "ecosystem",
    "hormone", "enzyme", "digestion", "circulation", "excretion",
    "nervous", "brain", "reproduction",
    # Space science
    "space", "planet", "sun", "moon", "orbit", "satellite", "star",
    "gravitation", "metallurgy", "lenses", "missions",
    # Textbook related (IMPORTANT!)
    "chapter", "unit", "lesson", "topic", "page",
    "summarize", "summary", "explain", "describe", "define",
    "fill", "blank", "blanks", "match", "following", "answer",
    "what is", "how does", "why", "who", "when", "which",
    "definition", "example", "process", "structure", "function",
    "list", "name", "state", "give", "write", "draw",
    # Famous scientists
    "hawking", "einstein", "newton", "faraday", "bohr", "rutherford",
    "mendeleev", "dalton", "curie", "galileo", "archimedes",
    "born", "discovered", "invented", "theory", "experiment",
    # General textbook terms
    "textbook", "book", "10th", "tenth", "standard", "class",
    "index", "content", "exercise", "question", "mcq"
]


@dataclass
class KGResult:
    """Result from Knowledge Graph query"""
    answer: str
    entities: List[str]
    relations: List[Dict[str, str]]
    latency_ms: int
    blocked: bool = False
    block_reason: str = ""


class KnowledgeGraphService:
    """
    Knowledge Graph Service for Science Q&A
    
    Flow:
    1. Load PDFs -> Extract text chunks
    2. Use LLM to extract entities and relations
    3. Build NetworkX graph
    4. On query: Find relevant entities -> Get subgraph -> Generate answer
    """

    def __init__(
        self,
        data_dir: Path,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        chat_model: str = "gpt-4o-mini",
        temperature: float = 0.1,
    ):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")

        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Text Splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # LLM
        self.llm = ChatOpenAI(model=chat_model, temperature=temperature)

        # Prompts for entity/relation extraction
        self.extract_prompt = ChatPromptTemplate.from_template(
            """Extract scientific entities and their relationships from this text.

TEXT:
{text}

Return a JSON object with:
{{
    "entities": [
        {{"name": "entity_name", "type": "concept|process|formula|unit|chapter|person"}},
        ...
    ],
    "relations": [
        {{"source": "entity1", "relation": "relation_type", "target": "entity2"}},
        ...
    ]
}}

IMPORTANT RULES:
- Entity names must be at least 2 words or meaningful single words (NOT single letters like 'e', 's', 'a', 'g')
- Use full names: "acceleration due to gravity" not "g"
- Use "photosynthesis" not "p" 
- Use "electric current" not "I"
- Relations should connect meaningful concepts

Focus on:
- Scientific concepts (photosynthesis, gravity, Ohm's law, etc.)
- Processes and their steps
- Formulas and equations (use full names)
- Units and measurements
- Cause-effect relationships
- Part-of relationships

Return ONLY valid JSON, no other text."""
        )

        self.query_prompt = ChatPromptTemplate.from_template(
            """Answer the question using the knowledge graph information below.

ENTITIES AND RELATIONS FROM KNOWLEDGE GRAPH:
{kg_context}

QUESTION: {question}

Provide a clear answer based on the entities and relations. If the information is not in the graph, use your knowledge to provide a helpful response.

ANSWER:"""
        )

        self.entity_match_prompt = ChatPromptTemplate.from_template(
            """Given this question, identify the main scientific concepts/entities to search for.

QUESTION: {question}

AVAILABLE ENTITIES (sample):
{sample_entities}

Return a JSON list of entity names that are most relevant to answer this question.
Return ONLY the JSON list, no other text.

Example: ["photosynthesis", "chlorophyll", "sunlight"]"""
        )
        
        # Fallback prompt when no entities found
        self.fallback_prompt = ChatPromptTemplate.from_template(
            """You are a 10th Standard Science tutor. Answer this question clearly.

Handle different question types:
- For "summarize chapter X": Provide a summary of that chapter's typical content
- For "fill in the blanks": Complete the sentence with the correct scientific term
- For questions about scientists: Provide biographical info and contributions
- For explaining concepts: Give clear, student-friendly explanations

Question: {q}

Provide a clear, educational answer:"""
        )
        
        # Guardrails prompt
        self.guardrails_prompt = ChatPromptTemplate.from_template(
            """Is this question related to a 10th standard Science textbook?

This includes:
- Direct science topics: Physics, Chemistry, Biology, Space Science
- Textbook questions: fill in blanks, summarize chapter, exercises
- Scientists and their discoveries
- Formulas, laws, experiments
- Any question that might appear in a science textbook

Question: {q}

Reply with ONLY one word: YES or NO
- YES = Related to science or textbook content
- NO = Completely unrelated (politics, entertainment, current affairs)"""
        )

        # Build Knowledge Graph at startup
        self.graph = nx.DiGraph()
        self._entity_index = {}  # For fast lookup
        self._build_knowledge_graph()

        log.info(f"KnowledgeGraphService initialized with {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def _check_science_topic(self, query: str) -> bool:
        """
        Guardrails: Check if query is related to science
        More lenient for textbook-related questions
        """
        query_lower = query.lower()
        
        # Quick keyword check first - if ANY science keyword found, allow it
        for topic in SCIENCE_TOPICS:
            if topic in query_lower:
                log.info(f"Guardrails PASS: Found keyword '{topic}'")
                return True
        
        # Check for chapter-related queries (always allow)
        chapter_patterns = ["chapter", "unit", "lesson", "summarize", "summary", 
                          "fill in", "blank", "match the", "exercise", "question",
                          "textbook", "book", "page"]
        for pattern in chapter_patterns:
            if pattern in query_lower:
                log.info(f"Guardrails PASS: Textbook query pattern '{pattern}'")
                return True
        
        # Check for numbers (likely chapter/page references)
        if re.search(r'\d+(st|nd|rd|th)?\s*(chapter|unit|lesson|page)?', query_lower):
            log.info("Guardrails PASS: Contains number reference")
            return True
        
        # If no keyword match, use LLM to verify
        try:
            verdict = self.llm.invoke(
                self.guardrails_prompt.format_messages(q=query)
            ).content.strip().upper()
            result = verdict.startswith("YES")
            log.info(f"Guardrails LLM check: {verdict} -> {result}")
            return result
        except Exception as e:
            log.warning(f"Guardrails check failed: {e}")
            return True

    def _load_documents(self) -> List[Any]:
        """Load all PDFs from data directory"""
        if not self.data_dir.exists():
            raise RuntimeError(f"DATA_DIR not found: {self.data_dir}")

        docs = DirectoryLoader(
            str(self.data_dir),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        ).load()

        if not docs:
            raise RuntimeError(f"No PDFs found in {self.data_dir}")

        chunks = self.splitter.split_documents(docs)
        log.info(f"Loaded {len(docs)} PDF pages -> {len(chunks)} chunks for KG extraction")
        
        return chunks

    def _extract_entities_relations(self, text: str) -> Dict[str, List]:
        """Use LLM to extract entities and relations from text"""
        try:
            response = self.llm.invoke(
                self.extract_prompt.format_messages(text=text[:3000])
            ).content
            
            # Parse JSON response
            # Handle potential markdown code blocks
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            response = response.strip()
            
            data = json.loads(response)
            return {
                "entities": data.get("entities", []),
                "relations": data.get("relations", [])
            }
        except json.JSONDecodeError as e:
            log.warning(f"JSON parse error in extraction: {e}")
            return {"entities": [], "relations": []}
        except Exception as e:
            log.warning(f"Extraction failed: {e}")
            return {"entities": [], "relations": []}

    def _build_knowledge_graph(self):
        """Build Knowledge Graph from PDF documents"""
        log.info("Building Knowledge Graph from documents...")
        
        chunks = self._load_documents()
        
        # Process chunks (sample for efficiency)
        # For large documents, process every Nth chunk
        sample_rate = max(1, len(chunks) // 50)  # ~50 chunks max
        sampled_chunks = chunks[::sample_rate]
        
        log.info(f"Processing {len(sampled_chunks)} chunks for KG extraction...")
        
        all_entities = []
        all_relations = []
        
        for i, chunk in enumerate(sampled_chunks):
            if i % 10 == 0:
                log.info(f"Processing chunk {i+1}/{len(sampled_chunks)}...")
            
            result = self._extract_entities_relations(chunk.page_content)
            all_entities.extend(result["entities"])
            all_relations.extend(result["relations"])
        
        # Add entities to graph (filter out noise)
        for entity in all_entities:
            name = entity.get("name", "").lower().strip()
            # Filter: must be at least 2 chars and not just numbers/symbols
            if name and len(name) >= 2 and any(c.isalpha() for c in name):
                entity_type = entity.get("type", "concept")
                
                if name not in self._entity_index:
                    self.graph.add_node(name, type=entity_type, count=1)
                    self._entity_index[name] = entity_type
                else:
                    # Increment count for existing entity
                    if self.graph.has_node(name):
                        self.graph.nodes[name]["count"] = self.graph.nodes[name].get("count", 1) + 1
        
        # Add relations to graph (filter out noise)
        for rel in all_relations:
            source = rel.get("source", "").lower().strip()
            target = rel.get("target", "").lower().strip()
            relation = rel.get("relation", "related_to").lower().strip()
            
            # Filter: both source and target must be valid (2+ chars, has letters)
            if (source and target and source != target and 
                len(source) >= 2 and len(target) >= 2 and
                any(c.isalpha() for c in source) and any(c.isalpha() for c in target)):
                # Ensure nodes exist
                if source not in self._entity_index:
                    self.graph.add_node(source, type="concept", count=1)
                    self._entity_index[source] = "concept"
                if target not in self._entity_index:
                    self.graph.add_node(target, type="concept", count=1)
                    self._entity_index[target] = "concept"
                
                # Add edge
                if self.graph.has_edge(source, target):
                    # Add relation type to existing edge
                    existing = self.graph.edges[source, target].get("relations", [])
                    if relation not in existing:
                        existing.append(relation)
                        self.graph.edges[source, target]["relations"] = existing
                else:
                    self.graph.add_edge(source, target, relations=[relation])
        
        log.info(f"Knowledge Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def _find_relevant_entities(self, query: str) -> List[str]:
        """Find entities relevant to the query"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Direct match - but only for entities with 3+ characters
        # This avoids matching single letters like 'e', 's', 'a'
        matches = []
        for entity in self._entity_index.keys():
            # Skip very short entities (likely noise)
            if len(entity) < 3:
                continue
            
            # Check if entity appears as whole word in query
            # or query word appears in entity
            entity_words = set(entity.split())
            
            # Full entity match in query
            if entity in query_lower:
                matches.append(entity)
            # Any entity word matches query word
            elif entity_words & query_words:
                matches.append(entity)
        
        # If no direct matches, use LLM to identify entities
        if not matches:
            try:
                # Filter out short/noisy entities for sample
                good_entities = [e for e in self._entity_index.keys() if len(e) >= 3]
                sample_entities = good_entities[:100]
                
                response = self.llm.invoke(
                    self.entity_match_prompt.format_messages(
                        question=query,
                        sample_entities=", ".join(sample_entities)
                    )
                ).content.strip()
                
                # Parse response
                if response.startswith("["):
                    entities = json.loads(response)
                    for e in entities:
                        e_lower = e.lower().strip()
                        if e_lower in self._entity_index and len(e_lower) >= 3:
                            matches.append(e_lower)
            except Exception as e:
                log.warning(f"Entity matching failed: {e}")
        
        return matches[:10]  # Limit to top 10

    def _get_subgraph_context(self, entities: List[str], depth: int = 2) -> str:
        """Get context from subgraph around entities"""
        if not entities:
            return "No relevant entities found in knowledge graph."
        
        # Collect nodes within depth
        relevant_nodes = set(entities)
        for entity in entities:
            if self.graph.has_node(entity):
                # Get neighbors
                for neighbor in self.graph.neighbors(entity):
                    relevant_nodes.add(neighbor)
                # Get predecessors
                for pred in self.graph.predecessors(entity):
                    relevant_nodes.add(pred)
        
        # Build context string
        context_parts = []
        
        # Add entity info (filter short ones)
        context_parts.append("ENTITIES:")
        for node in list(relevant_nodes)[:20]:
            if len(node) >= 3 and self.graph.has_node(node):
                node_data = self.graph.nodes[node]
                context_parts.append(f"  - {node} (type: {node_data.get('type', 'unknown')})")
        
        # Add relation info (filter short ones)
        context_parts.append("\nRELATIONS:")
        for node in list(relevant_nodes)[:15]:
            if len(node) >= 3 and self.graph.has_node(node):
                for neighbor in list(self.graph.neighbors(node))[:5]:
                    if len(neighbor) >= 3:
                        edge_data = self.graph.edges[node, neighbor]
                        relations = edge_data.get("relations", ["related_to"])
                        for rel in relations[:2]:
                            context_parts.append(f"  - {node} --[{rel}]--> {neighbor}")
        
        return "\n".join(context_parts)

    def ask(self, query: str) -> KGResult:
        """
        Query the Knowledge Graph with Guardrails
        
        Args:
            query: Question to answer
            
        Returns:
            KGResult with answer, entities, relations
        """
        t0 = time.time()
        query = (query or "").strip()
        
        if not query:
            raise ValueError("Query is empty")

        # Step 0: Guardrails - Check if science topic
        if not self._check_science_topic(query):
            latency_ms = int((time.time() - t0) * 1000)
            return KGResult(
                answer="❌ I can only answer questions related to 10th Standard Science topics (Physics, Chemistry, Biology). Please ask a science-related question!",
                entities=[],
                relations=[],
                latency_ms=latency_ms,
                blocked=True,
                block_reason="Non-science question"
            )

        # Step 1: Find relevant entities
        entities = self._find_relevant_entities(query)
        log.info(f"Found {len(entities)} relevant entities: {entities[:5]}")

        # Step 2: Get subgraph context
        kg_context = self._get_subgraph_context(entities)

        # Step 3: Generate answer
        # Check if we have meaningful entities
        has_good_entities = entities and len(entities) > 0
        
        if has_good_entities and "No relevant entities" not in kg_context:
            answer = self.llm.invoke(
                self.query_prompt.format_messages(
                    kg_context=kg_context,
                    question=query
                )
            ).content
        else:
            # Fallback to LLM knowledge
            log.info("Using LLM knowledge fallback (no entities found in KG)")
            answer = self.llm.invoke(
                self.fallback_prompt.format_messages(q=query)
            ).content
            answer = f"{answer}\n\n📝 *Note: This topic is not in the Knowledge Graph, answer provided from AI knowledge.*"

        # Step 4: Collect relations for response (filter short ones)
        relations = []
        for entity in entities[:5]:
            if len(entity) >= 3 and self.graph.has_node(entity):
                for neighbor in list(self.graph.neighbors(entity))[:3]:
                    if len(neighbor) >= 3:
                        edge_data = self.graph.edges[entity, neighbor]
                        for rel in edge_data.get("relations", ["related_to"])[:1]:
                            relations.append({
                                "source": entity,
                                "relation": rel,
                                "target": neighbor
                            })

        # Filter entities returned (remove short ones)
        filtered_entities = [e for e in entities if len(e) >= 3]

        latency_ms = int((time.time() - t0) * 1000)
        
        return KGResult(
            answer=answer,
            entities=filtered_entities,
            relations=relations,
            latency_ms=latency_ms,
            blocked=False,
            block_reason=""
        )

    def get_all_entities(self) -> List[Dict[str, Any]]:
        """Get all entities in the Knowledge Graph (filtered)"""
        entities = []
        for node in self.graph.nodes():
            # Filter out short/noisy entities
            if len(node) < 3:
                continue
            node_data = self.graph.nodes[node]
            entities.append({
                "name": node,
                "type": node_data.get("type", "unknown"),
                "count": node_data.get("count", 1),
                "connections": self.graph.degree(node)
            })
        
        # Sort by connections (most connected first)
        entities.sort(key=lambda x: x["connections"], reverse=True)
        return entities

    def get_all_relations(self) -> List[Dict[str, Any]]:
        """Get all relations in the Knowledge Graph (filtered)"""
        relations = []
        for source, target, data in self.graph.edges(data=True):
            # Filter out relations with short entities
            if len(source) < 3 or len(target) < 3:
                continue
            for rel in data.get("relations", ["related_to"]):
                relations.append({
                    "source": source,
                    "relation": rel,
                    "target": target
                })
        return relations

    def get_graph_data(self) -> Dict[str, Any]:
        """Get full graph data for visualization"""
        nodes = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            nodes.append({
                "id": node,
                "label": node,
                "type": node_data.get("type", "concept"),
                "count": node_data.get("count", 1)
            })
        
        edges = []
        for source, target, data in self.graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "relations": data.get("relations", ["related_to"])
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges()
            }
        }

    def get_entity_info(self, entity_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific entity"""
        entity_name = entity_name.lower().strip()
        
        if not self.graph.has_node(entity_name):
            return {"error": f"Entity '{entity_name}' not found"}
        
        node_data = self.graph.nodes[entity_name]
        
        # Get outgoing relations
        outgoing = []
        for neighbor in self.graph.neighbors(entity_name):
            edge_data = self.graph.edges[entity_name, neighbor]
            outgoing.append({
                "target": neighbor,
                "relations": edge_data.get("relations", [])
            })
        
        # Get incoming relations
        incoming = []
        for pred in self.graph.predecessors(entity_name):
            edge_data = self.graph.edges[pred, entity_name]
            incoming.append({
                "source": pred,
                "relations": edge_data.get("relations", [])
            })
        
        return {
            "name": entity_name,
            "type": node_data.get("type", "unknown"),
            "count": node_data.get("count", 1),
            "outgoing_relations": outgoing,
            "incoming_relations": incoming,
            "total_connections": len(outgoing) + len(incoming)
        }
