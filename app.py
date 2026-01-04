"""
10th Standard Science RAG API - Dual Mode
==========================================
Two search modes:
1. Vector DB Search (FAISS + Corrective RAG + Web Fallback)
2. Knowledge Graph Search (Entity-Relation based)

Author: Sathish Suresh
"""

import os
import logging
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import BadRequest

from vector_rag_service import VectorRAGService
from kg_service import KnowledgeGraphService


def app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    # Logging
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log = logging.getLogger("rag_api")

    # Config
    data_dir = Path(os.getenv("DATA_DIR", Path(__file__).parent / "data")).resolve()
    k = int(os.getenv("RAG_K", "5"))
    max_web_urls = int(os.getenv("MAX_WEB_URLS", "3"))

    # Initialize services
    vector_rag = None
    kg_service = None

    # Initialize Vector RAG Service
    try:
        log.info("🔄 Initializing Vector RAG Service...")
        vector_rag = VectorRAGService(
            data_dir=data_dir,
            k=k,
            max_web_urls=max_web_urls,
        )
        log.info("✅ Vector RAG Service initialized")
    except Exception as e:
        log.exception("❌ Failed to initialize Vector RAG: %s", e)

    # Initialize Knowledge Graph Service
    try:
        log.info("🔄 Initializing Knowledge Graph Service...")
        kg_service = KnowledgeGraphService(data_dir=data_dir)
        log.info("✅ Knowledge Graph Service initialized")
    except Exception as e:
        log.exception("❌ Failed to initialize Knowledge Graph: %s", e)

    # ============================================
    # HEALTH CHECK
    # ============================================
    @app.get("/health")
    def health():
        return jsonify({
            "status": "healthy",
            "services": {
                "vector_rag": vector_rag is not None,
                "knowledge_graph": kg_service is not None,
            },
            "data_dir": str(data_dir),
        }), 200

    # ============================================
    # API INFO
    # ============================================
    @app.get("/api/info")
    def api_info():
        return jsonify({
            "api": "10th Standard Science RAG API - Dual Mode",
            "version": "2.0.0",
            "author": "Sathish Suresh",
            "modes": {
                "vector": "Vector DB Search (FAISS + Corrective RAG + Web Fallback)",
                "kg": "Knowledge Graph Search (Entity-Relation based)",
                "hybrid": "Both Vector + Knowledge Graph combined"
            },
            "endpoints": {
                "POST /rag": "Query with mode selection (vector/kg/hybrid)",
                "POST /rag/vector": "Vector DB search only",
                "POST /rag/kg": "Knowledge Graph search only",
                "GET /kg/entities": "List all entities",
                "GET /kg/relations": "List all relations",
                "GET /kg/graph": "Get full graph data for visualization",
                "GET /health": "Health check"
            }
        })

    # ============================================
    # MAIN RAG ENDPOINT (with mode toggle)
    # ============================================
    @app.post("/rag")
    def rag_endpoint():
        """
        Main RAG endpoint with mode selection.
        
        Request Body:
        {
            "query": "What is photosynthesis?",
            "mode": "vector" | "kg" | "hybrid",  // default: "vector"
            "allow_web": true  // only for vector mode
        }
        """
        if vector_rag is None and kg_service is None:
            return jsonify({"error": "No services initialized"}), 503

        payload = request.get_json(silent=True) or {}
        query = (payload.get("query", "") or "").strip()
        mode = (payload.get("mode", "vector") or "vector").lower()
        allow_web = bool(payload.get("allow_web", True))

        if not query:
            raise BadRequest("Query is empty")

        if mode not in ["vector", "kg", "hybrid"]:
            raise BadRequest("Invalid mode. Use: vector, kg, or hybrid")

        log.info(f"Query: '{query[:50]}...' | Mode: {mode}")

        try:
            if mode == "vector":
                # Vector DB Search
                if vector_rag is None:
                    return jsonify({"error": "Vector RAG service not available"}), 503
                
                result = vector_rag.ask(query, allow_web=allow_web)
                return jsonify({
                    "query": query,
                    "mode": "vector",
                    "answer": result.answer,
                    "sources": result.sources,
                    "used_web": result.used_web,
                    "latency_ms": result.latency_ms
                })

            elif mode == "kg":
                # Knowledge Graph Search
                if kg_service is None:
                    return jsonify({"error": "Knowledge Graph service not available"}), 503
                
                result = kg_service.ask(query)
                return jsonify({
                    "query": query,
                    "mode": "kg",
                    "answer": result.answer,
                    "entities_found": result.entities,
                    "relations_found": result.relations,
                    "latency_ms": result.latency_ms
                })

            else:  # hybrid
                # Both Vector + Knowledge Graph
                if vector_rag is None or kg_service is None:
                    return jsonify({"error": "Both services required for hybrid mode"}), 503

                vector_result = vector_rag.ask(query, allow_web=allow_web)
                kg_result = kg_service.ask(query)

                # Combine answers
                combined_answer = f"""📚 **Vector DB Answer:**
{vector_result.answer}

🔗 **Knowledge Graph Insights:**
{kg_result.answer}

**Related Entities:** {', '.join(kg_result.entities[:5]) if kg_result.entities else 'None found'}
"""
                return jsonify({
                    "query": query,
                    "mode": "hybrid",
                    "answer": combined_answer,
                    "vector_answer": vector_result.answer,
                    "kg_answer": kg_result.answer,
                    "sources": vector_result.sources,
                    "entities": kg_result.entities,
                    "relations": kg_result.relations,
                    "used_web": vector_result.used_web,
                    "latency_ms": vector_result.latency_ms + kg_result.latency_ms
                })

        except ValueError as e:
            raise BadRequest(str(e))
        except Exception as e:
            log.exception("RAG failure: %s", e)
            return jsonify({"error": f"Internal error: {str(e)}"}), 500

    # ============================================
    # VECTOR DB ENDPOINT (Direct)
    # ============================================
    @app.post("/rag/vector")
    def vector_endpoint():
        """Direct Vector DB search"""
        if vector_rag is None:
            return jsonify({"error": "Vector RAG service not available"}), 503

        payload = request.get_json(silent=True) or {}
        query = (payload.get("query", "") or "").strip()
        allow_web = bool(payload.get("allow_web", True))

        if not query:
            raise BadRequest("Query is empty")

        try:
            result = vector_rag.ask(query, allow_web=allow_web)
            return jsonify({
                "query": query,
                "answer": result.answer,
                "sources": result.sources,
                "used_web": result.used_web,
                "latency_ms": result.latency_ms
            })
        except ValueError as e:
            raise BadRequest(str(e))
        except Exception as e:
            log.exception("Vector RAG failure: %s", e)
            return jsonify({"error": str(e)}), 500

    # ============================================
    # KNOWLEDGE GRAPH ENDPOINT (Direct)
    # ============================================
    @app.post("/rag/kg")
    def kg_endpoint():
        """Direct Knowledge Graph search"""
        if kg_service is None:
            return jsonify({"error": "Knowledge Graph service not available"}), 503

        payload = request.get_json(silent=True) or {}
        query = (payload.get("query", "") or "").strip()

        if not query:
            raise BadRequest("Query is empty")

        try:
            result = kg_service.ask(query)
            return jsonify({
                "query": query,
                "answer": result.answer,
                "entities": result.entities,
                "relations": result.relations,
                "latency_ms": result.latency_ms
            })
        except ValueError as e:
            raise BadRequest(str(e))
        except Exception as e:
            log.exception("KG failure: %s", e)
            return jsonify({"error": str(e)}), 500

    # ============================================
    # KNOWLEDGE GRAPH - LIST ENTITIES
    # ============================================
    @app.get("/kg/entities")
    def get_entities():
        """Get all entities in Knowledge Graph"""
        if kg_service is None:
            return jsonify({"error": "Knowledge Graph service not available"}), 503

        try:
            entities = kg_service.get_all_entities()
            return jsonify({
                "count": len(entities),
                "entities": entities
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ============================================
    # KNOWLEDGE GRAPH - LIST RELATIONS
    # ============================================
    @app.get("/kg/relations")
    def get_relations():
        """Get all relations in Knowledge Graph"""
        if kg_service is None:
            return jsonify({"error": "Knowledge Graph service not available"}), 503

        try:
            relations = kg_service.get_all_relations()
            return jsonify({
                "count": len(relations),
                "relations": relations
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ============================================
    # KNOWLEDGE GRAPH - FULL GRAPH DATA
    # ============================================
    @app.get("/kg/graph")
    def get_graph():
        """Get full Knowledge Graph for visualization"""
        if kg_service is None:
            return jsonify({"error": "Knowledge Graph service not available"}), 503

        try:
            graph_data = kg_service.get_graph_data()
            return jsonify(graph_data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ============================================
    # KNOWLEDGE GRAPH - QUERY ENTITY
    # ============================================
    @app.get("/kg/entity/<entity_name>")
    def get_entity_info(entity_name):
        """Get information about a specific entity"""
        if kg_service is None:
            return jsonify({"error": "Knowledge Graph service not available"}), 503

        try:
            info = kg_service.get_entity_info(entity_name)
            return jsonify(info)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ============================================
    # ERROR HANDLERS
    # ============================================
    @app.errorhandler(BadRequest)
    def handle_bad_request(e):
        return jsonify({"error": str(e.description)}), 400

    @app.errorhandler(404)
    def handle_not_found(e):
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(500)
    def handle_internal_error(e):
        return jsonify({"error": "Internal server error"}), 500

    return app


if __name__ == "__main__":
    app = app()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
