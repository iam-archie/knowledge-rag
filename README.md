# 🔬 10th Standard Science RAG API - Dual Mode

Production-ready REST API with **two search modes**:
1. **Vector DB Search** - FAISS + BM25 Hybrid with Corrective RAG
2. **Knowledge Graph Search** - Entity-Relation based reasoning

---

## 🚀 Features

### Vector DB Mode
- 📄 PDF Document Loading
- 🔍 FAISS Vector Search
- 📝 BM25 Keyword Search
- 🔀 Hybrid Search (Vector + BM25 combined)
- ✅ Corrective RAG (grades context, falls back to web)
- 🌐 Web Search Fallback

### Knowledge Graph Mode
- 🔗 Entity Extraction from Documents
- 📊 Relation Mapping
- 🕸️ NetworkX Graph Storage
- 🎯 Entity-based Query Resolution
- 📈 Graph Visualization Data

### Hybrid Mode
- 🔄 Combines both Vector DB + Knowledge Graph
- 📚 Best of both worlds!

---

## 📦 Installation

```bash
# Clone/Download the project
cd science_rag_dual

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Add your PDF files
cp your_science_book.pdf data/

# Run the server
python app.py
```

---

## 📡 API Endpoints

### Health Check
```http
GET /health
```

### API Info
```http
GET /api/info
```

### Main RAG Endpoint (with mode toggle)
```http
POST /rag
Content-Type: application/json

{
    "query": "What is photosynthesis?",
    "mode": "vector",      // "vector" | "kg" | "hybrid"
    "allow_web": true      // only for vector mode
}
```

### Vector DB Search (Direct)
```http
POST /rag/vector
Content-Type: application/json

{
    "query": "What is Ohm's law?",
    "allow_web": true
}
```

### Knowledge Graph Search (Direct)
```http
POST /rag/kg
Content-Type: application/json

{
    "query": "What is photosynthesis?"
}
```

### Knowledge Graph - List Entities
```http
GET /kg/entities
```

### Knowledge Graph - List Relations
```http
GET /kg/relations
```

### Knowledge Graph - Full Graph Data
```http
GET /kg/graph
```

### Knowledge Graph - Entity Info
```http
GET /kg/entity/{entity_name}
```

---

## 🧪 Usage Examples

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Vector DB search
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What is gravity?", "mode": "vector"}'

# Knowledge Graph search
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What is gravity?", "mode": "kg"}'

# Hybrid search (both!)
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What is gravity?", "mode": "hybrid"}'

# List all entities
curl http://localhost:8000/kg/entities

# Get entity info
curl http://localhost:8000/kg/entity/photosynthesis
```

### Python

```python
import requests

API = "http://localhost:8000"

# Vector DB Search
response = requests.post(f"{API}/rag", json={
    "query": "What is photosynthesis?",
    "mode": "vector"
})
print(response.json())

# Knowledge Graph Search
response = requests.post(f"{API}/rag", json={
    "query": "What is photosynthesis?",
    "mode": "kg"
})
print(response.json())

# Hybrid Search
response = requests.post(f"{API}/rag", json={
    "query": "What is photosynthesis?",
    "mode": "hybrid"
})
print(response.json())

# Get all entities
entities = requests.get(f"{API}/kg/entities").json()
print(f"Total entities: {entities['count']}")

# Get graph data for visualization
graph = requests.get(f"{API}/kg/graph").json()
print(f"Nodes: {graph['stats']['total_nodes']}")
print(f"Edges: {graph['stats']['total_edges']}")
```

---

## 📊 Response Examples

### Vector DB Response
```json
{
    "query": "What is photosynthesis?",
    "mode": "vector",
    "answer": "Photosynthesis is the process by which green plants...",
    "sources": ["10th_science.pdf"],
    "used_web": false,
    "latency_ms": 1250
}
```

### Knowledge Graph Response
```json
{
    "query": "What is photosynthesis?",
    "mode": "kg",
    "answer": "Based on the knowledge graph, photosynthesis involves...",
    "entities_found": ["photosynthesis", "chlorophyll", "sunlight", "carbon dioxide"],
    "relations_found": [
        {"source": "photosynthesis", "relation": "requires", "target": "sunlight"},
        {"source": "photosynthesis", "relation": "produces", "target": "glucose"}
    ],
    "latency_ms": 890
}
```

### Hybrid Response
```json
{
    "query": "What is photosynthesis?",
    "mode": "hybrid",
    "answer": "📚 **Vector DB Answer:**\nPhotosynthesis is...\n\n🔗 **Knowledge Graph Insights:**\nBased on the graph...",
    "vector_answer": "...",
    "kg_answer": "...",
    "sources": ["10th_science.pdf"],
    "entities": ["photosynthesis", "chlorophyll"],
    "relations": [...],
    "used_web": false,
    "latency_ms": 2100
}
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FLASK API LAYER                        │
│  POST /rag?mode=vector|kg|hybrid                           │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  VECTOR MODE    │  │    KG MODE      │  │  HYBRID MODE    │
│                 │  │                 │  │                 │
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ Vector + KG     │
│ │ FAISS Index │ │  │ │ NetworkX    │ │  │ Combined!       │
│ └─────────────┘ │  │ │ Graph       │ │  │                 │
│ ┌─────────────┐ │  │ └─────────────┘ │  │                 │
│ │ BM25 Index  │ │  │ ┌─────────────┐ │  │                 │
│ └─────────────┘ │  │ │ Entity Index│ │  │                 │
│ ┌─────────────┐ │  │ └─────────────┘ │  │                 │
│ │Corrective   │ │  │                 │  │                 │
│ │RAG + Web    │ │  │                 │  │                 │
│ └─────────────┘ │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                   │                    │
         └───────────────────┴────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   OpenAI LLM    │
                    │   (gpt-4o-mini) │
                    └─────────────────┘
```

---

## 📁 Project Structure

```
science_rag_dual/
├── app.py                    # Flask API (main entry)
├── vector_rag_service.py     # Vector DB RAG logic
├── kg_service.py             # Knowledge Graph logic
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── README.md                 # This file
└── data/
    └── *.pdf                 # Your PDF files here
```

---

## ⚙️ Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `PORT` | Server port | 8000 |
| `LOG_LEVEL` | Logging level | INFO |
| `RAG_K` | Number of documents to retrieve | 5 |
| `MAX_WEB_URLS` | Max web URLs for fallback | 3 |
| `DATA_DIR` | Path to PDF directory | ./data |

---

## 🔍 How It Works

### Vector DB Mode
1. Load PDFs → Split into chunks
2. Create FAISS vector index + BM25 keyword index
3. On query: Hybrid search (combine vector + BM25)
4. Grade context relevance (Corrective RAG)
5. If insufficient → Search web for more info
6. Generate answer from combined context

### Knowledge Graph Mode
1. Load PDFs → Split into chunks
2. Use LLM to extract entities & relations from each chunk
3. Build NetworkX directed graph
4. On query: Find relevant entities → Get subgraph
5. Generate answer from graph structure

### Hybrid Mode
1. Run both Vector DB and KG queries
2. Combine results into comprehensive answer
3. Return entities, relations, AND document sources

---

## 📝 License

MIT License - Sathish Suresh

---

## 🙏 Credits

- LangChain
- OpenAI
- FAISS
- NetworkX
- Flask
- Social Eagle AI Community
