"""
Science RAG API - Dual Mode Test Client
=======================================
Test script to verify both Vector DB and Knowledge Graph modes

Usage: python test_client.py
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def print_response(title: str, response):
    """Pretty print API response"""
    print(f"\n{'='*60}")
    print(f"📌 {title}")
    print(f"{'='*60}")
    print(f"Status: {response.status_code}")
    try:
        data = response.json()
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except:
        print(response.text)


def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print_response("Health Check", response)
    return response.status_code == 200


def test_api_info():
    """Test API info endpoint"""
    response = requests.get(f"{BASE_URL}/api/info")
    print_response("API Info", response)
    return response.status_code == 200


def test_vector_mode():
    """Test Vector DB mode"""
    response = requests.post(
        f"{BASE_URL}/rag",
        json={
            "query": "What is photosynthesis?",
            "mode": "vector",
            "allow_web": False
        }
    )
    print_response("Vector DB Mode: What is photosynthesis?", response)
    return response.status_code == 200


def test_kg_mode():
    """Test Knowledge Graph mode"""
    response = requests.post(
        f"{BASE_URL}/rag",
        json={
            "query": "What is photosynthesis?",
            "mode": "kg"
        }
    )
    print_response("Knowledge Graph Mode: What is photosynthesis?", response)
    return response.status_code == 200


def test_hybrid_mode():
    """Test Hybrid mode"""
    response = requests.post(
        f"{BASE_URL}/rag",
        json={
            "query": "What is gravity?",
            "mode": "hybrid",
            "allow_web": False
        }
    )
    print_response("Hybrid Mode: What is gravity?", response)
    return response.status_code == 200


def test_direct_vector():
    """Test direct vector endpoint"""
    response = requests.post(
        f"{BASE_URL}/rag/vector",
        json={
            "query": "What is Ohm's law?",
            "allow_web": False
        }
    )
    print_response("Direct Vector: What is Ohm's law?", response)
    return response.status_code == 200


def test_direct_kg():
    """Test direct KG endpoint"""
    response = requests.post(
        f"{BASE_URL}/rag/kg",
        json={
            "query": "What are acids?"
        }
    )
    print_response("Direct KG: What are acids?", response)
    return response.status_code == 200


def test_kg_entities():
    """Test KG entities endpoint"""
    response = requests.get(f"{BASE_URL}/kg/entities")
    print_response("KG Entities", response)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n📊 Total Entities: {data.get('count', 0)}")
        if data.get('entities'):
            print("Top 10 entities:")
            for e in data['entities'][:10]:
                print(f"  - {e['name']} ({e['type']}) - {e['connections']} connections")
    
    return response.status_code == 200


def test_kg_relations():
    """Test KG relations endpoint"""
    response = requests.get(f"{BASE_URL}/kg/relations")
    print_response("KG Relations", response)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n📊 Total Relations: {data.get('count', 0)}")
        if data.get('relations'):
            print("Sample relations:")
            for r in data['relations'][:10]:
                print(f"  - {r['source']} --[{r['relation']}]--> {r['target']}")
    
    return response.status_code == 200


def test_kg_graph():
    """Test KG graph data endpoint"""
    response = requests.get(f"{BASE_URL}/kg/graph")
    print_response("KG Graph Data", response)
    
    if response.status_code == 200:
        data = response.json()
        stats = data.get('stats', {})
        print(f"\n📊 Graph Stats:")
        print(f"  - Total Nodes: {stats.get('total_nodes', 0)}")
        print(f"  - Total Edges: {stats.get('total_edges', 0)}")
    
    return response.status_code == 200


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 SCIENCE RAG API - DUAL MODE TEST CLIENT")
    print("="*60)
    
    tests = [
        ("Health Check", test_health),
        ("API Info", test_api_info),
        ("Vector DB Mode", test_vector_mode),
        ("Knowledge Graph Mode", test_kg_mode),
        ("Hybrid Mode", test_hybrid_mode),
        ("Direct Vector Endpoint", test_direct_vector),
        ("Direct KG Endpoint", test_direct_kg),
        ("KG Entities List", test_kg_entities),
        ("KG Relations List", test_kg_relations),
        ("KG Graph Data", test_kg_graph),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")


if __name__ == "__main__":
    main()
