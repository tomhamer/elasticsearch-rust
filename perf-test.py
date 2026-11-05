#!/usr/bin/env python3
"""
Performance comparison: Qdrant vs Our ES vs Official ES 8.19.22
Dataset: BEIR trec-covid (100K docs, 1024-dim vectors)
"""
import json
import time
import statistics
import requests
import concurrent.futures
from typing import List, Dict, Any

# Config
QDRANT_URL = "http://localhost:6333"
OUR_ES_URL = "http://localhost:9500"
OFFICIAL_ES_URL = "http://localhost:9501"
DATASET_PATH = "/home/ec2-user/perf-test/beir_trec_covid_100k.json"
INDEX_NAME = "beir_trec_covid"
NUM_QUERIES = 100  # Number of queries to run per test
BATCH_SIZE = 500  # Bulk index batch size

def load_dataset() -> List[Dict[str, Any]]:
    """Load the BEIR dataset."""
    print(f"Loading dataset from {DATASET_PATH}...")
    with open(DATASET_PATH) as f:
        docs = json.load(f)
    print(f"Loaded {len(docs)} docs")
    return docs

def setup_qdrant(docs: List[Dict[str, Any]]):
    """Create Qdrant collection and index docs."""
    print("\n=== Setting up Qdrant ===")

    # Delete existing collection
    try:
        requests.delete(f"{QDRANT_URL}/collections/{INDEX_NAME}", timeout=5)
        time.sleep(1)
    except:
        pass

    # Create collection
    print("Creating collection...")
    r = requests.put(f"{QDRANT_URL}/collections/{INDEX_NAME}", json={
        "vectors": {"size": 1024, "distance": "Cosine"}
    }, timeout=30)
    print(f"  Create: {r.status_code}")

    # Index docs in batches (use uint64 IDs for Qdrant)
    import random
    random.seed(42)
    categories = ["virology", "immunology", "epidemiology", "clinical", "public_health", "vaccines", "treatment", "diagnosis"]
    print(f"Indexing {len(docs)} docs...")
    start = time.time()
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i+BATCH_SIZE]
        points = [
            {
                "id": j + i,  # uint64, not string
                "vector": doc["emb"],
                "payload": {
                    "title": doc["title"],
                    "text": doc["text"],
                    "doc_id": doc["_id"],
                    "category": categories[(i + j) % len(categories)],
                    "year": 2020 + ((i + j) % 5)
                }
            }
            for j, doc in enumerate(batch)
        ]
        r = requests.put(f"{QDRANT_URL}/collections/{INDEX_NAME}/points", json={
            "points": points
        }, timeout=60)
        if r.status_code != 200:
            print(f"  Error at batch {i}: {r.status_code} {r.text[:200]}")
            break
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  {i + len(batch)}/{len(docs)} docs...")

    elapsed = time.time() - start
    print(f"  Indexed in {elapsed:.1f}s ({len(docs)/elapsed:.0f} docs/s)")

    # Wait for indexing to complete
    print("Waiting for indexing to complete...")
    while True:
        r = requests.get(f"{QDRANT_URL}/collections/{INDEX_NAME}", timeout=10)
        status = r.json()["result"]["status"]
        if status == "green":
            break
        time.sleep(1)
    print("  Collection ready")

def setup_es(es_url: str, docs: List[Dict[str, Any]], name: str):
    """Create ES index and bulk index docs."""
    print(f"\n=== Setting up {name} ({es_url}) ===")

    # Delete existing index
    try:
        requests.delete(f"{es_url}/{INDEX_NAME}", timeout=5)
        time.sleep(1)
    except:
        pass

    # Create index with mapping
    print("Creating index...")
    mapping = {
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "text": {"type": "text"},
                "emb": {
                    "type": "dense_vector",
                    "dims": 1024,
                    "index": True,
                    "similarity": "cosine"
                },
                # Add filterable fields for filtered KNN tests
                "category": {"type": "keyword"},
                "year": {"type": "integer"}
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }
    r = requests.put(f"{es_url}/{INDEX_NAME}", json=mapping, timeout=30)
    if r.status_code != 200:
        print(f"  Create failed: {r.status_code} {r.text[:200]}")
    else:
        print(f"  Create: {r.status_code}")

    # Bulk index docs (add category and year for filtered KNN tests)
    import random
    random.seed(42)
    categories = ["virology", "immunology", "epidemiology", "clinical", "public_health", "vaccines", "treatment", "diagnosis"]
    print(f"Indexing {len(docs)} docs...")
    start = time.time()
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i+BATCH_SIZE]
        lines = []
        for j, doc in enumerate(batch):
            lines.append(json.dumps({"index": {"_index": INDEX_NAME, "_id": doc["_id"]}}))
            # Add synthetic filterable fields
            category = categories[(i + j) % len(categories)]
            year = 2020 + ((i + j) % 5)  # 2020-2024
            lines.append(json.dumps({
                "title": doc["title"],
                "text": doc["text"],
                "emb": doc["emb"],
                "category": category,
                "year": year
            }))
        body = "\n".join(lines) + "\n"

        r = requests.post(f"{es_url}/_bulk", data=body, headers={"Content-Type": "application/x-ndjson"}, timeout=120)
        if r.status_code != 200:
            print(f"  Error at batch {i}: {r.status_code} {r.text[:200]}")
            break
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  {i + len(batch)}/{len(docs)} docs...")

    elapsed = time.time() - start
    print(f"  Indexed in {elapsed:.1f}s ({len(docs)/elapsed:.0f} docs/s)")

    # Refresh
    requests.post(f"{es_url}/{INDEX_NAME}/_refresh", timeout=30)
    print("  Refreshed")

def generate_queries(docs: List[Dict[str, Any]], n: int = NUM_QUERIES) -> List[Dict[str, Any]]:
    """Generate test queries from the dataset."""
    import random
    random.seed(42)
    sample = random.sample(docs, n)
    queries = []
    for doc in sample:
        queries.append({
            "vector": doc["emb"],
            "text": doc["title"][:50],  # Use title as text query
            "id": doc["_id"]
        })
    return queries

def bench_qdrant(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Benchmark Qdrant vector search."""
    print("\n=== Benchmarking Qdrant (vector search) ===")
    latencies = []

    for q in queries:
        start = time.time()
        r = requests.post(f"{QDRANT_URL}/collections/{INDEX_NAME}/points/search", json={
            "vector": q["vector"],
            "limit": 10,
            "with_payload": False
        }, timeout=30)
        elapsed = (time.time() - start) * 1000  # ms
        latencies.append(elapsed)

        if r.status_code != 200:
            print(f"  Error: {r.status_code} {r.text[:100]}")
            break

    return compute_stats(latencies, "Qdrant vector")

def bench_our_es_vector(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Benchmark our ES kNN search."""
    print("\n=== Benchmarking Our ES (kNN search) ===")
    latencies = []

    for q in queries:
        body = {
            "knn": {
                "field": "emb",
                "query_vector": q["vector"],
                "k": 10
            },
            "size": 10
        }
        start = time.time()
        r = requests.post(f"{OUR_ES_URL}/{INDEX_NAME}/_search", json=body, timeout=30)
        elapsed = (time.time() - start) * 1000
        latencies.append(elapsed)

        if r.status_code != 200:
            print(f"  Error: {r.status_code} {r.text[:100]}")
            break

    return compute_stats(latencies, "Our ES kNN")

def bench_official_es_vector(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Benchmark official ES kNN search."""
    print("\n=== Benchmarking Official ES (kNN search) ===")
    latencies = []

    for q in queries:
        body = {
            "knn": {
                "field": "emb",
                "query_vector": q["vector"],
                "k": 10
            },
            "size": 10
        }
        start = time.time()
        r = requests.post(f"{OFFICIAL_ES_URL}/{INDEX_NAME}/_search", json=body, timeout=30)
        elapsed = (time.time() - start) * 1000
        latencies.append(elapsed)

        if r.status_code != 200:
            print(f"  Error: {r.status_code} {r.text[:100]}")
            break

    return compute_stats(latencies, "Official ES kNN")

def bench_our_es_text(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Benchmark our ES text search (BM25)."""
    print("\n=== Benchmarking Our ES (text search) ===")
    latencies = []

    for q in queries:
        body = {
            "query": {
                "match": {
                    "title": q["text"]
                }
            },
            "size": 10
        }
        start = time.time()
        r = requests.post(f"{OUR_ES_URL}/{INDEX_NAME}/_search", json=body, timeout=30)
        elapsed = (time.time() - start) * 1000
        latencies.append(elapsed)

        if r.status_code != 200:
            print(f"  Error: {r.status_code} {r.text[:100]}")
            break

    return compute_stats(latencies, "Our ES text")

def bench_official_es_text(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Benchmark official ES text search (BM25)."""
    print("\n=== Benchmarking Official ES (text search) ===")
    latencies = []

    for q in queries:
        body = {
            "query": {
                "match": {
                    "title": q["text"]
                }
            },
            "size": 10
        }
        start = time.time()
        r = requests.post(f"{OFFICIAL_ES_URL}/{INDEX_NAME}/_search", json=body, timeout=30)
        elapsed = (time.time() - start) * 1000
        latencies.append(elapsed)

        if r.status_code != 200:
            print(f"  Error: {r.status_code} {r.text[:100]}")
            break

    return compute_stats(latencies, "Official ES text")

def bench_our_es_hybrid(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Benchmark our ES hybrid search (RRF)."""
    print("\n=== Benchmarking Our ES (hybrid RRF) ===")
    latencies = []

    for q in queries:
        body = {
            "retriever": {
                "rrf": {
                    "retrievers": [
                        {
                            "standard": {
                                "query": {
                                    "match": {
                                        "title": q["text"]
                                    }
                                }
                            }
                        },
                        {
                            "knn": {
                                "field": "emb",
                                "query_vector": q["vector"],
                                "k": 10
                            }
                        }
                    ]
                }
            },
            "size": 10
        }
        start = time.time()
        r = requests.post(f"{OUR_ES_URL}/{INDEX_NAME}/_search", json=body, timeout=30)
        elapsed = (time.time() - start) * 1000
        latencies.append(elapsed)

        if r.status_code != 200:
            print(f"  Error: {r.status_code} {r.text[:100]}")
            break

    return compute_stats(latencies, "Our ES hybrid")

def bench_official_es_hybrid(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Benchmark official ES hybrid search (RRF)."""
    print("\n=== Benchmarking Official ES (hybrid RRF) ===")
    latencies = []

    for q in queries:
        body = {
            "retriever": {
                "rrf": {
                    "retrievers": [
                        {
                            "standard": {
                                "query": {
                                    "match": {
                                        "title": q["text"]
                                    }
                                }
                            }
                        },
                        {
                            "knn": {
                                "field": "emb",
                                "query_vector": q["vector"],
                                "k": 10
                            }
                        }
                    ]
                }
            },
            "size": 10
        }
        start = time.time()
        r = requests.post(f"{OFFICIAL_ES_URL}/{INDEX_NAME}/_search", json=body, timeout=30)
        elapsed = (time.time() - start) * 1000
        latencies.append(elapsed)

        if r.status_code != 200:
            print(f"  Error: {r.status_code} {r.text[:100]}")
            break

    return compute_stats(latencies, "Official ES hybrid")

def bench_qdrant_filtered(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Benchmark Qdrant filtered vector search."""
    print("\n=== Benchmarking Qdrant (filtered vector search) ===")
    latencies = []

    for i, q in enumerate(queries):
        # Filter by category (synthetic field) - use index to alternate
        category = "virology" if i % 2 == 0 else "immunology"
        body = {
            "vector": q["vector"],
            "limit": 10,
            "filter": {
                "must": [
                    {
                        "key": "category",
                        "match": {"value": category}
                    }
                ]
            },
            "with_payload": False
        }
        start = time.time()
        r = requests.post(f"{QDRANT_URL}/collections/{INDEX_NAME}/points/search", json=body, timeout=30)
        elapsed = (time.time() - start) * 1000
        latencies.append(elapsed)

        if r.status_code != 200:
            print(f"  Error: {r.status_code} {r.text[:100]}")
            break

    return compute_stats(latencies, "Qdrant filtered vector")

def bench_our_es_filtered_knn(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Benchmark our ES filtered kNN search."""
    print("\n=== Benchmarking Our ES (filtered kNN) ===")
    latencies = []

    for i, q in enumerate(queries):
        # Filter by category (synthetic field) - use index to alternate
        category = "virology" if i % 2 == 0 else "immunology"
        body = {
            "knn": {
                "field": "emb",
                "query_vector": q["vector"],
                "k": 10,
                "filter": {
                    "term": {
                        "category": category
                    }
                }
            },
            "size": 10
        }
        start = time.time()
        r = requests.post(f"{OUR_ES_URL}/{INDEX_NAME}/_search", json=body, timeout=30)
        elapsed = (time.time() - start) * 1000
        latencies.append(elapsed)

        if r.status_code != 200:
            print(f"  Error: {r.status_code} {r.text[:100]}")
            break

    return compute_stats(latencies, "Our ES filtered kNN")

def bench_official_es_filtered_knn(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Benchmark official ES filtered kNN search."""
    print("\n=== Benchmarking Official ES (filtered kNN) ===")
    latencies = []

    for i, q in enumerate(queries):
        # Filter by category (synthetic field) - use index to alternate
        category = "virology" if i % 2 == 0 else "immunology"
        body = {
            "knn": {
                "field": "emb",
                "query_vector": q["vector"],
                "k": 10,
                "filter": {
                    "term": {
                        "category": category
                    }
                }
            },
            "size": 10
        }
        start = time.time()
        r = requests.post(f"{OFFICIAL_ES_URL}/{INDEX_NAME}/_search", json=body, timeout=30)
        elapsed = (time.time() - start) * 1000
        latencies.append(elapsed)

        if r.status_code != 200:
            print(f"  Error: {r.status_code} {r.text[:100]}")
            break

    return compute_stats(latencies, "Official ES filtered kNN")

def compute_recall(queries: List[Dict[str, Any]], k: int = 10) -> Dict[str, Any]:
    """
    Compute recall@k for vector search.
    Ground truth: the k nearest neighbors by cosine similarity (brute force).
    """
    print(f"\n=== Computing Recall@{k} (brute force ground truth) ===")

    # Load all vectors
    with open(DATASET_PATH) as f:
        all_docs = json.load(f)

    all_vectors = [doc["emb"] for doc in all_docs]
    all_ids = [doc["_id"] for doc in all_docs]

    def cosine_similarity(v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = sum(a * a for a in v1) ** 0.5
        mag2 = sum(b * b for b in v2) ** 0.5
        return dot / (mag1 * mag2) if mag1 > 0 and mag2 > 0 else 0

    recalls = {"qdrant": [], "our_es": [], "official_es": []}

    for i, q in enumerate(queries):
        if i % 20 == 0:
            print(f"  Query {i + 1}/{len(queries)}...")

        # Compute ground truth (top-k by cosine similarity)
        scores = []
        for j, vec in enumerate(all_vectors):
            sim = cosine_similarity(q["vector"], vec)
            scores.append((sim, all_ids[j]))
        scores.sort(reverse=True)
        ground_truth = set(doc_id for _, doc_id in scores[:k])

        # Qdrant
        try:
            r = requests.post(f"{QDRANT_URL}/collections/{INDEX_NAME}/points/search", json={
                "vector": q["vector"],
                "limit": k,
                "with_payload": True
            }, timeout=30)
            if r.status_code == 200:
                results = r.json()["result"]
                result_ids = set(str(p["payload"]["doc_id"]) for p in results)
                recalls["qdrant"].append(len(ground_truth & result_ids) / k)
        except:
            pass

        # Our ES
        try:
            r = requests.post(f"{OUR_ES_URL}/{INDEX_NAME}/_search", json={
                "knn": {"field": "emb", "query_vector": q["vector"], "k": k},
                "size": k
            }, timeout=30)
            if r.status_code == 200:
                hits = r.json()["hits"]["hits"]
                result_ids = set(h["_id"] for h in hits)
                recalls["our_es"].append(len(ground_truth & result_ids) / k)
        except:
            pass

        # Official ES
        try:
            r = requests.post(f"{OFFICIAL_ES_URL}/{INDEX_NAME}/_search", json={
                "knn": {"field": "emb", "query_vector": q["vector"], "k": k},
                "size": k
            }, timeout=30)
            if r.status_code == 200:
                hits = r.json()["hits"]["hits"]
                result_ids = set(h["_id"] for h in hits)
                recalls["official_es"].append(len(ground_truth & result_ids) / k)
        except:
            pass

    return {
        "recall@10": {
            "qdrant": statistics.mean(recalls["qdrant"]) if recalls["qdrant"] else 0,
            "our_es": statistics.mean(recalls["our_es"]) if recalls["our_es"] else 0,
            "official_es": statistics.mean(recalls["official_es"]) if recalls["official_es"] else 0
        }
    }

def compute_stats(latencies: List[float], name: str) -> Dict[str, Any]:
    """Compute latency statistics."""
    if not latencies:
        return {"name": name, "error": "no data"}

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    return {
        "name": name,
        "count": n,
        "mean_ms": statistics.mean(latencies),
        "p50_ms": latencies_sorted[n // 2],
        "p95_ms": latencies_sorted[int(n * 0.95)],
        "p99_ms": latencies_sorted[int(n * 0.99)],
        "min_ms": latencies_sorted[0],
        "max_ms": latencies_sorted[-1],
        "qps": 1000 / statistics.mean(latencies)
    }

def print_results(results: List[Dict[str, Any]]):
    """Print results table."""
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"{'Test':<25} {'Count':>6} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'QPS':>8}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<25} {'ERROR':>6}")
        else:
            print(f"{r['name']:<25} {r['count']:>6} {r['mean_ms']:>7.1f} {r['p50_ms']:>7.1f} {r['p95_ms']:>7.1f} {r['p99_ms']:>7.1f} {r['qps']:>7.0f}")
    print("=" * 80)

def main():
    # Load dataset
    docs = load_dataset()

    # Setup all engines
    setup_qdrant(docs)
    setup_es(OUR_ES_URL, docs, "Our ES")
    setup_es(OFFICIAL_ES_URL, docs, "Official ES")

    # Generate queries
    print(f"\nGenerating {NUM_QUERIES} test queries...")
    queries = generate_queries(docs)

    # Run benchmarks
    results = []
    results.append(bench_qdrant(queries))
    results.append(bench_our_es_vector(queries))
    results.append(bench_official_es_vector(queries))
    results.append(bench_our_es_text(queries))
    results.append(bench_official_es_text(queries))
    results.append(bench_our_es_hybrid(queries))
    results.append(bench_official_es_hybrid(queries))
    results.append(bench_qdrant_filtered(queries))
    results.append(bench_our_es_filtered_knn(queries))
    results.append(bench_official_es_filtered_knn(queries))

    # Print results
    print_results(results)

    # Compute recall
    recall = compute_recall(queries)
    print(f"\nRecall@10: Qdrant={recall['recall@10']['qdrant']:.3f}, Our ES={recall['recall@10']['our_es']:.3f}, Official ES={recall['recall@10']['official_es']:.3f}")

    # Save results
    with open("/home/ec2-user/perf-test/results.json", "w") as f:
        json.dump({"latency": results, "recall": recall}, f, indent=2)
    print("\nResults saved to /home/ec2-user/perf-test/results.json")

if __name__ == "__main__":
    main()
