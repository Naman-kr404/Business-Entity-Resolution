# Business Entity Resolution

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![LLM](https://img.shields.io/badge/LLM-Qwen_2.5_7B-orange.svg)](https://ollama.ai/)
[![Matching Engine](https://img.shields.io/badge/Engine-RapidFuzz%20%2B%20Ollama-green.svg)](https://github.com/rapidfuzz/RapidFuzz)

An end-to-end, high-performance **Business Entity Resolution (Record Linkage)** pipeline designed to match and consolidate heterogeneous business records across multiple disparate data sources.

The system combines rule-based preprocessing, prefix blocking, fuzzy string matching (RapidFuzz), and local Large Language Model (Qwen 2.5 7B via Ollama) reasoning for semantic address disambiguation with persistent caching.

---

## 📌 Problem Statement

In enterprise data integration, business records from multiple sources often suffer from:
- **Naming variations & legal suffix noise**: e.g., *"Acme Corp Ltd."* vs. *"Acme Corporation"*.
- **Word permutations**: e.g., *"Alpha Logistics Solutions"* vs. *"Logistics Solutions Alpha"*.
- **Address format discrepancies**: Abbreviated street names, varying granularities (one source contains floor/suite while another only lists city/state), and typographical errors.
- **Scalability bottlenecks**: Comparing every record against every other record ($O(N \times M)$) is computationally prohibitive on large datasets.

This pipeline solves these challenges by combining fast algorithmic filtering with deep semantic verification.

---

## 🏗️ Architecture & Pipeline

```text
  ┌─────────────────────────────────────────────────────────┐
  │                 Input Datasets (TSV)                    │
  │     Source 1 (Canonical/Target) | Source 2 | Source 3   │
  └────────────────────────────┬────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────┐
  │              1. Normalization & Preprocessing           │
  │  • Lowercase, remove special characters & extra spaces  │
  │  • Strip legal suffixes (ltd, pvt, corp, llc, inc, etc) │
  │  • Alphabetically sort words for permutation invariance │
  └────────────────────────────┬────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────┐
  │            2. Prefix Index Blocking (Source 1)          │
  │  • Buckets canonical records by 3-character name prefix │
  │  • Reduces candidate search space from O(N) to O(K)     │
  └────────────────────────────┬────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────┐
  │            3. Fuzzy Candidate Matching (RapidFuzz)      │
  │  • Evaluates string similarity (fuzz.ratio >= 72)       │
  │  • Retrieves Top-K (K=3) potential entity candidates    │
  └────────────────────────────┬────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────┐
  │          4. Semantic Address Disambiguation             │
  │  • Exact match bypass: instant zero-cost resolution     │
  │  • Persistent 2-way disk cache lookup                   │
  │  • LLM reasoning: Ollama (Qwen 2.5:7b) with structured  │
  │    JSON output & confidence threshold (>= 80%)          │
  └────────────────────────────┬────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────┐
  │              5. Aggregation & Output Generation         │
  │  • Groups Source 2 & 3 entity IDs under Source 1 ID     │
  │  • Exports canonical TSV mapping with deduplication     │
  └─────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

- **Robust Name Normalization**: Automatically strips standardized company suffixes (`limited`, `ltd`, `pvt`, `inc`, `corp`, `llc`, etc.) and sorts words alphabetically to resolve order differences.
- **Scalable Prefix Blocking**: Avoids expensive Cartesian product comparisons by grouping candidates by n-gram name prefixes.
- **Fuzzy String Matching**: Uses [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) for blazing-fast C++ backed Levenshtein distance calculations.
- **LLM Semantic Address Verification**: Employs **Qwen 2.5 7B** via **Ollama** to evaluate complex real-world address relationships (e.g. recognizing that missing landmarks or minor spelling quirks do not indicate distinct locations, while detecting genuine city/state conflicts).
- **Multi-Tier Caching System**:
  - **Exact Match Fast-Path**: Identical normalized addresses resolve without LLM invocations.
  - **In-Memory Query Cache**: Prevents duplicate LLM calls within repeated source entries.
  - **Persistent Bidirectional Disk Cache**: Periodically saves address comparison results to JSON (`address_match_cache.json`) to persist across interrupted runs.
- **Automatic Daemon Lifecycle**: Checks if the Ollama server is running on `http://localhost:11434` and starts `ollama serve` in the background if necessary.

---

## 📂 Repository Structure

```text
├── a5.py                        # Main entity resolution pipeline
├── README.md                    # Project documentation
└── .gitignore                   # Ignores large datasets (*.tsv), caches, and temp files
```

---

## 🚀 Getting Started

### 1. Prerequisites

- **Python**: 3.8 or later
- **Ollama**: Installed locally ([Download Ollama](https://ollama.ai/))

Pull the model used for semantic address matching:
```bash
ollama pull qwen2.5:7b
```

### 2. Install Python Dependencies

```bash
pip install pandas rapidfuzz ollama
```

### 3. Configure File Paths

In [`a5.py`](a5.py), update the dataset and output paths according to your environment:

```python
BASE = "/path/to/your/dataset_directory"

TRAIN_DIR = f"{BASE}/dataset/train"
S1_FILE = f"{TRAIN_DIR}/train_source1.tsv"
S2_FILE = f"{TRAIN_DIR}/train_source2.tsv"
S3_FILE = f"{TRAIN_DIR}/train_source3.tsv"

OUTPUT_FILE = f"{BASE}/matching_results_train.tsv"
ADDRESS_CACHE_FILE = f"{BASE}/address_match_cache.json"
```

### 4. Run the Pipeline

Execute the script:
```bash
python3 a5.py
```

The script will:
1. Verify / start the local Ollama daemon.
2. Read and normalize Source 1, Source 2, and Source 3.
3. Build the prefix index for Source 1.
4. Match records from Source 2 and Source 3 against Source 1 with real-time progress logging.
5. Save the consolidated results to `OUTPUT_FILE`.

---

## ⚙️ Hyperparameters & Configuration

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `MODEL_NAME` | `"qwen2.5:7b"` | Ollama model utilized for address evaluation |
| `NAME_THRESHOLD` | `72` | Minimum RapidFuzz similarity ratio (`0-100`) to consider a candidate |
| `PREFIX_LENGTH` | `3` | Number of characters used for prefix blocking |
| `TOP_K` | `3` | Maximum number of candidate names evaluated per query |
| `CONFIDENCE_THRESHOLD` | `80` | Minimum LLM confidence score to accept an address match |

---

## 📊 Data Schema

### Input Data (`.tsv`)
Each input source file contains tab-separated records with the following headers:
- `entity_id`: Unique identifier for the record
- `business_name`: Registered or operational business name
- `business_address`: Full or partial address string
- `country`: Country identifier

### Output Data (`matching_results_train.tsv`)
The pipeline generates a mapped TSV with:
- `source1_entity_id`: The canonical Source 1 entity ID.
- `matched_entity_ids`: Comma-separated list of matched entity IDs from Source 2 and Source 3 (e.g., `s2_1042,s3_891`).

---

## 📝 License

This project is licensed under the MIT License - see the repository details for more information.