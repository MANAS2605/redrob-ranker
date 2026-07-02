#!/usr/bin/env python3
"""
parse_jd.py — Job Description Understanding Module

Reads a free-text job description and extracts structured requirements
using rule-based NLP (no LLM, no network, CPU-only).

Outputs a JSON config file with:
  - must_have_skills: list of skill clusters detected as required
  - nice_to_have_skills: list of skill clusters detected as preferred
  - experience_range: (min, max) years
  - ideal_locations: list of preferred cities/regions
  - title_keywords: role-relevant title keywords
  - domain_keywords: domain-specific terms found in the JD
  - seniority: detected seniority level
  - raw_skills_found: all individual skills detected
  - jd_summary: brief structured summary of what was understood

Usage:
  python precompute/parse_jd.py --jd-file custom_jd.txt --output jd_config.json
"""

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Comprehensive skill dictionary (650+ tech skills, grouped by domain)
# ---------------------------------------------------------------------------

SKILL_DICTIONARY = {
    # AI / ML Core
    "machine_learning": [
        "machine learning", "deep learning", "neural network", "neural networks",
        "supervised learning", "unsupervised learning", "reinforcement learning",
        "transfer learning", "online learning", "federated learning",
        "gradient descent", "backpropagation", "feature engineering",
        "model training", "model evaluation", "hyperparameter tuning",
        "cross-validation", "ensemble methods", "bagging", "boosting",
        "random forest", "decision tree", "svm", "support vector",
        "logistic regression", "linear regression", "bayesian",
        "xgboost", "lightgbm", "catboost", "gradient boosting",
    ],
    "deep_learning_frameworks": [
        "pytorch", "tensorflow", "keras", "jax", "mxnet", "caffe",
        "onnx", "tensorrt", "triton", "openvino", "tflite",
    ],
    "nlp": [
        "nlp", "natural language processing", "text classification",
        "named entity recognition", "ner", "sentiment analysis",
        "text mining", "tokenization", "word embeddings", "text generation",
        "language model", "language models", "question answering",
        "text summarization", "machine translation", "spacy", "nltk",
        "hugging face", "huggingface", "transformers",
    ],
    "embeddings_retrieval": [
        "embeddings", "embedding", "sentence-transformers", "sentence transformers",
        "sbert", "bge", "e5", "openai embeddings", "text embeddings",
        "word2vec", "doc2vec", "glove", "fasttext",
        "semantic search", "semantic similarity", "vector search",
        "dense retrieval", "sparse retrieval", "hybrid search",
        "approximate nearest neighbor", "ann", "hnsw",
    ],
    "vector_databases": [
        "pinecone", "weaviate", "qdrant", "milvus", "chromadb", "chroma",
        "faiss", "annoy", "scann", "vector database", "vector databases",
        "vector db", "pgvector",
    ],
    "search_ranking": [
        "elasticsearch", "elastic search", "opensearch", "solr",
        "lucene", "information retrieval", "search engine",
        "ranking", "learning to rank", "lambdamart",
        "ndcg", "mrr", "map", "recall", "precision",
        "search relevance", "query understanding", "click model",
        "bm25", "tf-idf", "tfidf",
    ],
    "recommendation_systems": [
        "recommendation system", "recommendation engine", "recommender",
        "collaborative filtering", "content-based filtering",
        "matrix factorization", "implicit feedback",
        "cold start", "item2vec", "user2vec",
    ],
    "llm_genai": [
        "llm", "large language model", "gpt", "chatgpt", "openai",
        "claude", "gemini", "llama", "mistral", "anthropic",
        "prompt engineering", "chain of thought", "few-shot",
        "fine-tuning", "finetuning", "lora", "qlora", "peft",
        "rlhf", "instruction tuning", "adapter",
        "rag", "retrieval augmented generation",
        "langchain", "llamaindex", "llama index",
        "agent", "ai agent", "tool use",
    ],
    "computer_vision": [
        "computer vision", "image classification", "object detection",
        "image segmentation", "yolo", "resnet", "efficientnet",
        "opencv", "image processing", "video analysis",
        "ocr", "optical character recognition", "gan",
        "generative adversarial", "stable diffusion", "dall-e",
        "convolutional neural network", "cnn",
    ],
    "data_science": [
        "data science", "data analysis", "statistical analysis",
        "a/b testing", "ab testing", "experimentation",
        "hypothesis testing", "statistics", "probability",
        "time series", "forecasting", "anomaly detection",
        "clustering", "dimensionality reduction", "pca",
        "pandas", "numpy", "scipy", "matplotlib", "seaborn",
        "jupyter", "notebook", "data visualization",
    ],
    "mlops": [
        "mlops", "ml pipeline", "model serving", "model deployment",
        "model monitoring", "feature store", "experiment tracking",
        "mlflow", "kubeflow", "airflow", "dagster", "prefect",
        "wandb", "weights and biases", "neptune",
        "distributed training", "distributed inference",
        "model registry", "ci/cd for ml",
    ],

    # Programming Languages
    "python": ["python"],
    "java": ["java", "spring boot", "spring framework", "maven", "gradle"],
    "javascript_typescript": [
        "javascript", "typescript", "node.js", "nodejs",
        "react", "angular", "vue", "next.js", "nextjs",
        "express", "fastify", "deno", "bun",
    ],
    "go": ["golang", "go language", "go programming"],
    "rust": ["rust", "cargo"],
    "cpp": ["c++", "cpp", "c programming"],
    "scala": ["scala", "spark scala"],
    "r_language": ["r programming", "r language", "rstudio", "tidyverse"],

    # Data Engineering
    "data_engineering": [
        "data engineering", "data pipeline", "etl", "elt",
        "data warehouse", "data lake", "data lakehouse",
        "data modeling", "data governance", "data quality",
        "batch processing", "stream processing",
    ],
    "big_data": [
        "spark", "apache spark", "pyspark", "hadoop",
        "hive", "presto", "trino", "flink", "kafka",
        "apache kafka", "streaming", "real-time processing",
    ],
    "databases": [
        "sql", "postgresql", "postgres", "mysql", "mongodb",
        "redis", "cassandra", "dynamodb", "neo4j", "graph database",
        "oracle", "sql server", "sqlite", "cockroachdb",
        "timescaledb", "influxdb", "clickhouse",
    ],

    # Cloud & Infrastructure
    "aws": [
        "aws", "amazon web services", "s3", "ec2", "lambda",
        "sagemaker", "bedrock", "emr", "glue", "redshift",
        "kinesis", "cloudformation", "ecs", "eks",
    ],
    "gcp": [
        "gcp", "google cloud", "bigquery", "vertex ai",
        "cloud run", "gke", "dataflow", "pub/sub",
    ],
    "azure": [
        "azure", "microsoft azure", "azure ml", "cosmos db",
        "azure devops", "azure functions",
    ],
    "devops_infra": [
        "docker", "kubernetes", "k8s", "terraform",
        "helm", "ansible", "ci/cd", "github actions",
        "jenkins", "gitlab ci", "circleci",
        "monitoring", "prometheus", "grafana", "datadog",
        "logging", "elk stack",
    ],

    # Backend & Systems
    "backend": [
        "rest api", "restful", "graphql", "grpc",
        "microservices", "api design", "api gateway",
        "fastapi", "flask", "django",
        "gin", "fiber", "actix",
    ],
    "systems": [
        "distributed systems", "system design", "scalability",
        "high availability", "fault tolerance", "load balancing",
        "caching", "message queue", "event-driven",
        "concurrency", "multithreading", "async",
    ],

    # Frontend & Mobile
    "frontend": [
        "html", "css", "responsive design", "ui/ux",
        "react", "angular", "vue.js", "svelte",
        "tailwind", "bootstrap", "material ui",
        "webpack", "vite", "figma",
    ],
    "mobile": [
        "ios", "android", "react native", "flutter",
        "swift", "kotlin", "mobile development",
    ],

    # Security
    "security": [
        "cybersecurity", "penetration testing", "security audit",
        "encryption", "authentication", "authorization",
        "oauth", "jwt", "sso", "zero trust",
        "vulnerability", "soc", "siem",
    ],

    # Product & Management
    "product_management": [
        "product management", "product strategy", "roadmap",
        "agile", "scrum", "kanban", "jira",
        "stakeholder management", "user research",
    ],
}

# Flatten for quick lookup
ALL_SKILLS_FLAT = {}
for domain, skills in SKILL_DICTIONARY.items():
    for skill in skills:
        ALL_SKILLS_FLAT[skill.lower()] = domain


# ---------------------------------------------------------------------------
# Experience extraction patterns
# ---------------------------------------------------------------------------

EXPERIENCE_PATTERNS = [
    # "5+ years", "5+ yrs"
    r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)",
    # "5-8 years experience"
    r"(\d+)\s*[-–to]+\s*(\d+)\s*(?:years?|yrs?)",
    # "at least 5 years"
    r"(?:at\s+least|minimum|min)\s*(\d+)\s*(?:years?|yrs?)",
    # "experience of 5 years"
    r"experience\s+(?:of\s+)?(\d+)\s*(?:years?|yrs?)",
    # "5 years of experience"
    r"(\d+)\s*(?:years?|yrs?)\s+(?:of\s+)?(?:relevant\s+)?(?:experience|exp)",
]

# Seniority patterns
SENIORITY_MAP = {
    "intern": "intern",
    "junior": "junior",
    "entry level": "junior",
    "entry-level": "junior",
    "mid level": "mid",
    "mid-level": "mid",
    "senior": "senior",
    "sr.": "senior",
    "lead": "lead",
    "staff": "staff",
    "principal": "principal",
    "architect": "senior",
    "director": "director",
    "head of": "director",
    "vp": "executive",
    "vice president": "executive",
    "manager": "manager",
}

# Location patterns (global cities)
LOCATION_DICTIONARY = {
    # India
    "noida": "India", "pune": "India", "bangalore": "India",
    "bengaluru": "India", "hyderabad": "India", "mumbai": "India",
    "delhi": "India", "new delhi": "India", "gurgaon": "India",
    "gurugram": "India", "chennai": "India", "kolkata": "India",
    "ahmedabad": "India", "jaipur": "India", "kochi": "India",
    "thiruvananthapuram": "India", "indore": "India", "lucknow": "India",
    # US
    "san francisco": "US", "new york": "US", "seattle": "US",
    "austin": "US", "boston": "US", "chicago": "US",
    "los angeles": "US", "denver": "US", "atlanta": "US",
    "silicon valley": "US", "bay area": "US",
    # Europe
    "london": "UK", "berlin": "Germany", "amsterdam": "Netherlands",
    "paris": "France", "dublin": "Ireland", "zurich": "Switzerland",
    "barcelona": "Spain", "stockholm": "Sweden",
    # Others
    "singapore": "Singapore", "dubai": "UAE", "tokyo": "Japan",
    "toronto": "Canada", "vancouver": "Canada", "sydney": "Australia",
    "tel aviv": "Israel",
    # Special
    "remote": "Remote", "hybrid": "Hybrid", "on-site": "On-site",
    "onsite": "On-site", "work from home": "Remote", "wfh": "Remote",
}


# ---------------------------------------------------------------------------
# JD Parsing functions
# ---------------------------------------------------------------------------

def extract_skills_from_text(text: str) -> dict:
    """
    Extract skills from JD text using dictionary matching.
    Returns {domain: [matched_terms]} mapping.
    """
    text_lower = text.lower()
    found = {}

    for domain, terms in SKILL_DICTIONARY.items():
        matched = []
        for term in terms:
            # Use word boundary matching where possible
            pattern = r'(?:^|[\s,;./()\-])' + re.escape(term) + r'(?:[\s,;./()\-]|$)'
            if re.search(pattern, text_lower):
                matched.append(term)
        if matched:
            found[domain] = matched

    return found


def classify_skill_importance(text: str, found_skills: dict) -> tuple:
    """
    Classify found skills as must-have or nice-to-have based on context.
    
    Heuristic: Skills near "required", "must have", "essential" → must-have
               Skills near "preferred", "nice to have", "bonus" → nice-to-have
               Default: if in first half of JD → must-have, second half → nice-to-have
    """
    text_lower = text.lower()

    # Find "required" and "preferred" sections
    required_markers = [
        "required", "must have", "must-have", "essential",
        "mandatory", "requirements", "qualifications",
        "what you need", "what we need", "you should have",
        "key skills", "core skills", "necessary",
    ]
    preferred_markers = [
        "preferred", "nice to have", "nice-to-have", "bonus",
        "good to have", "good-to-have", "plus point",
        "advantageous", "desirable", "ideally", "would be great",
        "nice to have:", "good to have:",
    ]

    # Find positions of markers (word-boundary matching)
    required_positions = []
    for marker in required_markers:
        pattern = r'(?:^|[\s,;.:\-])' + re.escape(marker) + r'(?:[\s,;.:\-]|$)'
        for m in re.finditer(pattern, text_lower):
            required_positions.append(m.start())

    preferred_positions = []
    for marker in preferred_markers:
        pattern = r'(?:^|[\s,;.:\-])' + re.escape(marker) + r'(?:[\s,;.:\-]|$)'
        for m in re.finditer(pattern, text_lower):
            preferred_positions.append(m.start())

    must_have = {}
    nice_to_have = {}

    for domain, terms in found_skills.items():
        # Find where the first mention of this domain's terms appears
        first_pos = len(text_lower)
        for term in terms:
            pos = text_lower.find(term)
            if pos >= 0:
                first_pos = min(first_pos, pos)

        # Check if the term is closer to a required or preferred marker
        min_req_dist = min((abs(first_pos - rp) for rp in required_positions), default=float("inf"))
        min_pref_dist = min((abs(first_pos - pp) for pp in preferred_positions), default=float("inf"))

        # Check if term appears after a preferred marker (section-based)
        in_preferred_section = any(
            first_pos > pp and (first_pos - pp) < 300
            for pp in preferred_positions
        )

        # Also check if a preferred marker appears on the SAME LINE as the skill
        text_lines = text_lower.split('\n')
        skill_on_preferred_line = False
        for line in text_lines:
            has_skill = any(t in line for t in terms)
            has_pref = any(pm in line for pm in preferred_markers)
            if has_skill and has_pref:
                skill_on_preferred_line = True
                break

        if in_preferred_section or skill_on_preferred_line:
            nice_to_have[domain] = terms
        elif min_pref_dist < min_req_dist and min_pref_dist < 200:
            nice_to_have[domain] = terms
        elif min_req_dist < 600 or first_pos < len(text_lower) * 0.6:
            must_have[domain] = terms
        else:
            nice_to_have[domain] = terms

    return must_have, nice_to_have


def extract_experience_range(text: str) -> tuple:
    """Extract (min_years, max_years) from JD text."""
    text_lower = text.lower()

    for pattern in EXPERIENCE_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            groups = match.groups()
            if len(groups) == 2 and groups[1]:
                return (int(groups[0]), int(groups[1]))
            elif len(groups) >= 1:
                years = int(groups[0])
                # "5+ years" → (5, years+3)
                if "+" in text_lower[max(0, match.start()-2):match.end()+2]:
                    return (years, years + 3)
                # "at least N" → (N, N+4)
                if any(m in text_lower[max(0, match.start()-15):match.start()] for m in ["at least", "minimum", "min"]):
                    return (years, years + 4)
                return (max(0, years - 1), years + 2)

    return (0, 99)  # No experience mentioned


def extract_locations(text: str) -> list:
    """Extract mentioned locations from JD text."""
    text_lower = text.lower()
    found = []

    for city, country in LOCATION_DICTIONARY.items():
        if city in text_lower:
            found.append({"city": city.title(), "country": country})

    return found


def detect_seniority(text: str) -> str:
    """Detect seniority level from JD text."""
    text_lower = text.lower()

    # Check title-like patterns first
    for keyword, level in sorted(SENIORITY_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if keyword in text_lower:
            return level

    return "mid"  # Default


def extract_title_keywords(text: str) -> list:
    """Extract role-relevant title keywords from JD."""
    text_lower = text.lower()

    title_patterns = [
        r"(?:looking for|hiring|seeking|need)\s+(?:a\s+)?(.+?)(?:\.|,|\n|with|who)",
        r"(?:role|position|job)\s*(?:title)?[:\s]+(.+?)(?:\.|,|\n)",
        r"^(.+?engineer.+?)(?:\.|,|\n)",
        r"^(.+?developer.+?)(?:\.|,|\n)",
    ]

    keywords = set()
    for pattern in title_patterns:
        match = re.search(pattern, text_lower)
        if match:
            phrase = match.group(1).strip()
            # Extract meaningful words
            for word in phrase.split():
                if word not in {"a", "an", "the", "for", "and", "or", "in", "at", "with", "to", "of"}:
                    keywords.add(word)

    # Also extract from common role titles found in text
    role_keywords = [
        "engineer", "developer", "scientist", "analyst", "architect",
        "manager", "lead", "specialist", "consultant", "designer",
        "devops", "sre", "backend", "frontend", "fullstack", "full-stack",
        "data", "ml", "ai", "cloud", "security", "platform",
        "mobile", "ios", "android", "web", "software",
    ]

    for kw in role_keywords:
        if kw in text_lower:
            keywords.add(kw)

    return sorted(keywords)


def extract_domain_keywords(text: str) -> list:
    """Extract domain-specific terms that indicate the industry/focus area."""
    text_lower = text.lower()

    domain_terms = {
        "fintech": ["fintech", "banking", "payments", "financial"],
        "healthtech": ["healthtech", "healthcare", "medical", "clinical"],
        "edtech": ["edtech", "education", "learning platform"],
        "ecommerce": ["ecommerce", "e-commerce", "retail", "marketplace"],
        "adtech": ["adtech", "advertising", "ad platform"],
        "hrtech": ["hrtech", "hr tech", "recruiting", "talent", "hiring", "candidate"],
        "saas": ["saas", "b2b", "enterprise"],
        "gaming": ["gaming", "game development"],
        "social": ["social media", "social network", "content platform"],
        "autonomous": ["autonomous", "self-driving", "robotics"],
        "iot": ["iot", "internet of things", "embedded"],
        "blockchain": ["blockchain", "web3", "crypto", "smart contract"],
    }

    found = []
    for domain, terms in domain_terms.items():
        for term in terms:
            if term in text_lower:
                found.append(domain)
                break

    return found


def build_jd_config(text: str) -> dict:
    """
    Master function: parse a JD and produce a structured config.
    """
    # 1. Extract skills
    found_skills = extract_skills_from_text(text)
    must_have, nice_to_have = classify_skill_importance(text, found_skills)

    # 2. Experience
    exp_min, exp_max = extract_experience_range(text)

    # 3. Locations
    locations = extract_locations(text)

    # 4. Seniority
    seniority = detect_seniority(text)

    # 5. Title keywords
    title_keywords = extract_title_keywords(text)

    # 6. Domain
    domain_keywords = extract_domain_keywords(text)

    # 7. Build flat skill lists for matching
    all_must_have_terms = []
    for domain, terms in must_have.items():
        all_must_have_terms.extend(terms)

    all_nice_to_have_terms = []
    for domain, terms in nice_to_have.items():
        all_nice_to_have_terms.extend(terms)

    # 8. Build readable summary
    summary_parts = []
    summary_parts.append(f"Seniority: {seniority}")
    if exp_min > 0 or exp_max < 99:
        summary_parts.append(f"Experience: {exp_min}-{exp_max} years")
    if locations:
        loc_str = ", ".join(f"{l['city']}" for l in locations[:5])
        summary_parts.append(f"Location: {loc_str}")
    if must_have:
        domains = ", ".join(must_have.keys())
        summary_parts.append(f"Must-have domains: {domains}")
    if nice_to_have:
        domains = ", ".join(nice_to_have.keys())
        summary_parts.append(f"Nice-to-have domains: {domains}")
    if domain_keywords:
        summary_parts.append(f"Industry focus: {', '.join(domain_keywords)}")

    config = {
        "must_have_skills": must_have,
        "must_have_terms": list(set(all_must_have_terms)),
        "nice_to_have_skills": nice_to_have,
        "nice_to_have_terms": list(set(all_nice_to_have_terms)),
        "experience_range": {"min": exp_min, "max": exp_max},
        "locations": locations,
        "ideal_locations": [l["city"].lower() for l in locations],
        "seniority": seniority,
        "title_keywords": title_keywords,
        "domain_keywords": domain_keywords,
        "all_skills_found": {d: t for d, t in found_skills.items()},
        "n_must_have_domains": len(must_have),
        "n_nice_to_have_domains": len(nice_to_have),
        "n_total_skills_found": sum(len(t) for t in found_skills.values()),
        "jd_summary": " | ".join(summary_parts),
        "jd_length_chars": len(text),
    }

    return config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parse a job description into structured requirements")
    parser.add_argument("--jd-file", required=True, help="Path to JD text file")
    parser.add_argument("--output", default="jd_config.json", help="Output JSON config path")
    args = parser.parse_args()

    print(f"Reading JD from {args.jd_file}...")
    with open(args.jd_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    print(f"JD length: {len(text)} characters")

    config = build_jd_config(text)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print("JD UNDERSTANDING SUMMARY")
    print(f"{'='*60}")
    print(f"  Seniority: {config['seniority']}")
    print(f"  Experience: {config['experience_range']['min']}-{config['experience_range']['max']} years")
    print(f"  Locations: {config['locations']}")
    print(f"  Must-have skill domains ({config['n_must_have_domains']}):")
    for domain, terms in config["must_have_skills"].items():
        print(f"    {domain}: {', '.join(terms[:5])}")
    print(f"  Nice-to-have skill domains ({config['n_nice_to_have_domains']}):")
    for domain, terms in config["nice_to_have_skills"].items():
        print(f"    {domain}: {', '.join(terms[:5])}")
    print(f"  Domain keywords: {config['domain_keywords']}")
    print(f"  Title keywords: {config['title_keywords']}")
    print(f"\nSaved config to {args.output}")


if __name__ == "__main__":
    main()
