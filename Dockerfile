FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt streamlit

# Pre-download the sentence-transformers model so it's baked into the image
# (no internet needed at runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code
COPY app.py .
COPY rank.py .
COPY validate_submission.py .
COPY sample_candidates.json .
COPY precompute/ precompute/
COPY .streamlit/ .streamlit/

# Create artifacts directory
RUN mkdir -p artifacts

# HuggingFace Spaces expects port 7860
EXPOSE 7860

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Run Streamlit with light theme
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false", \
     "--theme.primaryColor=#4F46E5", \
     "--theme.backgroundColor=#FAFBFF", \
     "--theme.secondaryBackgroundColor=#F1F5F9", \
     "--theme.textColor=#1E293B"]
