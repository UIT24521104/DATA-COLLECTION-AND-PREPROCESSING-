# Multi-Stage Dockerfile for WorldBank Data Analysis Project
# Supports: DIAGNOSTIC_PIPELINE, STREAMLIT_APP, TRAJECTORY_APP

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app/

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Create entrypoint script
RUN cat > /app/entrypoint.sh << 'EOF'
#!/bin/bash

# Default command
CMD="${1:-diagnostic}"

case "$CMD" in
    # ============================================================================
    # 1. DIAGNOSTIC PIPELINE: Data Integration & Auto-Profiling
    # ============================================================================
    diagnostic|pipeline)
        echo "🔍 Starting DIAGNOSTIC_PIPELINE..."
        cd /app/DIAGNOSTIC_PIPELINE
        SCENARIO="${2:-default}"
        echo "📊 Running: python run_pipeline.py full --scenario $SCENARIO"
        python run_pipeline.py full --scenario "$SCENARIO"
        ;;
    
    # ============================================================================
    # 2. STREAMLIT APP: Interactive Dashboard
    # ============================================================================
    streamlit|dashboard|main)
        echo "📈 Starting STREAMLIT_APP..."
        cd /app/STREAMLIT_APP
        PORT="${2:-8501}"
        echo "🌐 Dashboard running on port $PORT"
        echo "📍 Open: http://localhost:$PORT"
        streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
        ;;
    
    # ============================================================================
    # 3. TRAJECTORY APP: Trajectory Analysis Dashboard
    # ============================================================================
    trajectory)
        echo "🚀 Starting TRAJECTORY_APP..."
        cd /app/TRAJECTORY_APP
        PORT="${2:-8502}"
        echo "🌐 Trajectory app running on port $PORT"
        echo "📍 Open: http://localhost:$PORT"
        streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
        ;;
    
    # ============================================================================
    # HELP & DOCUMENTATION
    # ============================================================================
    help|--help|-h)
        cat << HELP_TEXT

╔════════════════════════════════════════════════════════════════════════════════╗
║         WorldBank Data Analysis - Docker Entrypoint Commands                   ║
╚════════════════════════════════════════════════════════════════════════════════╝

📌 SYNTAX:
   docker run [OPTIONS] IMAGE COMMAND [ARGS]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 1. DIAGNOSTIC PIPELINE
   Purpose: Data integration & auto-profiling
   
   docker run -v $(pwd)/outputs:/app/DIAGNOSTIC_PIPELINE/outputs \\
     worldbank:latest diagnostic [scenario]
   
   Examples:
     docker run worldbank:latest diagnostic default
     docker run worldbank:latest diagnostic test
   
   Output: DIAGNOSTIC_PIPELINE/outputs/runs/*/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 2. STREAMLIT APP (Main Dashboard)
   Purpose: Interactive data exploration & diagnostics
   
   docker run -p 8501:8501 worldbank:latest streamlit [port]
   
   Examples:
     docker run -p 8501:8501 worldbank:latest streamlit
     docker run -p 9000:9000 worldbank:latest streamlit 9000
   
   Access: http://localhost:8501

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 3. TRAJECTORY APP (Similarity Analysis)
   Purpose: Vietnam trajectory vs similar countries
   
   docker run -p 8502:8502 worldbank:latest trajectory [port]
   
   Examples:
     docker run -p 8502:8502 worldbank:latest trajectory
     docker run -p 9001:9001 worldbank:latest trajectory 9001
   
   Access: http://localhost:8502

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 COMMON USE CASES:

1️⃣  Run pipeline + save outputs:
    docker run -v $(pwd)/outputs:/app/DIAGNOSTIC_PIPELINE/outputs \\
      worldbank:latest diagnostic default

2️⃣  Run both Streamlit apps (separate terminals):
    Terminal 1:
      docker run -p 8501:8501 worldbank:latest streamlit
    
    Terminal 2:
      docker run -p 8502:8502 worldbank:latest trajectory

3️⃣  Run with custom port mapping:
    docker run -p 9000:8501 worldbank:latest streamlit 8501

4️⃣  Run with shell access (troubleshooting):
    docker run -it worldbank:latest /bin/bash

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔧 ALIASES:
   Diagnostic: diagnostic, pipeline
   Streamlit:  streamlit, dashboard, main
   Trajectory: trajectory
   Help:       help, --help, -h

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📖 More info: See DOCKER_SETUP.md

HELP_TEXT
        ;;
    
    # ============================================================================
    # UNKNOWN COMMAND
    # ============================================================================
    *)
        echo "❌ Unknown command: $CMD"
        echo ""
        echo "Available commands:"
        echo "  • diagnostic [scenario]    - Run DIAGNOSTIC_PIPELINE"
        echo "  • streamlit [port]         - Run STREAMLIT_APP"
        echo "  • trajectory [port]        - Run TRAJECTORY_APP"
        echo "  • help                     - Show this help message"
        echo ""
        echo "Run 'docker run IMAGE help' for detailed examples"
        exit 1
        ;;
esac
EOF

RUN chmod +x /app/entrypoint.sh && sed -i 's/\r$//' /app/entrypoint.sh

# Expose ports for Streamlit apps
EXPOSE 8501 8502

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (can be overridden)
CMD ["streamlit"]
