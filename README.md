# GenAI Risk Assessment Tool

A Streamlit-based application that uses Google's Gemini AI to perform comprehensive risk assessments for generative AI projects.

> **This is a quickstart, not a production system.** It is a starting point for building agent workflows that perform analysis. This repository performs minimal threat scanning and has no authentication; run it locally rather than exposing it.

## Overview

This application helps evaluate potential risks in GenAI projects by analyzing project details, documentation, and relevant URLs. It provides detailed risk assessments across multiple risk categories, with reasoning and potential mitigations for each identified risk.

## How It Works

### Architecture

The application consists of three main components:

1. **Streamlit Front End** (`front_end/streamlit_front_end.py`)
   - Provides the user interface for data input and results display
   - Collects the inputs and displays progress as the assessment runs
   - Manages application state and user session

2. **Authentication Module** (`back_end/gemini_authentication.py`)
   - Handles Gemini API key validation and client initialization
   - Lists the models available to that key for the sidebar picker

3. **AI Agent Graph** (`back_end/adk/`)
   - A Google ADK 2.0 workflow that orchestrates the whole assessment
   - `agent.py` holds the agent itself: schemas, prompts, model config and the graph
   - `evidence.py` reads URLs and PDFs, `ratelimit.py` paces free-tier requests,
     `runner.py` bridges the graph to Streamlit

### AI Agent System

The AI agent graph runs in two phases, so no category agent ever re-reads the raw documents:

1. **Evidence Extraction**
   - Reads each supplied URL, and accepts it only if the API confirms the page was actually retrieved
   - Extracts text from each uploaded PDF locally, flagging scanned files that yield none
   - Folds everything into a single evidence brief

2. **Risk Assessment Agent**
   - Evaluates each risk category from the matrix (e.g. Model Training, Decision Making)
   - Determines the risk level from the category's own level descriptions
   - Provides reasoning and suggested mitigations
   - Returns a structured result whose risk level is constrained to that matrix's levels

   There is one agent, called once per category. It reads which category it is
   rating from session state, so the graph is the same shape for a three-row
   matrix or a fifty-row one.

3. **Report Generation** (`back_end/report.py`)
   - Validates each risk level against the matrix and sanitizes the model's free text
   - Builds the downloadable Excel report

See [agent_architecture.md](agent_architecture.md) for a diagram of the run.

### Risk Analysis Process

When a risk analysis is initiated:

1. **Data Collection**
   - User provides project details, URLs, and/or PDF documents
   - User selects or uploads a risk matrix that defines risk categories and levels

2. **Document Processing**
   - PDFs are read for text and URLs are fetched, once each
   - Sources that could not be read are reported rather than silently skipped

3. **Risk Assessment**
   - The system analyzes each risk category from the risk matrix
   - For each category, the Risk Assessment Agent:
     - Evaluates the risk level
     - Provides detailed reasoning
     - Suggests specific mitigations

4. **Results Presentation**
   - Results are displayed in an easy-to-understand format
   - Each risk category shows the assessed risk level, reasoning, and mitigations
   - Results can be exported to Excel for further analysis or reporting

## Requirements

- A valid Google Gemini API key (the free tier is sufficient)
- Python 3.11+
- Required packages: `google-adk`, `google-genai`, `streamlit`, `pandas`, `openpyxl`, `pypdf`

### About your API key

Google has retired the older standard `AIza…` API keys; as of September 2026 the Gemini API rejects them. Create a key at [Google AI Studio](https://aistudio.google.com/apikey) — new keys are issued in the supported format automatically.

Free-tier quotas are per model and tighter than they look: `gemini-3.8-flash` allows about 20 requests per **day**, and one assessment costs roughly one call per risk category. Stick with a **lite** model such as the default `gemini-3.5-flash-lite` unless you have a paid key.

## Getting Started

1. Clone the repository
2. Install dependencies with `pip install -r requirements.txt`
3. Run the application with `streamlit run front_end/streamlit_front_end.py`
4. Enter your Gemini API key and follow the instructions in the UI

## Risk Matrix

The application uses a risk matrix to define:
- Risk categories to evaluate (e.g., Model Training, Model Retention, Decision Making)
- Risk levels and their descriptions (e.g., Low, Medium, High)

Users can use the default risk matrix or upload a custom one. A custom matrix must use the same column layout: first column the risk type, second the risk description, and the third onward the risk levels from lowest to highest.
