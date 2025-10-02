# GenAI Risk Assessment Tool

A Streamlit-based application that uses Google's Gemini AI to perform comprehensive risk assessments for generative AI projects.

## Overview

This application helps evaluate potential risks in GenAI projects by analyzing project details, documentation, and relevant URLs. It provides detailed risk assessments across multiple risk categories, with reasoning and potential mitigations for each identified risk.

## How It Works

### Architecture

The application consists of three main components:

1. **Streamlit Front End** (`front_end/streamlit_front_end.py`)
   - Provides the user interface for data input and results display
   - Orchestrates the overall assessment process
   - Manages application state and user session

2. **Authentication Module** (`back_end/gemini_authentication.py`)
   - Handles Gemini API key validation and client initialization
   - Manages secure connection to Google's Generative AI services

3. **AI Agents** (`back_end/gemini_agents.py`)
   - Implements specialized AI agents for different aspects of risk assessment
   - Processes PDFs and URLs to extract relevant information
   - Analyzes risk categories based on project details and reference materials

### Agent System

The application uses a multi-agent approach, with specialized agents working together:

1. **URL Agent**
   - Analyzes web content from provided URLs
   - Extracts information relevant to risk assessment
   - Formats findings for the Compiling Agent

2. **Risk Assessment Agent**
   - Evaluates specific risk categories (e.g., Privacy, Ethics, Security)
   - Determines appropriate risk levels based on project details
   - Provides reasoning and suggested mitigations
   - Returns structured assessment for each category

3. **Compiling Agent**
   - Synthesizes information from multiple sources
   - Combines PDF analysis and URL research
   - Provides comprehensive risk insights

4. **PDF Processing**
   - Handles PDF document uploads
   - Converts PDFs to a format that Gemini can analyze
   - Extracts relevant content for risk assessment

### Risk Analysis Process

When a risk analysis is initiated:

1. **Data Collection**
   - User provides project details, URLs, and/or PDF documents
   - User selects or uploads a risk matrix that defines risk categories and levels

2. **Document Processing**
   - PDFs are processed and converted to a format that Gemini can analyze
   - URLs are validated and prepared for analysis

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

- A valid Google Gemini API key
- Python 3.7+
- Required packages: streamlit, google-generativeai, pandas, and others

## Getting Started

1. Clone the repository
2. Install dependencies with `pip install -r requirements.txt`
3. Run the application with `streamlit run streamlit_front_end.py`
4. Enter your Gemini API key and follow the instructions in the UI

## Risk Matrix

The application uses a risk matrix to define:
- Risk categories to evaluate (e.g., Data Privacy, Bias, Security)
- Risk levels and their descriptions (e.g., Low, Medium, High)

Users can use the default risk matrix or upload a custom one.
