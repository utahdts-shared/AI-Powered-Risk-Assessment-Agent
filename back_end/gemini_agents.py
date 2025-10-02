"""
Gemini AI agents for risk assessment.
Implements various specialized agents for risk analysis with robust error handling
and structured response processing.
"""

import json
import time
import re
import streamlit as st
import tempfile
import pathlib
import os
import sys

# Add the parent directory to Python path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Gemini authentication module
from back_end.gemini_authentication import initialize_gemini_client

# Import Google Generative AI libraries
from google import genai
from google.genai.types import Tool, GenerateContentConfig, UrlContext

# Constants for agent configuration
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 6144
MODEL_ID = "gemini-2.5-flash"  # Using Gemini 2.5 Flash for optimal performance

# Helper function to extract text from API responses
def extract_response_text(response):
    """
    Extracts text from a Gemini API response with multiple fallback options.
    Handles various response structures to retrieve the text content.
    
    Args:
        response: The Gemini API response object
        
    Returns:
        String: The extracted text or an empty string if none found
    """
    # Direct text property
    if hasattr(response, 'text') and response.text is not None:
        return response.text
    
    # Try to extract from candidates
    if hasattr(response, 'candidates') and response.candidates:
        for candidate in response.candidates:
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text is not None:
                        return part.text
    
    # If nothing works, return empty string
    return ""

# Create a wrapper function to handle all Gemini API calls
def execute_gemini_request(client, model, contents, config, agent_name="Gemini API", retries=3, delay=2, backoff_factor=5):
    """
    Comprehensive wrapper function for Gemini API calls with robust error handling.
    Provides retry logic with exponential backoff, checks for blocked prompts, 
    validates response content, and handles various error conditions.
    
    Args:
        client: The Gemini API client
        model: The model ID to use
        contents: The content to send to the API
        config: The API configuration
        agent_name: Name of the agent for display purposes
        retries: Number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for exponential backoff
        
    Returns:
        Tuple of (success_bool, response_or_error, is_error_bool)
    """
    for attempt in range(retries):
        try:
            # Make the API call
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )

            # Check for immediate prompt blocking
            if hasattr(response, 'prompt_feedback') and hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                error_msg = f"Prompt blocked due to: {response.prompt_feedback.block_reason}"
                st.error(error_msg)
                if attempt < retries - 1:
                    st.warning(f"Prompt blocked. Modifying and retrying ({attempt+1}/{retries})...")
                    time.sleep(delay * (backoff_factor ** attempt))
                    continue
                return False, error_msg, True

            # Check for finish_reason in the candidate
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    # Store finish reason for visibility
                    finish_reason = candidate.finish_reason
                    
                    # Check if finish reason indicates a problem
                    if finish_reason not in ["STOP", "MAX_TOKENS"]:
                        # This indicates a block on the *output*
                        error_msg = f"API call failed after generation. Finish Reason: {finish_reason}"
                        st.warning(error_msg)
                        if attempt < retries - 1:
                            st.info(f"Retrying ({attempt + 1}/{retries})...")
                            time.sleep(delay * (backoff_factor ** attempt))
                            continue
                        return False, error_msg, True
            
            # Extract text using the helper function
            response_text = extract_response_text(response)

            # Check if the extracted text is empty
            if not response_text or not response_text.strip():
                # Get the finish reason if available for better error reporting
                finish_reason = "UNKNOWN"
                if hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0], 'finish_reason'):
                    finish_reason = response.candidates[0].finish_reason
                    
                error_msg = f"API returned an empty response. Finish reason: {finish_reason}"
                if attempt < retries - 1:
                    st.warning(f"{error_msg} Retrying ({attempt + 1}/{retries})...")
                    time.sleep(delay * (backoff_factor ** attempt))
                    continue
                return False, error_msg, True

            # Success case
            st.success(f"Utilizing the {agent_name}, analyzing response...")
            return True, response, False
            
        except Exception as e:
            # Handle rate limiting errors specifically
            if "429" in str(e) or "rate limit" in str(e).lower():
                if attempt < retries - 1:
                    wait_time = delay * (backoff_factor ** attempt)
                    st.warning(f"Rate limit hit. Waiting {wait_time}s before retry ({attempt + 1}/{retries})...")
                    time.sleep(wait_time)
                    continue
                return False, f"Rate limit exceeded: {str(e)}", True
                
            # For other errors, retry with backoff
            if attempt < retries - 1:
                st.warning(f"API error: {str(e)}. Retrying ({attempt + 1}/{retries})...")
                time.sleep(delay * (backoff_factor ** attempt))
                continue
            
            # If all retries fail
            return False, f"API error after {retries} retries: {str(e)}", True
    
    # This should never be reached, but just in case
    return False, "Maximum retries exceeded.", True

# Function to get response from Gemini Model API
def get_gemini_response(client, user_message, chat_history=None):
    """Get response from Gemini Model API"""
    # Prepare the conversation
    contents = []
    
    # Add chat history if available
    if chat_history and len(chat_history) > 0:
        for msg in chat_history:
            contents.append(msg['content'])
    
    # Add the new message
    contents.append(user_message)
    
    try:
        # Generate content with Gemini
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=contents,
            config=GenerateContentConfig(
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                response_modalities=["TEXT"],
            )
        )
        
        return extract_response_text(response)
    except Exception as e:
        st.error(f"Error getting response from Gemini Model API: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}"

# URL Agent using Gemini
def url_agent(client, urls, project_details, risk_matrix_csv, user_question):
    """Researches information from URLs in the context of project details"""
    
    # Construct the system prompt
    system_prompt = (
        "System Instructions: You are a URL Research Agent specialized in analyzing web content "
        "for risk assessment projects. Use the provided URLs to find information relevant to the user's question. "
        f"\n\n**Project Details:**\n{json.dumps(project_details, indent=2)}\n\n"
    )
    
    # Add risk matrix as CSV if available
    if risk_matrix_csv:
        system_prompt += f"**Risk Matrix (CSV format):**\n```\n{risk_matrix_csv}\n```\n\n"
    
    # Add URLs if available
    if urls and len(urls) > 0:
        system_prompt += f"**URLs for Analysis:**\n{', '.join(urls)}\n\n"
    else:
        system_prompt += "**URLs for Analysis:** No URLs were provided for analysis.\n\n"
        
    system_prompt += (
        f"**User Question:**\n{user_question}\n\n"
        "Focus on extracting factual information from the URLs that relates to the user's question and the project details. "
        "Your research will be used by another agent to compile a final response, so be thorough and precise."
    )
    
    try:
        # Create URL context tool
        url_context_tool = Tool(url_context=UrlContext())
        
        # Prepare the API config with URL tool
        config = GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            tools=[url_context_tool],
            response_modalities=["TEXT"],
        )
        
        # Make the API call using our wrapper
        success, response_or_error, is_error = execute_gemini_request(
            client=client,
            model=MODEL_ID,
            contents=system_prompt,
            config=config,
            agent_name="URL Research Agent"
        )
        
        # Handle API call results
        if not success:
            # Log the error and return a structured error response
            st.error(f"Error in URL Agent: {response_or_error}")
            return f"Error in URL research: {response_or_error}", None, None
        
        # Process the successful response
        response = response_or_error
        
        # Check for URL metadata in the response
        url_metadata = None
        if hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0], 'url_context_metadata'):
            url_metadata = response.candidates[0].url_context_metadata
        
        # Extract the response text
        response_text = extract_response_text(response)
        
        # Just return the response text directly
        return response_text.strip() if response_text else "No response was generated from URL analysis.", None, url_metadata
    
    except Exception as e:
        st.error(f"Error in URL Agent: {str(e)}")
        return f"Error in URL research: {str(e)}", None, None

# Compiling Agent using Gemini
def compiling_agent(client, pdf_agent_result, url_agent_result, url_metadata, project_details, risk_matrix_csv, user_question):
    """Compiles results from PDF and URL agents to provide a comprehensive answer"""
    
    # Construct the system prompt
    system_prompt = (
        "System Instructions: You are a Risk Assessment Compiling Agent responsible for synthesizing information "
        "from multiple sources to provide comprehensive answers about project risks. "
        f"\n\n**Project Details:**\n{json.dumps(project_details, indent=2)}\n\n"
    )
    
    # Add risk matrix as CSV if available
    if risk_matrix_csv:
        system_prompt += f"**Risk Matrix (CSV format):**\n```\n{risk_matrix_csv}\n```\n\n"
    
    system_prompt += f"**User Question:**\n{user_question}\n\n"
    
    # Add PDF agent results if available
    if pdf_agent_result:
        system_prompt += f"**PDF Analysis Results:**\n{pdf_agent_result}\n\n"
    else:
        system_prompt += "**PDF Analysis Results:** No PDF data was provided or analyzed.\n\n"
    
    # Add URL agent results if available
    if url_agent_result:
        system_prompt += f"**URL Research Results:**\n{url_agent_result}\n\n"
        if url_metadata:
            system_prompt += f"**URL Metadata:**\n{str(url_metadata)}\n\n"
    else:
        system_prompt += "**URL Research Results:** No URL data was provided or analyzed.\n\n"
    
    system_prompt += (
        "Synthesize the information from all available sources to provide a clear, comprehensive answer to the user's question. "
        "Focus on identifying risks, their implications, and potential mitigations based on the available information. "
        "Present your answer in a structured, easy-to-understand format."
    )
    
    # Prepare the API config
    config = GenerateContentConfig(
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_modalities=["TEXT"],
    )
    
    try:
        # Make the API call using our wrapper
        success, response_or_error, is_error = execute_gemini_request(
            client=client,
            model=MODEL_ID,
            contents=system_prompt,
            config=config,
            agent_name="Compiling Agent"
        )
        
        # Handle API call results
        if not success:
            # Log the error and return a structured error response
            st.error(f"Error in Compiling Agent: {response_or_error}")
            return f"Error in compiling results: {response_or_error}", None
        
        # Process the successful response
        response = response_or_error
        
        # Extract the response text
        response_text = extract_response_text(response)
        
        # Return the answer
        answer_text = response_text.strip() if response_text else "No response was generated from compiling agent."
            
        return answer_text, None
        
    except Exception as e:
        st.error(f"Error in Compiling Agent: {str(e)}")
        return f"Error in compiling results: {str(e)}", None

# Risk Assessment Agent using Gemini
def risk_assessment_agent(client, risk_category, risk_levels, category_explanations, project_details, pdf_data=None, urls=None):
    """Agent that assesses a specific risk category based on project details and reference materials"""
    
    # Construct the system prompt
    system_prompt = (
        f"System Instructions: You are a Risk Assessment Agent specialized in evaluating '{risk_category}' risks "
        f"for GenAI projects. Your task is to determine the appropriate risk level for this specific risk category "
        f"based on the project details provided.\n\n"
        
        f"**Risk Category:** {risk_category}\n\n"
        
        f"**Available Risk Levels (from lowest to highest risk):**\n"
    )
    
    # Add risk level explanations for this specific category
    for level in risk_levels:
        explanation = category_explanations.get(level, "No explanation provided")
        system_prompt += f"- **{level}**: {explanation}\n"
    
    # Add project details
    system_prompt += f"\n\n**Project Details:**\n{json.dumps(project_details, indent=2)}\n\n"
    
    # Add PDF content if available
    if pdf_data:
        # Truncate to avoid token limits
        truncated_pdf_data = pdf_data[:50000] if len(pdf_data) > 50000 else pdf_data
        system_prompt += f"**PDF Content:**\n{truncated_pdf_data}\n\n"
    
    # Add URLs if available
    if urls:
        system_prompt += f"**URLs for Reference:**\n{urls}\n\n"
    
    system_prompt += (
        "Based on all this information, assess the risk level for this category. "
        "Provide your response in this exact JSON format:\n"
        "{\n"
        '  "risk_level": "THE EXACT RISK LEVEL NAME",\n'
        '  "reasoning": "2-10 sentences explaining your reasoning (concise explanations are better)",\n'
        '  "mitigations": "2-10 sentences suggesting potential mitigations (concise explanations are better)"\n'
        "}\n\n"
        "Important: Your reasoning and mitigations must be specific to this risk category and the project details provided. "
        "The risk_level must be exactly one of the provided risk level names."
    )
    
    # Prepare the API config
    config = GenerateContentConfig(
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_modalities=["TEXT"],
    )
    
    try:
        # Make the API call using our wrapper
        success, response_or_error, is_error = execute_gemini_request(
            client=client,
            model=MODEL_ID,
            contents=system_prompt,
            config=config,
            agent_name=f"Risk Assessment Agent - Investigating: {risk_category}"
        )
        
        # Handle API call results
        if not success:
            # Log the error and return a structured error response
            st.error(f"Error in Risk Assessment Agent for {risk_category}: {response_or_error}")
            return {
                "risk_level": "Error",
                "reasoning": f"Error in assessment for {risk_category}: {response_or_error}",
                "mitigations": "Please try again or assess manually."
            }
        
        # Process the successful response
        response = response_or_error
        
        # Extract the response text
        response_text = extract_response_text(response)
        
        # Handle API call results
        if not success:
            # Log the error
            st.error(f"Error in Risk Assessment Agent for {risk_category}: {response_or_error}")
            return {
                "risk_level": "Error",
                "reasoning": f"Error processing this risk category: {response_or_error}",
                "mitigations": "Please try again or assess manually."
            }
        
        # Process the successful response
        response = response_or_error
        
        # Extract the response text
        response_text = extract_response_text(response)
        
        # Parse the JSON response - First, find JSON content between curly braces
        json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                # Ensure the result has all required fields
                if not all(k in result for k in ["risk_level", "reasoning", "mitigations"]):
                    missing = [k for k in ["risk_level", "reasoning", "mitigations"] if k not in result]
                    result = {
                        **result,
                        **{k: f"Missing {k} in response" for k in missing}
                    }
                
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, create a structured error response
                return {
                    "risk_level": "Error",
                    "reasoning": "Failed to parse JSON response from model",
                    "mitigations": "Please try again or assess manually.",
                    "raw_response": response_text[:500] + "..." if len(response_text) > 500 else response_text
                }
        else:
            # If no JSON found, create a structured response with the raw text
            return {
                "risk_level": "Error",
                "reasoning": "No JSON format found in response",
                "mitigations": "Please try again or assess manually.",
                "raw_response": response_text[:500] + "..." if len(response_text) > 500 else response_text
            }
    
    except Exception as e:
        st.error(f"Error in Risk Assessment Agent for {risk_category}: {str(e)}")
        return {
            "risk_level": "Error",
            "reasoning": f"Error processing this risk category: {str(e)}",
            "mitigations": "Please try again or assess manually."
        }

# Perform Risk Assessment function
def perform_risk_assessment(client, risk_matrix_df, risk_levels, project_details, pdf_files=None, urls=None):
    """
    Evaluates all risk categories in the risk matrix and returns a comprehensive assessment.
    
    Args:
        client: The Gemini API client
        risk_matrix_df: DataFrame containing risk categories and level descriptions
        risk_levels: List of risk levels from lowest to highest
        project_details: Dictionary of project information for assessment
        pdf_files: Optional list of PDF files for additional context
        urls: Optional list of URLs for additional context
        
    Returns:
        List of dictionaries with risk assessment results for each category
    """
    
    risk_assessment_results = []
    
    # Extract risk categories from the matrix
    risk_categories = risk_matrix_df.iloc[0:, 0].tolist()
    
    # Convert risk matrix to CSV format for API context
    risk_matrix_csv = risk_matrix_df.to_csv(index=False)
    
    # Process PDF data if available
    pdf_data = None
    processed_pdf_files = []
    
    # Handle PDF file processing for additional context
    if pdf_files and len(pdf_files) > 0:
        try:
            # Update status and process PDFs
            st.session_state.current_agent_status = "Processing PDFs..."
            
            # Only process up to 3 PDFs to avoid overwhelming the API
            if len(pdf_files) > 3:
                st.warning(f"Processing only the first 3 of {len(pdf_files)} PDF files to ensure stability. The remaining files will be skipped.")
                pdf_files = pdf_files[:3]
            
            # Process the PDFs with delay between each
            st.info(f"Processing {len(pdf_files)} PDF files. This may take a few minutes...")
            
            # Get PDF parts for Gemini API
            pdf_parts = process_pdfs_with_gemini_file_api(client, pdf_files, risk_matrix_df)
            
            # Prepare PDF data summary for context
            if pdf_parts:
                pdf_data = f"Successfully processed {len(pdf_parts)} PDF files for analysis."
                
                # Create a simple list of PDF names for reference
                pdf_summary = []
                for i, pdf_file in enumerate(pdf_files[:len(pdf_parts)]):
                    file_name = pdf_file.get('name', f'PDF {i+1}')
                    pdf_summary.append(f"File: {file_name}")
                
                # Add the PDF summary to pdf_data
                if pdf_summary:
                    pdf_data += "\n\n## PDF Files Processed:\n" + "\n".join(pdf_summary)
            else:
                pdf_data = "No PDF files were successfully processed."
                
        except Exception as e:
            st.error(f"Error processing PDFs: {str(e)}")
            pdf_data = f"Error processing PDFs: {str(e)}"
    
    # SIMPLIFIED URL HANDLING - Always use a comma-separated string
    formatted_urls = ""
    
    # Try to get URLs from direct input
    if urls:
        if isinstance(urls, list):
            # Join all non-empty values with commas
            formatted_urls = ", ".join([str(url) for url in urls if url])
        elif isinstance(urls, str):
            formatted_urls = urls
    
    # If we don't have URLs yet, try from project_details as backup
    if not formatted_urls and 'company_details' in project_details and 'urls' in project_details['company_details']:
        url_data = project_details['company_details']['urls']
        if url_data:
            if isinstance(url_data, list):
                formatted_urls = ", ".join([str(url) for url in url_data if url])
            elif isinstance(url_data, str):
                formatted_urls = url_data
    
    # Add a small delay between API calls to avoid rate limits
    delay_seconds = 1
    
    # Process each risk category (row in the DataFrame)
    for i, category in enumerate(risk_categories):
        # Update status (without causing reruns)
        st.session_state.current_agent_status = f"Assessing {category} risk..."
        
        # Get explanations for this category from the DataFrame
        # For each risk level (column), get the corresponding explanation
        category_row_index = i  # Index in the DataFrame (row number)
        category_explanations = {}
        
        for j, level in enumerate(risk_levels):
            explanation = str(risk_matrix_df.iloc[category_row_index, j+1])  # +1 to skip the first column
            category_explanations[level] = explanation
        
        try:
            # Call the risk assessment agent
            assessment = risk_assessment_agent(
                client=client,
                risk_category=category,
                risk_levels=risk_levels,
                category_explanations=category_explanations,
                project_details=project_details,
                pdf_data=pdf_data if pdf_data else None,
                urls=formatted_urls if formatted_urls else None
            )
            
            # Add category to the result
            assessment["category"] = category
            risk_assessment_results.append(assessment)
            
            # Add a small delay between API calls to avoid hitting rate limits
            time.sleep(delay_seconds)
        except Exception as e:
            st.error(f"Error assessing {category} risk: {str(e)}")
            # Create an error result
            assessment = {
                "category": category,
                "risk_level": "Error",
                "reasoning": f"Error processing this risk category: {str(e)}",
                "mitigations": "Please try again or assess manually."
            }
            risk_assessment_results.append(assessment)
            # Still add delay before next call
            time.sleep(delay_seconds)
    
    return risk_assessment_results

# Function to process PDFs with Gemini
def process_pdfs_with_gemini_file_api(client, pdf_files, risk_matrix_df=None):
    """
    Uploads PDFs directly to Gemini using the File API
    
    Args:
        client: The Gemini client instance
        pdf_files: List of PDF file dictionaries from Streamlit uploader
        risk_matrix_df: Optional parameter for compatibility, not used in this implementation
        
    Returns:
        List of uploaded file objects that can be used in Gemini API calls
    """
    uploaded_gemini_files = []
    
    # Process each PDF file
    for idx, pdf_file in enumerate(pdf_files):
        try:
            file_name = pdf_file["name"]
            file_content = pdf_file["content"]
            
            # Calculate file size in MB for informational purposes
            file_size_mb = len(file_content) / (1024 * 1024)
            
            # Skip very large files
            if file_size_mb > 10:  # If file is larger than 10MB
                st.error(f"Skipping {file_name} (size: {file_size_mb:.1f}MB) - exceeds 10MB limit for reliable processing.")
                continue
                
            st.info(f"Processing PDF {idx+1} of {len(pdf_files)}: {file_name} ({file_size_mb:.1f}MB)")
            
            # Add delay between PDF processing to avoid rate limits
            if idx > 0:
                time.sleep(2)  # 2-second delay between PDFs
            
            # Create a temporary file for the PDF
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            
            # Create a Path object from the temp file
            file_path = pathlib.Path(tmp_path)
            
            try:
                # Create the PDF part using the correct method
                pdf_part = genai.types.Part.from_bytes(
                    data=open(tmp_path, 'rb').read(),
                    mime_type='application/pdf'
                )
                
                # Add the part to our list
                uploaded_gemini_files.append(pdf_part)
                
            except Exception as e:
                st.error(f"Error processing {file_name}: {str(e)}")
            
            finally:
                # Clean up the temporary file
                try:
                    file_path.unlink(missing_ok=True)
                except Exception as cleanup_error:
                    pass  # Ignore cleanup errors
                    
        except Exception as e:
            st.error(f"Error processing {pdf_file.get('name', 'unknown file')}: {str(e)}")
    
    return uploaded_gemini_files
