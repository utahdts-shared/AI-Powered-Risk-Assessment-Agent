"""
Main application file for the GenAI Risk Assessment App.
This file contains the Streamlit UI and orchestrates the risk assessment process.
It provides a detailed analysis of AI risks across multiple categories with mitigations.
"""

import streamlit as st
import pandas as pd
import os
import json
import io
import time
import sys

# Add the parent directory to Python path so we can import the back_end package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules for API integration and agent functionality
from back_end.gemini_authentication import initialize_gemini_client, validate_api_key
from back_end.gemini_agents import (
    execute_gemini_request,
    extract_response_text,
    get_gemini_response,
    perform_risk_assessment
)

# Set page configuration and title
st.set_page_config(layout="wide", page_title="AI Powered Risk Analyzer", page_icon="⚡️")

# Replace the text title with a GIF image
try:
    # Check if the GIF file exists in the back_end directory
    gif_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "back_end", "V2_Title.gif")
    
    if os.path.exists(gif_path):
        # Display the image using the absolute path
        st.image(gif_path, use_container_width=False)
    
    else:
        # Fallback to text title if GIF not found
        st.title("**GenAI Risk Assessment Agent**")
        st.warning("GIF file 'V2_Title.gif' not found. Make sure it's in the same directory as this script.")
except Exception as e:
    st.title("**GenAI Risk Evaluation Agent 🥷**")
    st.error(f"Error loading GIF: {str(e)}")

# Create a sidebar for the file upload and controls
with st.sidebar:
    # API Key section at the very top of the sidebar
    st.header("Gemini Authentication ")
    # Add API key input with password masking for security
    gemini_api_key = st.text_input(
        "Enter your Gemini API Key",
        type="password",
        help="Required to use the AI features of this app. Your API key is not stored.",
        placeholder="Enter API key here...",
        key="gemini_api_key"
    )
    
    # Display important privacy notice regarding API key usage
    st.info("📝 **Note:** Please review the terms of service and privacy policy associated with your Gemini API key.")
    
    # Visual separator between authentication and configuration sections
    st.markdown("---")
    
    # Risk evaluation configuration section
    st.header("Risk Evaluation Criteria")
   
    # Add radio toggle for risk matrix selection
    st.markdown("### Select Risk Template")
    risk_matrix_option = st.radio(
        "Choose a risk matrix option:",
        ["Use State of Utah Gen-AI Risk Matrix", "Upload Custom Gen-Risk Matrix"],
        index=0  # Default to using the provided matrix
    )
    
    # Only show template download and upload sections if custom upload is selected
    if risk_matrix_option == "Upload Custom Gen-Risk Matrix":
        # Add template download option
        st.markdown("### Need a template?")
        
        # Read the template file
        template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "back_end", "risk_matrix_template.xlsx")
        
        if os.path.exists(template_path):
            with open(template_path, "rb") as template_file:
                template_bytes = template_file.read()
            
            st.download_button(
                label="Download Template",
                data=template_bytes,
                file_name="risk_matrix_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download a template to get started"
            )
            
            # Add a blue information box below the template download button
            st.info("""
            *Using a custom Risk Matrix requires following the same format on the template.*
            **NOTE:** This app can run **ANY** custom risk matrix, with more or less risk levels or categories. Follow the same template structure to ensure proper processing: \n\n\n
                first column = risk categories\n 
                second column = risk descriptions\n 
                third - ∞ columns = risk levels\n
            """)
        else:
            st.error("Template file not found. Please contact the administrator.")
                
        # Then add the file upload section
        st.markdown("### Upload Risk Matrix")
            
        # File uploader in the sidebar (only include once)
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["xlsx", "xls", "csv"])
        
        # Show upload status or instructions
        if uploaded_file is not None:
            st.success(f"Uploaded: {uploaded_file.name}")
        else:
            st.info("Please upload a risk matrix file to begin.")
        
        st.markdown("---")
    else:
        # If using default matrix, store that information
        st.success("Using the State of Utah Gen-AI Risk Matrix")
        uploaded_file = "default"  # Use a marker to indicate default matrix usage
    
    st.markdown("---")

    st.markdown("### Company / Technology Details")
    
    # Use columns to make the input fields more compact
    col1, col2 = st.columns(2)
    with col1:
        company_name = st.text_input("Company Name", placeholder="ex: Google", 
                                    help="Name of the company developing or using this technology")
    
    with col2:
        technology_name = st.text_input("Technology Name", placeholder="ex: Vertex AI Conversational Agent", 
                                       help="The specific GenAI technology being evaluated")
    
    # Important notice about required documentation for accurate risk assessment
    st.warning("⚠️ **Important:** At least 1 URL OR 1 PDF document is required for this analysis. This information helps the AI understand the context and details of your technology.")
    
    # URL input section with format validation and instructions
    st.markdown("#### Company / Technology URLs")
    st.caption("Enter URLs for more information *(URLs ending in .pdf files are not supported)*")
    
    # Collapsible URL input section to save screen space
    with st.expander("Add URLs (up to 5)"):
        # Input fields for up to 5 URLs with validation
        urls = []
        for i in range(5):
            url = st.text_input(f"URL {i+1}", key=f"url_{i}", placeholder="https://...")
            
            # Validate URL format and content type
            if url:
                if url.lower().endswith('.pdf'):
                    st.error(f"URL {i+1}: *URLs ending in .pdf files are not supported*")
                else:
                    urls.append(url)
        
        # Store validated URLs in session state for processing
        if urls:
            st.session_state['relevant_urls'] = urls
    
    # PDF file upload section
    st.markdown("#### Company / Technology PDF Files")
    st.caption("Upload PDF documents with additional information (up to 5 files, max 10MB each)")
    
    # Create expandable section for PDF uploads
    with st.expander("Upload PDF Documents (up to 5)"):
        # Create containers for up to 5 PDF uploads
        pdf_files = []
        
        for i in range(5):
            uploaded_pdf = st.file_uploader(
                f"PDF Document {i+1}", 
                type=["pdf"],
                key=f"pdf_{i}",
                accept_multiple_files=False,
                help="Upload a PDF document with additional information"
            )
            
            if uploaded_pdf is not None:
                # Validate file size (limit to 10MB)
                if uploaded_pdf.size > 10 * 1024 * 1024:  # 10MB in bytes
                    st.error(f"PDF {i+1}: File size exceeds 10MB limit")
                else:
                    # Store the PDF content and metadata
                    pdf_files.append({
                        "name": uploaded_pdf.name,
                        "size": uploaded_pdf.size,
                        "type": uploaded_pdf.type,
                        "content": uploaded_pdf.getvalue()  # Get binary content
                    })
                    st.success(f"PDF {i+1}: {uploaded_pdf.name} uploaded successfully")
    
    # Store uploaded PDFs in session state
    if pdf_files:
        st.session_state['uploaded_pdfs'] = pdf_files
        st.info(f"{len(pdf_files)} PDF document(s) uploaded successfully")
    
    st.markdown("---")

    st.markdown("### Project Details")
    
    # Project description with character count
    project_details = st.text_area(
        "Project Description", 
        placeholder="Describe the project in detail...", 
        help="Provide a comprehensive description of the project, including its purpose, scope, and objectives.",
        height=120,
        max_chars=2000
    )
    
    # Data sensitivity options
    st.markdown("#### Data Sensitivity")
    
    # Create checkboxes for data sensitivity options
    data_options = []
    
    data_sensitivity_options = [
        "Sensitive Data",
        "Sensitive Data (ALL De-Identified)",
        "ONLY Non-Sensitive Data",
    ]
    
    # Display checkboxes with dynamic keys
    for i, option in enumerate(data_sensitivity_options):
        if st.checkbox(option, key=f"data_opt_{i}"):
            data_options.append(option)
    
    # If "None of the above" is selected, clear other selections
    if "None of the above" in data_options and len(data_options) > 1:
        data_options = ["None of the above"]
        st.warning("Selected 'None of the above' - other selections have been cleared.")
    
    # Data description field
    data_description = st.text_area(
        "Data Description",
        placeholder="Describe the data being used...",
        help="Describe the data being used, including its source, format, and any special considerations.",
        height=100,
        max_chars=1000
    )
    
    # Other considerations field
    other_considerations = st.text_area(
        "Other Considerations",
        placeholder="Add any other relevant information...",
        help="Provide any additional information that might be relevant for the risk assessment.",
        height=100,
        max_chars=1000
    )
    
    # Add model configuration display
    st.markdown("---")
    st.markdown("### Default Model Configuration")
    
    # Import constants from gemini_agents
    from back_end.gemini_agents import TEMPERATURE, MAX_OUTPUT_TOKENS, MODEL_ID
    
    # Create a container for the model configuration
    config_container = st.container(border=True)
    
    with config_container:
        # Use two columns for a more compact display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model:**")
            st.markdown("**Max Output Tokens:**")
            st.markdown("**Temperature:**")
        
        with col2:
            st.markdown(f"`{MODEL_ID}`")
            st.markdown(f"`{MAX_OUTPUT_TOKENS}`")
            st.markdown(f"`{TEMPERATURE}`")
        
    # Run Risk Analysis button at the bottom of the sidebar
    st.markdown("---")
    submit_button = st.button("🚀 Run Risk Analysis", type="primary", use_container_width=True)

# Main content area
# This is where we'll display the results of the risk analysis
if submit_button:
    try:
        # Create a loading placeholder
        loading_placeholder = st.empty()
        loading_placeholder.info("Initializing risk assessment process...")
        
        # Step 1: Process the risk matrix - CHANGED TO KEEP AS DATAFRAME
        if uploaded_file == "default":
            # Load the default State of Utah Gen-AI Risk Matrix
            default_matrix_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "back_end", "ai_risk_justifications.xlsx")
            if os.path.exists(default_matrix_path):
                risk_matrix_df = pd.read_excel(default_matrix_path, engine="openpyxl")
                loading_placeholder.success("Using the default State of Utah Gen-AI Risk Matrix")
            else:
                loading_placeholder.error("Default matrix file not found. Please contact the administrator.")
                st.stop()
        elif uploaded_file is not None:
            # Get the file extension
            file_extension = os.path.splitext(uploaded_file.name)[1]
            
            # Use the correct pandas function based on the file extension
            if file_extension == ".csv":
                risk_matrix_df = pd.read_csv(uploaded_file)
            else:
                risk_matrix_df = pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            loading_placeholder.error("Please select a risk matrix option or upload a file.")
            st.stop()
        
        # Now store the DataFrame directly instead of converting to JSON
        st.session_state["risk_matrix_df"] = risk_matrix_df
        
        # Extract risk levels for reference (column headers excluding first column)
        risk_levels = risk_matrix_df.columns.tolist()[1:]
        st.session_state["risk_levels"] = risk_levels
        
        # Store the CSV version for context in agents
        risk_matrix_csv = risk_matrix_df.to_csv(index=False)
        st.session_state["risk_matrix_csv"] = risk_matrix_csv
        
        # Step 2: Process PDF files
        pdf_text_content = []
        
        # Try to process PDF files if they were uploaded
        if 'uploaded_pdfs' in st.session_state and st.session_state['uploaded_pdfs']:
            loading_placeholder.info("Processing PDF documents...")
            
            # Store PDFs for later use
            pdf_files = st.session_state['uploaded_pdfs']
            
            # Add basic information about the PDFs
            for pdf_file in pdf_files:
                pdf_text_content.append({
                    "filename": pdf_file["name"],
                    "size": pdf_file["size"]
                })
        
        # Step 3: Format URLs as comma-separated string
        formatted_urls = ""
        if 'relevant_urls' in st.session_state and st.session_state['relevant_urls']:
            formatted_urls = ", ".join(st.session_state['relevant_urls'])
        
        # Step 4: Compile all user inputs into a JSON structure
        user_inputted_project_details = {
            "company_details": {
                "company_name": company_name,
                "technology_name": technology_name,
                "urls": formatted_urls
            },
            "project_information": {
                "project_details": project_details,
                "data_sensitivity": data_options,
                "data_description": data_description,
                "other_considerations": other_considerations
            },
            "pdf_documents": [{"filename": pdf["filename"]} for pdf in pdf_text_content],
            "pdf_text_content": pdf_text_content
        }
        
        # Store user project details in session state
        st.session_state["user_inputted_project_details"] = user_inputted_project_details
        
        # Set analysis started flag
        st.session_state["analysis_started"] = True
        
        # Check if we have a valid Gemini client
        if st.session_state.gemini_client:
            # Set processing flag to start the assessment on next rerun
            st.session_state.processing = True
            st.session_state.risk_analysis_mode_active = True
            st.session_state.risk_assessment_complete = False
            st.session_state.current_agent_status = "Starting risk assessment..."
            
            loading_placeholder.success("Risk analysis started! *Please be patient while I perform analysis...*")
            
            # Force a rerun to start processing
            time.sleep(1)
            st.rerun()
        else:
            loading_placeholder.warning("Please provide a valid Gemini API key to perform risk assessment.")
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
else:
    # We don't need to display anything here since we have the welcome message at the bottom
    pass

# Initialize session state variables for app state management
if "gemini_client" not in st.session_state:
    st.session_state.gemini_client = None

if "processing" not in st.session_state:
    st.session_state.processing = False

if "current_agent_status" not in st.session_state:
    st.session_state.current_agent_status = ""

if "analysis_started" not in st.session_state:
    st.session_state["analysis_started"] = False

if "risk_assessment_results" not in st.session_state:
    st.session_state.risk_assessment_results = []

if "risk_assessment_complete" not in st.session_state:
    st.session_state.risk_assessment_complete = False

if "risk_analysis_mode_active" not in st.session_state:
    st.session_state.risk_analysis_mode_active = False

# Initialize or update Gemini client when API key is provided
if gemini_api_key:
    if not st.session_state.gemini_client or st.session_state.get("last_api_key") != gemini_api_key:
        st.session_state.gemini_client = initialize_gemini_client(gemini_api_key)
        st.session_state["last_api_key"] = gemini_api_key

# Process the request if in processing state
if st.session_state.get("processing", False) and st.session_state.gemini_client:
    # Risk analysis processing
    if st.session_state.get("risk_analysis_mode_active", False) and not st.session_state.get("risk_assessment_complete", False):
        try:
            # Get project details from session state
            project_details = st.session_state.get("user_inputted_project_details", {})
            
            # Get risk matrix from session state - now as DataFrame
            risk_matrix_df = st.session_state.get("risk_matrix_df", None)
            risk_levels = st.session_state.get("risk_levels", [])
            
            # Get PDFs if available
            pdf_files = st.session_state.get('uploaded_pdfs', [])
            
            # Get URLs if available
            urls = st.session_state.get('relevant_urls', [])
            
            # Perform risk assessment
            st.session_state.current_agent_status = "Performing risk assessment..."
            risk_results = perform_risk_assessment(
                client=st.session_state.gemini_client,
                risk_matrix_df=risk_matrix_df,
                risk_levels=risk_levels,
                project_details=project_details,
                pdf_files=pdf_files,
                urls=urls
            )
            
            # Store results in session state
            st.session_state.risk_assessment_results = risk_results
            st.session_state.risk_assessment_complete = True
            
            # Reset processing state
            st.session_state.processing = False
            st.session_state.current_agent_status = ""
            st.rerun()
            
        except Exception as e:
            st.error(f"Error in risk assessment: {str(e)}")
            
            # Reset processing state
            st.session_state.processing = False
            st.session_state.current_agent_status = ""
            st.rerun()

# Display risk analysis if that mode is active
if st.session_state.get("risk_analysis_mode_active", False):
    st.header("Risk Assessment Analysis")
    
    # Show processing status if still working - with improved visual progress indicator
    if st.session_state.get("processing", False):
        # Create a dedicated progress container
        progress_container = st.container()
        
        with progress_container:
            # Create a multi-step progress indicator
            st.markdown("""
            ### Risk assessment in progress
            The system is analyzing all risk categories based on your project details.
            This may take a few minutes depending on the complexity of your project.
            """)
            
            # Create dynamic progress tracking UI components
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Retrieve risk categories from the risk matrix to calculate progress
            risk_matrix_df = st.session_state.get("risk_matrix_df", None)
            risk_categories = []
            
            if risk_matrix_df is not None:
                # Extract risk categories from the first column (excluding header)
                risk_categories = risk_matrix_df.iloc[0:, 0].tolist()
            
            if risk_categories:
                # Extract current category from status message for progress tracking
                current_category = st.session_state.current_agent_status.replace("Assessing ", "").replace(" risk...", "")
                try:
                    # Calculate progress as percentage of completed categories
                    current_idx = risk_categories.index(current_category) if current_category in risk_categories else 0
                    progress_value = current_idx / len(risk_categories)
                    progress_bar.progress(progress_value)
                    status_text.markdown(f"**Current Category:** {current_category} ({current_idx + 1}/{len(risk_categories)})")
                except:
                    # Display indeterminate progress when category lookup fails
                    progress_bar.progress(0.1)
                    status_text.markdown(f"**{st.session_state.current_agent_status}**")
            else:
                # Show minimal progress when categories aren't available
                progress_bar.progress(0.1)
                status_text.markdown(f"**{st.session_state.current_agent_status}**")
    
    # Show results when complete
    elif st.session_state.get("risk_assessment_complete", False) and st.session_state.risk_assessment_results:
        st.success("⚡️ Risk assessment complete!")
        
        # Add download button for Excel export
        risk_results = st.session_state.risk_assessment_results
        
        # Create a function to convert risk assessment results to Excel format
        def convert_to_excel():
            """
            Converts risk assessment results to an Excel file with formatted columns.
            Returns the binary Excel file data for download.
            """
            # Create a DataFrame from risk assessment results
            data = []
            for result in risk_results:
                data.append({
                    "Risk Type": result.get("category", "Unknown"),
                    "Risk Level": result.get("risk_level", "Unknown"),
                    "Reasoning": result.get("reasoning", "No reasoning provided"),
                    "Potential Mitigations": result.get("mitigations", "No mitigations suggested")
                })
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Create a BytesIO buffer for in-memory file creation
            buffer = io.BytesIO()
            
            # Write DataFrame to Excel file with formatting
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Risk Assessment')
                
                # Auto-adjust column widths for better readability
                worksheet = writer.sheets['Risk Assessment']
                for i, col in enumerate(df.columns):
                    # Find the maximum length in the column
                    max_len = max(
                        df[col].astype(str).map(len).max(),  # max length of values
                        len(col)  # length of column name
                    ) + 2  # add a little extra space
                    
                    # Set the column width
                    worksheet.column_dimensions[chr(65 + i)].width = min(max_len, 100)  # limit max width
            
            # Get the value from the buffer
            buffer.seek(0)
            return buffer.getvalue()
        
        # Display each risk category assessment in expandable sections
        st.subheader("Detailed Risk Assessment")
        
        # Retrieve risk levels for visual color coding of risk severity
        risk_levels = st.session_state.get("risk_levels", [])
        
        for result in st.session_state.risk_assessment_results:
            category = result.get("category", "Unknown Category")
            risk_level = result.get("risk_level", "Unknown")
            reasoning = result.get("reasoning", "No reasoning provided")
            mitigations = result.get("mitigations", "No mitigations suggested")
            
            # Apply color coding based on risk level severity
            # Maps risk levels from lowest (green) to highest (red)
            if risk_level in risk_levels:
                level_idx = risk_levels.index(risk_level)
                level_percent = level_idx / max(1, len(risk_levels) - 1)
                
                if level_percent <= 0.25:
                    color = "green"  # Low risk
                elif level_percent <= 0.5:
                    color = "blue"  # Medium-low risk
                elif level_percent <= 0.75:
                    color = "orange"  # Medium-high risk
                else:
                    color = "red"  # High risk
            else:
                color = "gray"  # Unknown risk level
            
            # Create expandable section for each risk category
            with st.expander(f"{category} - Risk Level: {risk_level}", expanded=False):
                st.markdown(f"**Risk Rating:** :{color}[{risk_level}]")
                
                st.markdown("**Reasoning:**")
                st.write(reasoning)
                
                st.markdown("**Potential Mitigations:**")
                st.write(mitigations)
        
        # Add download button after all risk categories are displayed
        st.markdown("---")
        
        # Create a container for the download button with better visual separation
        download_container = st.container()
        with download_container:
            # Extract company and technology names for the filename
            company_name = st.session_state.get("user_inputted_project_details", {}).get("company_details", {}).get("company_name", "Company")
            technology_name = st.session_state.get("user_inputted_project_details", {}).get("company_details", {}).get("technology_name", "Technology")
            
            # Create a sanitized filename from company and technology names
            safe_company = ''.join(c if c.isalnum() else '_' for c in company_name)
            safe_technology = ''.join(c if c.isalnum() else '_' for c in technology_name)
            filename = f"{safe_company}_{safe_technology}_Risk_Assessment.xlsx"
            
            # Add the download button with clear labeling and primary styling
            st.download_button(
                label="📥 Download Detailed Risk Assessment (.xlsx)",
                data=convert_to_excel(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download the complete risk assessment results as an Excel file",
                type="primary",
                use_container_width=True
            )
    
    # Show message if no results yet
    else:
        st.warning("No risk assessment results available yet. Please start the analysis.")
            
# Display a message that the analysis is complete and provide button to view results
elif st.session_state.get("risk_assessment_complete", False):
    # Create a container for the risk assessment results
    results_container = st.container()
    
    with results_container:
        st.header("Risk Assessment Complete")
        
        # Get risk assessment results
        risk_results = st.session_state.get("risk_assessment_results", [])
        
        if not risk_results:
            st.warning("No risk assessment results found. Please try running the analysis again.")
        else:
            # Create a function to convert results to Excel (same as above)
            def convert_to_excel():
                # Create a DataFrame from risk assessment results
                data = []
                for result in risk_results:
                    data.append({
                        "Risk Type": result.get("category", "Unknown"),
                        "Risk Level": result.get("risk_level", "Unknown"),
                        "Reasoning": result.get("reasoning", "No reasoning provided"),
                        "Potential Mitigations": result.get("mitigations", "No mitigations suggested")
                    })
                
                # Create DataFrame
                df = pd.DataFrame(data)
                
                # Create a BytesIO buffer
                buffer = io.BytesIO()
                
                # Write DataFrame to Excel file
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Risk Assessment')
                    
                    # Auto-adjust column widths
                    worksheet = writer.sheets['Risk Assessment']
                    for i, col in enumerate(df.columns):
                        # Find the maximum length in the column
                        max_len = max(
                            df[col].astype(str).map(len).max(),  # max length of values
                            len(col)  # length of column name
                        ) + 2  # add a little extra space
                        
                        # Set the column width
                        worksheet.column_dimensions[chr(65 + i)].width = min(max_len, 100)  # limit max width
                
                # Get the value from the buffer
                buffer.seek(0)
                return buffer.getvalue()
            
            # Create filename based on company and technology
            company_name = st.session_state.get("user_inputted_project_details", {}).get("company_details", {}).get("company_name", "Company")
            technology_name = st.session_state.get("user_inputted_project_details", {}).get("company_details", {}).get("technology_name", "Technology")
            
            # Create filename based on company and technology
            safe_company = ''.join(c if c.isalnum() else '_' for c in company_name)
            safe_technology = ''.join(c if c.isalnum() else '_' for c in technology_name)
            filename = f"{safe_company}_{safe_technology}_Risk_Assessment.xlsx"
            
            # Add buttons in columns
            col1, col2 = st.columns(2)
            
            with col1:
                # Button to view risk assessment
                if st.button("📊 View Risk Assessment Results", type="primary", use_container_width=True):
                    st.session_state.risk_analysis_mode_active = True
                    st.rerun()
            
            with col2:
                # Add the download button
                st.download_button(
                    label="📥 Download as Excel",
                    data=convert_to_excel(),
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Download the complete risk assessment results as an Excel file",
                    type="primary",
                    use_container_width=True
                )
else:
    # Display a welcome message
    st.markdown("""
    ## Welcome to the GenAI Risk Assessment Tool
    
    This tool helps you evaluate potential risks in your GenAI projects by analyzing project details,
    documentation, and relevant URLs. Follow these steps to get started:
    
    1. Enter your Gemini API key in the sidebar
    2. Fill in your project details
    3. Upload PDF documentation or provide URLs
    4. Click "Run Risk Analysis" to begin
    
    The analysis will evaluate your project across multiple risk categories and provide
    detailed reasoning and potential mitigations for each identified risk.
    """)
