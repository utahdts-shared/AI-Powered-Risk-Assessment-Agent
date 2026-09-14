# Notice:
# Carter Saar. Copyright (C) 2025 State of Utah
# Licensed under Apache License, Version 2.0 (Apache v2). This program is distributed on an "AS IS" BASIS, WITHOUT ANY WARRANTY OR CONDITIONS OF ANY KIND, either express or implied. See the Apache License, Version 2.0 (Apache v2) for more details.

"""
Main application file for the GenAI Risk Assessment App.
This file contains the Streamlit UI and orchestrates the risk assessment process.
It provides a detailed analysis of AI risks across multiple categories with mitigations.
"""

import os
import sys
import time

import streamlit as st

# Required, not leftover: `streamlit run` puts this file's own directory on
# sys.path rather than the repository root, so `back_end` is otherwise
# unimportable. Replacing this needs a real entry point such as a console script.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules for API integration and agent functionality
from back_end.adk.agent import AssessmentInputs
from back_end.adk.runner import run_assessment
from back_end.gemini_authentication import (
    ApiKeyError,
    create_client,
    list_available_models,
    normalize_api_key,
)
from back_end.matrix_loader import (
    MAX_UPLOAD_BYTES,
    MatrixError,
    load_matrix,
    load_matrix_path,
    risk_level_color,
)
from back_end.matrix_loader import (
    risk_levels as parse_risk_levels,
)
from back_end.report import build_report_workbook, safe_label, sanitize_result

# Set page configuration and title
st.set_page_config(layout="wide", page_title="AI Powered Risk Analyzer", page_icon="⚡️")

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

if "available_models" not in st.session_state:
    st.session_state.available_models = []

# Replace the text title with a GIF image
try:
    # Check if the GIF file exists in the back_end directory
    gif_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "back_end", "V2_Title.gif")

    if os.path.exists(gif_path):
        # Display the image using the absolute path
        st.image(gif_path, width="content")

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

    # Build and validate the client here, inside the sidebar and before the
    # submit handler runs, so pasting a key and clicking Run in the same rerun
    # works on the first click.
    if gemini_api_key:
        if (st.session_state.gemini_client is None
                or st.session_state.get("last_api_key") != gemini_api_key):
            try:
                client = create_client(gemini_api_key)
                # A real round-trip: validates the key and returns the models
                # it can actually use.
                st.session_state.available_models = list_available_models(client)
                st.session_state.gemini_client = client
                st.session_state["last_api_key"] = normalize_api_key(gemini_api_key)
                st.session_state["api_key_error"] = None
            except ApiKeyError as exc:
                st.session_state.gemini_client = None
                st.session_state.available_models = []
                st.session_state["last_api_key"] = normalize_api_key(gemini_api_key)
                st.session_state["api_key_error"] = str(exc)

        if st.session_state.get("api_key_error"):
            st.error(st.session_state["api_key_error"])
        else:
            st.success(f"Key verified — {len(st.session_state.available_models)} models available")

    # Display important privacy notice regarding API key usage
    st.info("📝 **Note:** Please review the terms of service and privacy policy associated with your Gemini API key.")

    # Model configuration
    st.markdown("---")
    st.markdown("### Model Configuration")

    available_models = st.session_state.get("available_models", [])
    if available_models:
        model_names = [m.name for m in available_models]
        selected_model = st.selectbox(
            "Model",
            options=model_names,
            index=0,
            help=(
                "Populated from your account via the Gemini API, filtered to the models "
                "this app supports. Listed best-first for this workload."
            ),
        )
        model_info = next(m for m in available_models if m.name == selected_model)

        st.caption(
            "On a free-tier Gemini API key, a `flash-lite` model is recommended — "
            "the larger models have much lower free-tier request limits."
        )

        # Remembered per model, so switching models restores the value last
        # used for it. The range comes from what the API reports for that model.
        per_model_temperature = st.session_state.setdefault("model_temperatures", {})
        current = per_model_temperature.get(selected_model, model_info.default_temperature)

        selected_temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=float(model_info.max_temperature),
            value=float(current),
            step=0.05,
            help=(
                f"Set the sampling temperature for {selected_model}. "
                f"This model reports a default of {model_info.default_temperature} and a "
                f"maximum of {model_info.max_temperature}. Lower values are more "
                "deterministic; higher values more varied."
            ),
        )
        per_model_temperature[selected_model] = selected_temperature

        if selected_temperature < 1.0 and selected_model.startswith("gemini-3"):
            st.caption(
                f":orange[Google advises against temperature below 1.0 on Gemini 3.x models, "
                f"where it can cause looping or degraded reasoning. This model's own default "
                f"is {model_info.default_temperature}.]"
            )
    else:
        selected_model = None
        selected_temperature = None
        st.caption("Enter a valid API key to choose a model and temperature.")

    st.session_state["selected_model"] = selected_model
    st.session_state["selected_temperature"] = selected_temperature

    # Visual separator between model configuration and the risk matrix
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
            *Using a custom Risk Matrix requires following the same format as the template.*
            **NOTE:** This app can run **ANY** custom risk matrix, with more or fewer risk levels or categories. Follow the same structure so it parses correctly: \n\n\n
                first column = risk type\n
                second column = risk description\n
                third - ∞ columns = risk levels\n\n
            The shipped template has 9 categories rated Low / Medium / High.\n
            """)
        else:
            st.error("Template file not found. Please contact the administrator.")

        # Then add the file upload section
        st.markdown("### Upload Risk Matrix")

        # File uploader in the sidebar (only include once)
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["xlsx", "xls", "csv"])

        # Show upload status or instructions
        if uploaded_file is not None:
            st.success(f"Uploaded: {safe_label(uploaded_file.name, 200)}")
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
                    st.success(
                        f"PDF {i+1}: {safe_label(uploaded_pdf.name, 200)} uploaded successfully"
                    )

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

    # Data description field
    data_description = st.text_area(
        "Data Description",
        placeholder="Describe the data being used...",
        help="Describe the data being used, including its source, format, and any special considerations.",
        height=100,
        max_chars=1000
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

    # Other considerations field
    other_considerations = st.text_area(
        "Other Considerations",
        placeholder="Add any other relevant information...",
        help="Provide any additional information that might be relevant for the risk assessment.",
        height=100,
        max_chars=1000
    )

    # Run Risk Analysis button at the bottom of the sidebar
    st.markdown("---")
    submit_button = st.button("Run Risk Analysis", type="primary", width="stretch")

# Main content area
# This is where we'll display the results of the risk analysis
if submit_button:
    try:
        # Create a loading placeholder
        loading_placeholder = st.empty()
        loading_placeholder.info("Initializing risk assessment process...")

        # Step 0: no expensive work without a verified API key.
        if not st.session_state.gemini_client:
            loading_placeholder.warning(
                "Enter a valid Gemini API key before running an assessment."
            )
            st.stop()

        # Step 1: Load the risk matrix. Size and shape limits live in the loader.
        try:
            if uploaded_file == "default":
                default_matrix_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), "back_end", "risk_matrix_template.xlsx"
                )
                if not os.path.exists(default_matrix_path):
                    loading_placeholder.error(
                        "Default matrix file not found. Please contact the administrator."
                    )
                    st.stop()
                risk_matrix_df = load_matrix_path(default_matrix_path)
                loading_placeholder.success("Using the default State of Utah Gen-AI Risk Matrix")
            elif uploaded_file is not None:
                if uploaded_file.size > MAX_UPLOAD_BYTES:
                    loading_placeholder.error(
                        f"Risk matrix exceeds the {MAX_UPLOAD_BYTES // 1024 // 1024} MB limit."
                    )
                    st.stop()
                risk_matrix_df = load_matrix(uploaded_file.getvalue(), uploaded_file.name)
            else:
                loading_placeholder.error("Please select a risk matrix option or upload a file.")
                st.stop()
        except MatrixError as exc:
            loading_placeholder.error(f"Risk matrix rejected: {safe_label(exc, 500)}")
            st.stop()

        # One representation per session, rather than several copies.
        st.session_state["risk_matrix_df"] = risk_matrix_df

        risk_levels = parse_risk_levels(risk_matrix_df)
        st.session_state["risk_levels"] = risk_levels

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

            # Assemble the run's inputs for the graph.
            company = project_details.get("company_details", {})
            info = project_details.get("project_information", {})
            assessment_inputs = AssessmentInputs(
                company_name=company.get("company_name", ""),
                technology_name=company.get("technology_name", ""),
                project_details=info.get("project_details", ""),
                data_sensitivity=info.get("data_sensitivity", []) or [],
                data_description=info.get("data_description", ""),
                other_considerations=info.get("other_considerations", ""),
                urls=urls or [],
                pdfs=pdf_files or [],
            )

            # Live progress. ADK separates Event.message (shown to the user)
            # from Event.output (passed downstream), so each node can report
            # itself as the run proceeds.
            st.session_state.current_agent_status = "Starting risk assessment..."
            total_steps = len(urls or []) + len(pdf_files or []) + len(risk_matrix_df)
            progress_bar = st.progress(0.0)
            status_line = st.empty()
            step_counter = {"done": 0}

            def report(message):
                step_counter["done"] += 1
                st.session_state.current_agent_status = message
                # Already sanitised by the graph, at the point each untrusted
                # value was interpolated. Escaping it again would double the
                # backslashes and render them.
                status_line.markdown(f"**{message}**")
                if total_steps:
                    progress_bar.progress(min(step_counter["done"] / total_steps, 1.0))

            risk_results, evidence_warnings, quota_warning, _run = run_assessment(
                api_key=st.session_state.get("last_api_key", ""),
                model_id=st.session_state.get("selected_model"),
                temperature=st.session_state.get("selected_temperature"),
                matrix_df=risk_matrix_df,
                inputs=assessment_inputs,
                client=st.session_state.gemini_client,
                on_progress=report,
            )
            progress_bar.progress(1.0)

            # Store results in session state
            st.session_state.risk_assessment_results = risk_results
            st.session_state.evidence_warnings = evidence_warnings
            st.session_state.quota_warning = quota_warning
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

    # Progress during a run is rendered inline by the processing block above,
    # driven by Event.message from the graph. This branch only guards against a
    # rerun landing while the flag is still set.
    if st.session_state.get("processing", False):
        st.info("Risk assessment in progress...")

    # Show results when complete
    elif st.session_state.get("risk_assessment_complete", False) and st.session_state.risk_assessment_results:
        st.success("⚡️ Risk assessment complete!")

        quota_warning = st.session_state.get("quota_warning")
        if quota_warning:
            st.warning(f"**Free-tier quota:** {quota_warning}")

        evidence_warnings = st.session_state.get("evidence_warnings", [])
        if evidence_warnings:
            with st.container(border=True):
                st.warning(
                    f"{len(evidence_warnings)} source(s) could not be read and contributed no "
                    "evidence to this assessment:"
                )
                for item in evidence_warnings:
                    # source is a URL the user typed or a filename: both are
                    # names they need to read, so escape without stripping.
                    kind = safe_label(item.get("kind", ""), 20)
                    source = safe_label(item.get("source", ""), 300)
                    detail = safe_label(item.get("warning", ""), 1000)
                    st.markdown(f"- {kind}: {source} — {detail}")

        # Add download button for Excel export
        risk_results = st.session_state.risk_assessment_results

        # Create a function to convert risk assessment results to Excel format
# Display each risk category assessment in expandable sections
        st.subheader("Detailed Risk Assessment")

        # Retrieve risk levels for visual color coding of risk severity
        risk_levels = st.session_state.get("risk_levels", [])

        for result in st.session_state.risk_assessment_results:
            # Sanitize before anything reaches a rendering surface: strip
            # Markdown images and links from model text, and constrain the risk
            # level to the matrix's real levels.
            clean = sanitize_result(result, risk_levels)
            category = clean["category"]
            risk_level = clean["risk_level"]
            reasoning = clean["reasoning"]
            mitigations = clean["mitigations"]

            # Apply color coding based on risk level severity.
            # Band by position so the scale reads correctly at any level count:
            # the lowest level is always green and the highest always red. Fixed
            # percentage thresholds assumed five levels and rendered the middle
            # of a three-level matrix blue, reading as "medium-low".
            color = risk_level_color(risk_level, risk_levels)

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
                data=build_report_workbook(risk_results, st.session_state.get("risk_levels", [])),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download the complete risk assessment results as an Excel file",
                type="primary",
                width="stretch"
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
                if st.button("📊 View Risk Assessment Results", type="primary", width="stretch"):
                    st.session_state.risk_analysis_mode_active = True
                    st.rerun()

            with col2:
                # Add the download button
                st.download_button(
                    label="📥 Download as Excel",
                    data=build_report_workbook(risk_results, st.session_state.get("risk_levels", [])),
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Download the complete risk assessment results as an Excel file",
                    type="primary",
                    width="stretch"
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
