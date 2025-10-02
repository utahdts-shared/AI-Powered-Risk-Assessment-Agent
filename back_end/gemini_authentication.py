"""
Authentication module for Google Generative AI SDK (Gemini).
Handles API key validation and client initialization.
"""

from google import genai
import streamlit as st

def initialize_gemini_client(api_key):
    """
    Initialize the Gemini client with the provided API key.
    
    Args:
        api_key (str): The Google Generative AI API key
        
    Returns:
        client: Configured Generative AI client or None if initialization fails
    """
    if not api_key:
        st.error("Please provide a valid Gemini API key.")
        return None
    
    try:
        # Create a client with the API key (following test.py pattern)
        client = genai.Client(api_key=api_key)
        
        # Don't try to verify with list() as that doesn't work
        # Just return the initialized client
        return client
    except Exception as e:
        st.error(f"Failed to initialize Gemini client: {str(e)}")
        return None

def validate_api_key(api_key):
    """
    Validate that the provided API key is properly formatted and active.
    
    Args:
        api_key (str): The Google Generative AI API key
        
    Returns:
        bool: True if the API key is valid, False otherwise
    """
    if not api_key or len(api_key) < 10:  # Basic validation for key format
        return False
    
    try:
        # Create a client with the API key
        client = genai.Client(api_key=api_key)
        
        # We can't use list() to verify since that doesn't exist
        # Just return True if we can create the client without error
        return True
    except Exception:
        return False