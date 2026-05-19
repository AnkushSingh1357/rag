import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load from .env directly
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

print(f"Gemini Package Version: {genai.__version__}")

if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env file.")
else:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("Success: Model 'gemini-1.5-flash-latest' is recognized and API key is valid.")
    except Exception as e:
        print(f"API Error: {e}")