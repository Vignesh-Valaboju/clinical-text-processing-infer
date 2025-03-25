import requests
import json
import argparse
import sys

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Clinical Text Analysis Tool")
    parser.add_argument("--temperature", type=float, help="Temperature for text generation (default: 0.2)")
    parser.add_argument("--top_p", type=float, help="Top-p sampling parameter (default: 0.85)")
    parser.add_argument("--top_k", type=int, help="Top-k sampling parameter (default: 40)")
    parser.add_argument("--max_length", type=int, help="Maximum tokens to generate (default: 256)")
    parser.add_argument("--frequency_penalty", type=float, help="Frequency penalty parameter (default: 0.3)")
    args = parser.parse_args()

    # Clinical note (keeping this consistent with your original script)
    clinical_note = {
        "clinical_note": "Patient is a 67-year-old male presenting to the emergency department with a three-day history of productive cough, fever of 38.7°C, and difficulty breathing. He has a past medical history significant for type 2 diabetes mellitus and hypertension. Chest X-ray reveals right lower lobe infiltrate suggestive of pneumonia. He is started on IV antibiotics and supplemental oxygen."
    }

    # Add optional parameters if provided
    if args.temperature is not None:
        clinical_note["temperature"] = args.temperature
    if args.top_p is not None:
        clinical_note["top_p"] = args.top_p
    if args.top_k is not None:
        clinical_note["top_k"] = args.top_k
    if args.max_length is not None:
        clinical_note["max_length"] = args.max_length
    if args.frequency_penalty is not None:
        clinical_note["frequency_penalty"] = args.frequency_penalty

    try:
        # Send request to API
        response = requests.post(
            "http://localhost:8000/generate",
            json=clinical_note
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        # Print formatted results
        result = response.json()
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
