from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import os
import torch
import logging
import uvicorn
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler()  # Output to console
    ]
)
logger = logging.getLogger("logger")

app = FastAPI()

# Automatically detect if CUDA is available
gpu_available = torch.cuda.is_available()
# Allow override via environment variable
use_gpu = os.environ.get("USE_GPU", "auto") != "0"
# Only use GPU if available AND not explicitly disabled
use_gpu_for_inference = gpu_available and use_gpu

logger.info(f"Using GPU for inference: {use_gpu_for_inference}")

# Model configuration
MODEL_NAME = "microsoft/BioGPT-Large" # decoder style model
MAX_LENGTH = 512

# Initialize vLLM
logger.info(f"Loading model: {MODEL_NAME}")
llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    cpu_only=not use_gpu_for_inference # Auto-detect but allow override from env variable if only cpu is available
    # single gpu inference right now but can implement sharding and tensor parallelism for multi-gpu inference in the LLM() call. vllm already supports this.
)
logger.info("Model loaded successfully")

class ClinicalNoteRequest(BaseModel):
    clinical_note: str
    max_length: int = 256 # Reduced from default 512 since diagnoses lists are usually short (future improvements includes dynamic length)
    # hyperparameters below likely needs to be experimented with and tuned on a small dataset 
    temperature: float = 0.2 
    top_p: float = 0.85 
    top_k: int = 40 
    frequency_penalty: float = 0.3 

class DiagnosesResponse(BaseModel):
    diagnoses: List[str]

def parse_diagnoses(text):
    """Parse a comma-separated list of diagnoses from model output text.
    
    This function handles comma-separated output and cleans up the diagnoses list.
    """
    # Handle empty case
    if not text or text == " ":
        return []
    
    # Split by commas and clean each entry
    diagnoses = [
        d.strip(' "\'[]{}()').strip()  # remove unnecessary characters
        for d in text.split(',')
    ]
    
    # Filter out empty strings and common non-diagnosis text
    filtered_diagnoses = []
    for d in diagnoses:
        if d and not d.lower().startswith(('diagnosis', 'assess', 'the', 'include', 'following')): # some common words that may appear that I have noticed in prior work when working with clinical text processing w/ llms. requires further exploration on a small dataset
            filtered_diagnoses.append(d)
    
    diagnoses = filtered_diagnoses
    
    # If no valid diagnoses found, check if there's a colon with diagnoses after it
    if not diagnoses and ':' in text:
        # This handles cases where model outputs something like "Diagnoses: pneumonia, diabetes"
        after_colon = text.split(':', 1)[1].strip()
        diagnoses = [
            d.strip(' "\'[]{}()').strip() 
            for d in after_colon.split(',')
        ]
        diagnoses = [d for d in diagnoses if d]
    
    # if nothing is found, return an empty list
    if not diagnoses:
        return ["No diagnoses found"]
    
    return diagnoses

@app.post("/generate", response_model=DiagnosesResponse)
async def generate_text(request: ClinicalNoteRequest):
    try:
        # Clearer prompt for comma-separated list only
        # Tuning and experimentation is needed to improve the prompt. Need a strong prompt for a decoder style model since less structured input and output
        prompt = f"""You are a medical expert. Extract all diagnoses from the clinical note below. Carefully read the following clinical note and list ALL possible diagnoses mentioned. The clinical note has all the information you need and it comes from the hospital.

Clinical Note: {request.clinical_note}

Provide your answer as a simple comma-separated list of diagnoses without numbering, explanations, or other text:"""

        # Enhanced sampling parameters
        # Using both (top_k=40, top_p=0.85) means: "First limit to the top 40 tokens, then further limit to those that cover 85% of the probability mass within those 40"
        sampling_params = SamplingParams(
            temperature=request.temperature, # Default 0.2 (makes the model more deterministic)
            top_p=request.top_p, # Default 0.85 (controls diversity)
            top_k=request.top_k, # Default 40 (controls diversity)
            max_tokens=request.max_length, # Default 256 (controls length)
            frequency_penalty=request.frequency_penalty # Default 0.3 (penalizes frequent tokens to avoid repeat diagnoses)
            # can add more vllm sample params to optimize kv caching, do quantization, do streaming, etc.
        )

        # Generate using vLLM
        outputs = llm.generate(prompt, sampling_params)
        generated_text = outputs.outputs[0].text

        # Process the output to extract diagnoses
        diagnoses = parse_diagnoses(generated_text)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_diagnoses = [d for d in diagnoses if not (d.lower() in seen or seen.add(d.lower()))]

        return DiagnosesResponse(diagnoses=unique_diagnoses)

    # some common error handling so you can see why things went wrong with the request. error codes from restapi
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"GPU out of memory: {str(e)}")
        raise HTTPException(status_code=503, detail="Server resource limit exceeded")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# defualt check for FastAPI to make sure it is running fine. asynchronously runs with generate_text
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    p = 8000
    logger.info(f"Starting server on port {p}")
    uvicorn.run(app, host="0.0.0.0", port=p) 