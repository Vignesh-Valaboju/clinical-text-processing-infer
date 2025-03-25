# Clinical Text Processing Inference with vLLM

## Setup and Installation
### Docker setup

1. Clone the repo:
   ```bash
   git clone https://github.com/Vignesh-Valaboju/clinical-text-processing-infer.git
   cd clinical-text-processing-infer
   ```

2. Build Docker image:
   ```bash
   # For CPU inference
   docker-compose up --build
   
   # For GPU inference
   USE_GPU=1 docker-compose up --build
   ```

## Docker Compose Configuration

The included `docker-compose.yml` provides:

- Automatic GPU detection and utilization
- Volume mounting for model caching
- Memory limits and resource management
- Health checks

## Usage

### API Endpoints

Extracts diagnoses from a clinical note.

Request body:
```json
{
  "clinical_note": "Patient is a 67-year-old male with a history of hypertension...",
}
```

Response:
```json
{
  "diagnoses": [
    "hypertension",
    "type 2 diabetes mellitus",
    "pneumonia",
  ]
}
```

### Command Line Interface to run inference

```bash
# run with fastapi by using infer.py file
python infer.py

```
or

```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "clinical_note": "Patient is a 67-year-old male presenting to the emergency department with a three-day history of productive cough, fever of 38.7°C, and difficulty breathing. He has a past medical history significant for type 2 diabetes mellitus and hypertension. Chest X-ray reveals right lower lobe infiltrate suggestive of pneumonia. He is started on IV antibiotics and supplemental oxygen.",
         }'
```

## Unit Testing

Run the very basic test suite with:

```bash
# Run specific test file
pytest test_server.py

```

## Configuration Options

### Environment Variables
- `USE_GPU`: Set to "1" to enable GPU inference, "0" for CPU-only (default: "0")

## Modelling Choice 

There are several modeling approaches that we can take to best create a structured output of diagnoses. A decoder style model works well here because you can prompt different structured input and outputs. It allows flexibility, and its generative capabilities allow us to better capture the long tail of diagnoses. An encoder style model would be great at extracting relevant features but it would have a fixed set of classes and lack flexibility to ouput a structured diagnosis list. vLLM allows for efficient serving so I did not worry too much about the computational cost of using a generative model. BioGPT from microsoft is finetuned on pubmed abstracts so it should be more context aware than other general purpose LLMs. There are several medically aware decoder and encoder-decoder styles models that we can choose from on Huggingface, and further experimentation is needed to find the best one.

## Inference Serving and Design

I chose to use vLLM as my inference server since it allows for easy inference scaling. Many technqiues like KV caching, memory efficiency, and parallelism are easy to implement since they are just hyperparameters. It allows for easy integration with FastAPI and quick setup for HTTP Endpoint. We use uvicorn as the asynchronous server gateway interface. 

Main desing components
1. ClinicalNoteRequest class to handle the input json. Using pydantic for strong typing
2. DiagnosisResponse class is used to handle the structured list output. We use parse_diagnoses() function to clean up the model output into the right format. I handled some edge cases in the code for wary outputs, but further testing is needed to best handle all LLM output edge cases. 
3. used async processing so that no requests are blocked by each other. the api health check is also async and helps monitor the service. Async will be even more needed if we implemented batched processing or streaming
4. I did some basic error handling for things like bad api request or OOM or invalid inputs
5. added logging steps where needed

### Current Inference Parameters

I estimated what could be good values for these parameters and left comments in the code, but extensive testing on some datasets will be needed to finetune these paramters. 
- `temperature`: Controls randomness (0.0-1.0, lower = more deterministic)
- `top_p`: Controls diversity (0.0-1.0)
- `top_k`: Limits token selection to top k tokens
- `max_length`: Maximum tokens to generate
- `frequency_penalty`: Penalizes token repetition (0.0-2.0)

## Optimizations that I would make with more time
Below are some optimizations that could be useful:
- tuning block_size parameter in SamplingParams for Paged Attention so we have better KV caching handling 
- implementing batch processing if we are running this model at known intervals
- setting the streaming parameter in SamplingParams if we are getting continuous requests to the model 
- use model sharding param and set up tensor parallelism 
- model quantization 
- Testing an encoder-decoder style model (like T5 model family) since they may have better ability to retrieve useful clinical context but still have genrative output
- have dynamic max lengths for the model depending on how long the clinical notes are. I'd expect shorter notes to have fewer diagnoses.
