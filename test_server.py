import pytest
from fastapi.testclient import TestClient
import unittest.mock as mock
import torch
from server import app, parse_diagnoses, llm

## Create a test client using your FastAPI app
client = TestClient(app)

## very basic unit tests for parsing the output of the LLM

def test_parse_diagnoses_empty():
    """Test parsing an empty text."""
    assert parse_diagnoses("") == []
    assert parse_diagnoses("   ") == []

def test_parse_diagnoses_simple_list():
    """Test parsing a simple comma-separated list."""
    text = "pneumonia, hypertension, diabetes"
    expected = ["pneumonia", "hypertension", "diabetes"]
    assert parse_diagnoses(text) == expected

def test_parse_diagnoses_with_formatting():
    """Test parsing a list with various formatting issues."""
    text = " pneumonia,  hypertension , diabetes mellitus, 'asthma', [COPD], (anxiety)"
    expected = ["pneumonia", "hypertension", "diabetes mellitus", "asthma", "COPD", "anxiety"]
    assert parse_diagnoses(text) == expected

def test_parse_diagnoses_with_prefixes():
    """Test that common prefixes are filtered out."""
    text = "diagnosis: pneumonia, the hypertension, assessment: diabetes"
    # "diagnosis: pneumonia" should be filtered due to prefix "diagnosis"
    expected = ["pneumonia", "hypertension", "diabetes"]
    result = parse_diagnoses(text)
    assert all(item in result for item in expected)

def test_parse_diagnoses_no_matches():
    """Test when no diagnoses are found."""
    text = "No significant findings."
    assert parse_diagnoses(text) == ["No diagnoses found"]

## mock the llm

class MockOutput:
    """Mock for vLLM output."""
    def __init__(self, text):
        self.text = text
        
class MockOutputs:
    """Mock for vLLM outputs list."""
    def __init__(self, text):
        self.outputs = [MockOutput(text)]

@pytest.fixture
def mock_llm_generate():
    """Fixture to mock the LLM generate method."""
    with mock.patch.object(llm, 'generate') as mock_generate:
        yield mock_generate

## api endpoint tests

def test_generate_text_endpoint_success(mock_llm_generate):
    """Test successful text generation endpoint."""
    # Mock the LLM response
    mock_llm_generate.return_value = MockOutputs("pneumonia, hypertension, diabetes")
    
    # Create test request
    request_data = {
        "clinical_note": "Patient has pneumonia, hypertension, and diabetes."
    }
    
    # Send request to the endpoint
    response = client.post("/generate", json=request_data)
    
    # Check response
    assert response.status_code == 200
    assert "diagnoses" in response.json()
    assert set(response.json()["diagnoses"]) == {"pneumonia", "hypertension", "diabetes"}
    
    # Verify LLM was called with expected parameters
    mock_llm_generate.assert_called_once()

def test_generate_text_with_parameters(mock_llm_generate):
    """Test that custom parameters are used."""
    mock_llm_generate.return_value = MockOutputs("pneumonia")
    
    request_data = {
        "clinical_note": "Patient has pneumonia.",
        "temperature": 0.1,
        "top_p": 0.5,
        "top_k": 20,
        "max_length": 100,
        "frequency_penalty": 0.5
    }
    
    response = client.post("/generate", json=request_data)
    assert response.status_code == 200
    
    # Check that parameters were passed to the LLM
    args, kwargs = mock_llm_generate.call_args
    sampling_params = kwargs.get('sampling_params') or args[1]
    
    assert sampling_params.temperature == 0.1
    assert sampling_params.top_p == 0.5
    assert sampling_params.top_k == 20
    assert sampling_params.max_tokens == 100
    assert sampling_params.frequency_penalty == 0.5

@pytest.mark.parametrize(
    "error,status_code",
    [
        (ValueError("Invalid input"), 400),
        (torch.cuda.OutOfMemoryError("Out of memory"), 503),
        (Exception("General error"), 500)
    ]
)
def test_generate_text_endpoint_error_handling(mock_llm_generate, error, status_code):
    """Test error handling in the endpoint."""
    # Mock LLM to raise an error
    mock_llm_generate.side_effect = error
    
    request_data = {
        "clinical_note": "Test note"
    }
    
    response = client.post("/generate", json=request_data)
    assert response.status_code == status_code

## run the tests
if __name__ == "__main__":
    pytest.main(["-xvs", "test_server.py"]) 