from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The input text to generate a response for.")
    max_tokens: int = Field(0, description="The maximum length of the generated response.")
    temperature: float = Field(0.7, description="The sampling temperature for generation.")
    top_p: float = Field(0.9, description="The cumulative probability for nucleus sampling.")
    num_return_sequences: int = Field(1, description="The number of sequences to return.")

class LoadModelRequest(BaseModel):
    model_name: str = Field(..., description="The name of the model to load.")
    model_path: str = Field(..., description="The path to the model files.")
    device: str = Field("cuda", description="The device to load the model on (e.g., 'cpu', 'cuda').")

class ApiKeyResponse(BaseModel):
    api_key: str = Field(..., description="The API key for accessing the model API.")
