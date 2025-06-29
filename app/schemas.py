from pydantic import BaseModel, Field
from typing import Optional

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The input text to generate a response for.")
    max_tokens: Optional[int] = Field(300, description="The maximum length of the generated response.")
    temperature: Optional[float] = Field(1.0, description="The sampling temperature for generation.")
    top_p: Optional[float] = Field(1.0, description="The cumulative probability for nucleus sampling.")

class LoadModelRequest(BaseModel):
    model_name: str = Field(..., description="The name of the model to load.")
    device: Optional[str] = Field("cuda", description="The device to load the model on (e.g., 'cpu', 'cuda').")
    hf_token: Optional[str] = Field(None, description="The Hugging Face tokenizer to use, if applicable.")

class ApiKeyResponse(BaseModel):
    api_key: str = Field(..., description="The API key for accessing the model API.")

class LDAPUserRequest(BaseModel):
    username: str = Field(..., description="The username for LDAP authentication.")
    
