from pydantic import BaseModel, Field
from typing import Optional, List

class GenerationParameters(BaseModel):
    temperature: float = Field(1.0, description="The sampling temperature for generation.", ge=0)
    max_new_tokens: int = Field(300, description="The maximum number of new tokens to generate.", gt=0)
    top_p: float = Field(1.0, description="The cumulative probability for nucleus sampling.", ge=0, le=1)
    num_return_sequences: int = Field(1, description="The number of completions to generate.", ge=1)
    stop_sequences: List[str] = Field([], description="A list of strings to stop generation at.")

class GenerateRequest(BaseModel):
    model_name: str = Field(..., description="The name of the model to use for generation.")
    prompt: str = Field(..., description="The input text to generate a response for.")
    hf_token: Optional[str] = Field(None, description="The Hugging Face token, if required by the model.")
    parameters: GenerationParameters = Field(..., description="The parameters for text generation.")

class LoadModelRequest(BaseModel):
    model_name: str = Field(..., description="The name of the model to load.")
    device: Optional[str] = Field("cuda", description="The device to load the model on (e.g., 'cpu', 'cuda').")
    hf_token: Optional[str] = Field(None, description="The Hugging Face tokenizer to use, if applicable.")

class ApiKeyResponse(BaseModel):
    api_key: str = Field(..., description="The API key for accessing the model API.")

class LDAPUserRequest(BaseModel):
    username: str = Field(..., description="The username for LDAP authentication.")
    
