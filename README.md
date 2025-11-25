# LLM Inference API (FastAPI + Power9 Architecture)

This project provides a lightweight and efficient **Inference API** for serving **Large Language Models (LLMs)** using **FastAPI**.  
It supports dynamic model loading, text generation, GPU usage detection, API key authentication, and structured logging.

The system was designed for high-performance inference on **Power9-based environments**, but also works on standard x86 machines.

---

## 🚀 Features

- 🔑 **API Key authentication**
- ⚡ **Load and unload Hugging Face models dynamically**
- ✨ **Text generation with configurable parameters**
- 🧩 **Custom stop sequences**
- 📊 **Token-level log-probabilities**
- 🖥️ **Automatic GPU/CPU selection**
- 📝 **Rotating log files**
- 🛡️ **Secure local API key storage (`apikey_store.json`)**

---

## 📦 Requirements

- Python 3.9+
- PyTorch
- FastAPI
- Transformers (Hugging Face)
- Uvicorn

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the API

Start the FastAPI server using Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

You can access the interactive API documentation at:

- Swagger UI → http://localhost:8000/docs
- ReDoc → http://localhost:8000/redoc


## 🔐 Authentication

All protected endpoints require an **API key** to be sent in the request header.

* **Header Name:** `x-API-Key`
* **Header Value:** `<your_api_key>`

### Generate API Key

To get a new API key, use the following endpoint:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/generate_apikey` | Generates a new API key for a specified user. |

**Request Body Example:**

```json
{
  "username": "myuser"
}
```

**Example Response:**

```JSON
{
  "api_key": "ab12cd34ef56..."
}
```

Note: Generated keys are securely stored in app/apikey_store.json.


## 📚 Endpoints Overview
POST /load_model

Loads a Hugging Face model into memory.

Request:

```JSON
{
  "model_name": "meta-llama/Llama-3-8b",
  "device": "cuda",
  "hf_token": null
}
```
POST /generate

Generates text from the loaded model.

Request:
```JSON
{
  "model_name": "meta-llama/Llama-3-8b",
  "prompt": "Explain quantum computing.",
  "parameters": {
    "temperature": 0.7,
    "max_new_tokens": 200,
    "top_p": 0.9,
    "num_return_sequences": 1,
    "stop_sequences": []
  }
}
```

Response example:

```JSON
{
  "success": true,
  "request_time": 0.58,
  "completions": [
    {
      "text": "Quantum computing works by...",
      "tokens": [
        {"text": "Quantum", "logprob": -1.23},
        ...
      ]
    }
  ]
}
```
GET /status

Returns model load state and device information.

Example response:
```JSON
{
  "status": "cuda: model loaded on GPU"
}
```
POST /unload_model

Unloads the current model from memory.

Response:
```JSON
{
  "message": "Modelo meta-llama/Llama-3-8b descarregado com sucesso."
}
```
📁 Project Structure
```
app/
├── __init__.py
├── auth.py              # API key generation & validation
├── logger.py            # Logging setup with rotating handlers
├── main.py              # FastAPI application and endpoints
├── model_manager.py     # Model loading, unloading, and text generation
├── schemas.py           # Pydantic request/response models
├── utils.py             # Utility functions (GPU checks, etc.)
├── .gitignore
└── requirements.txt     # Python dependencies
```
