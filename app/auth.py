import secrets 
import json
from fastapi import HTTPException, Request

APIKEY_STORE_FILE = "app/apikey_store.json"

def load_apikeys():
    try:
        with open(APIKEY_STORE_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Arquivo de API keys não encontrado: {APIKEY_STORE_FILE}"
        )
    
def save_apikeys(keys):
    with open(APIKEY_STORE_FILE, "w") as f:
        json.dump(keys, f, indent=4)

def generate_apikey(user):
    key = secrets.token_hex(32)
    keys = load_apikeys()
    keys[user] = key
    save_apikeys(keys)
    return key

async def verify_apikey(request: Request):
    apikey = request.headers.get("x-API-Key")
    if not apikey:
        raise HTTPException(
            status_code=401,
            detail="API key não fornecida."
        )
    try:
        keys = load_apikeys()
        return keys.get(apikey)
    
    except json.JSONDecodeError:
        raise HTTPException(
        status_code=403,
        detail="API key inválida."
        )