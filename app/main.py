from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from app import schemas, model_manager, auth

app = FastAPI()

async def require_api_key(request: Request) -> schemas.LDAPUserRequest:
    user = await auth.verify_apikey(request)
    if not user:
        raise HTTPException(status_code=401, detail="API key invalida.")
    return user

@app.post("/generate_apikey")
async def generate_apikey(payload: schemas.LDAPUserRequest) -> JSONResponse:
    key = auth.generate_apikey(payload.username)
    return JSONResponse(status_code=200, content={"api_key": key})

@app.post("/load_model", dependencies=[Depends(require_api_key)])
async def load_model(payload: schemas.LoadModelRequest) -> JSONResponse:
    try:
        model_manager.manager.load_model(payload.model_name, payload.hf_token, payload.device)
        return JSONResponse(content={"message": f"Modelo {payload.model_name} carregado com sucesso."})
    except Exception as e:
        raise HTTPException(status_code=500, content={"error": str(e)})
    
@app.post("/generate", dependencies=[Depends(require_api_key)])
async def generate(payload: schemas.GenerateRequest)-> JSONResponse:
    try:
        result = model_manager.manager.generate(payload.prompt, payload.max_tokens, payload.temperature, payload.top_p)
        return {"result": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.get("/status", dependencies=[Depends(require_api_key)])
async def status()-> JSONResponse:
    str_status = model_manager.manager.get_status()
    return JSONResponse(content={"status": str_status})