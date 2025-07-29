import time 
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from app import schemas, model_manager, auth
from app.logger import setup_logger

logger = setup_logger("API")

app = FastAPI()

async def require_api_key(request: Request) -> schemas.LDAPUserRequest:
    user = await auth.verify_apikey(request)
    if not user:
        logger.warning("API key inválida ou não fornecida.")
        raise HTTPException(status_code=401, detail="API key invalida.")
    logger.info(f"API key verificada para o usuário: {user}.")
    return user

@app.post("/generate_apikey")
async def generate_apikey(payload: schemas.LDAPUserRequest) -> JSONResponse:
    key = auth.generate_apikey(payload.username)
    logger.info(f"API key gerada para o usuário {payload.username}.")
    return JSONResponse(status_code=200, content={"api_key": key})

@app.post("/load_model", dependencies=[Depends(require_api_key)])
async def load_model(payload: schemas.LoadModelRequest) -> JSONResponse:
    try:
        model_manager.manager.load_model(payload.model_name, payload.hf_token, payload.device)
        return JSONResponse(content={"message": f"Modelo {payload.model_name} carregado com sucesso."})
    except Exception as e:
        logger.error(f"Erro ao carregar o modelo: {payload.model_name}. Erro: {str(e)}")
        raise HTTPException(status_code=500, content={"error": str(e)})
    
@app.post("/generate", dependencies=[Depends(require_api_key)])
async def generate(payload: schemas.GenerateRequest) -> JSONResponse:
    start_time = time.time()
    try:
        completions_data = model_manager.manager.generate(payload)

        request_time = time.time() - start_time
        response_content = {
            "success": True,
            "request_time": request_time,
            "completions": completions_data
        }
        return JSONResponse(status_code=200, content=response_content)

    except Exception as e:
        logger.error(f"Erro ao gerar texto: {str(e)} - Payload: {payload.dict()}", exc_info=True)

        error_response = {
            "success": False,
            "error": str(e)
        }
        return JSONResponse(status_code=500, content=error_response)
    
@app.get("/status", dependencies=[Depends(require_api_key)])
async def status()-> JSONResponse:
    str_status = model_manager.manager.get_status()
    logger.info(f"Status do modelo: {str_status}")
    return JSONResponse(content={"status": str_status})

@app.post("/unload_model", dependencies=[Depends(require_api_key)])
async def unload_model() -> JSONResponse:
    try:
        str_unload = model_manager.manager.unload_model()
        return JSONResponse(content={"message":str_unload})
    except Exception as e:
        raise HTTPException(status_code=500, content={"error": str(e)})
