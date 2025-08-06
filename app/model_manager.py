import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from fastapi import HTTPException
import gc
from typing import List
from .utils import is_model_on_gpu
from app.logger import setup_logger
from app import schemas

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, stop_sequence: List[int]):
        super().__init__()
        self.stop_sequence = torch.tensor(stop_sequence, dtype=torch.long)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_sequence = self.stop_sequence.to(input_ids.device)
        
        if len(input_ids[0]) >= len(stop_sequence):
            if torch.equal(input_ids[0][-len(stop_sequence):], stop_sequence):
                return True
        return False

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.logger = setup_logger("ModelManager")

    def load_model(self, model_name: str, hf_token: str = None, device: str = DEVICE):
        if self.model and self.model_name == model_name:
            self.logger.info(f"O modelo {model_name} já está carregado.")
            return

        if self.model:
            self.logger.info(f"Descarregando modelo {self.model_name} antes de carregar {model_name}.")
            self.unload_model()

        self.logger.info(f"Carregando modelo {model_name} no dispositivo {device}...")
        try:
            tokenizer_args = {"token": hf_token} if hf_token else {}
            model_args = {"token": hf_token, "device_map": "auto"} 
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_args)
            self.model.eval()
            self.model_name = model_name
            self.logger.info(f"Modelo {model_name} carregado com sucesso.")

        except Exception as e:
            self.logger.error(f"Erro ao carregar o modelo {model_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Erro ao carregar modelo: {str(e)}")

    def generate(self, payload: schemas.GenerateRequest) -> list:

        params = payload.parameters
        prompt = payload.prompt
        
        if self.model_name != payload.model_name:
            self.load_model(payload.model_name, payload.hf_token)

        if not self.model or not self.tokenizer:
            raise HTTPException(status_code=400, detail="Nenhum modelo carregado.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[-1]
        
        stopping_criteria_list = StoppingCriteriaList()
        if params.stop_sequences:
            self.logger.info(f"Aplicando sequências de parada: {params.stop_sequences}")
            stop_sequence_ids = [self.tokenizer.encode(seq, add_special_tokens=False) for seq in params.stop_sequences]
            for stop_ids in stop_sequence_ids:
                stopping_criteria_list.append(StopAtSpecificTokenCriteria(stop_sequence=stop_ids))

        try:
            with torch.no_grad():
                gen_out = self.model.generate(
                    **inputs,
                    max_new_tokens=params.max_new_tokens,
                    temperature=params.temperature,
                    top_p=params.top_p,
                    num_return_sequences=params.num_return_sequences,
                    stopping_criteria=stopping_criteria_list,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro durante a geração do modelo: {e}")

        completions = []
        for i in range(params.num_return_sequences):
            sequence = gen_out.sequences[i]
            gen_token_ids = sequence[prompt_len:]
            
            text = self.tokenizer.decode(gen_token_ids, skip_special_tokens=True)
            
            tokens = []
            log_probs = F.log_softmax(torch.stack(gen_out.scores, dim=1), dim=-1)[i]
            
            for j, token_id in enumerate(gen_token_ids):
                token_text = self.tokenizer.decode(token_id)
                log_prob = log_probs[j, token_id].item()
                tokens.append({"text": token_text, "logprob": log_prob})
                
            completions.append({"text": text, "tokens": tokens})
            
        return completions

    def get_status(self) -> str:
        if not self.model:
            return "Nenhum modelo carregado."
        return is_model_on_gpu(self.model.hf_device_map, self.model_name)

    def unload_model(self):
        if not self.model:
            self.logger.info("Nenhum modelo carregado para descarregar.")
            return "Nenhum modelo carregado para descarregar."
        
        old_model = self.model_name
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self.model_name = None
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.info(f"Modelo {old_model} descarregado com sucesso.")
        return f"Modelo {old_model} descarregado com sucesso."

manager = ModelManager()