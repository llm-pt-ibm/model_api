import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import HTTPException
import gc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None

    def load_model(self, model_name: str, hf_token:str = None, device: str = DEVICE):
        if self.model_name != None and self.model_name != model_name:
            print("Removendo modelo carregado anteriormente...")
            
        self.unload_model()        
        print(f"Carregando modelo {model_name} no dispositivo {device}...")
        try:
            self.model_name = model_name
            if hf_token:           
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
                self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="balanced", token=hf_token)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="balanced")
            self.model.eval()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao carregar modelo: {str(e)}")

    def generate(self, prompt:str, max_tokens:int = 300, temperature:float = 1.0, top_p:float = 1.0):
        
        if self.model is None or self.tokenizer is None:
            raise HTTPException(status_code=400, detail="Nenhum modelo carregado.")

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad(): 
                outputs = self.model.generate(**inputs, max_new_tokens=max_tokens,temperature=temperature, top_p=top_p, eos_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao gerar texto: {str(e)}")
    
    def get_status(self):        
        if self.model is None:
            self.unload_model()
            return "Nenhum modelo carregado."       
        return f"O modelo {self.model_name} está carregado no dispositivo {self.model.device}."

    def unload_model(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None

        gc.collect()
        torch.cuda.empty_cache()

manager = ModelManager()
        

