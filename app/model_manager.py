import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import HTTPException

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None

    def load_model(self, model_name: str, hf_token:str = None, device: str = DEVICE):
        print(f"Loading model {model_name} on device {device}...")
        try:
            self.model_name = model_name
            if hf_token:           
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
                self.model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token).to(device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            self.model.eval()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao carregar modelo: {str(e)}")

    def generate(self, prompt:str, max_tokens:int = 50, temperature:float = 1.0, top_p:float = 1.0):
        
        if self.model is None or self.tokenizer is None:
            raise HTTPException(status_code=400, detail="Nenhum modelo carregado.")

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad(): 
                outputs = self.model.generate(**inputs, max_new_tokens=max_tokens,temperature=temperature, top_p=top_p)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao gerar texto: {str(e)}")
        
manager = ModelManager()
        

