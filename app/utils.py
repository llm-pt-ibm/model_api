def is_model_on_gpu(hf_device_map: dict, model_name: str) -> str:
    if '' in hf_device_map.keys() and hf_device_map[''] == 'cpu':
        return f"Modelo {model_name} carregado totalmente na CPU."
    elif 'cpu' in hf_device_map.values():
        return f"Algumas camadas do modelo {model_name} estão carregadas na CPU."
    else:
        return f"Modelo {model_name} carregado totalmente na GPU."
