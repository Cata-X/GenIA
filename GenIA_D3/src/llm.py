"""Acceso al modelo. TODOS los agentes usan el mismo: Qwen2.5-1.5B-Instruct (open-weight, <= 8B).

Interfaz que usan los agentes:
    generar(prompt, max_new_tokens, n=1, temperatura=0.0, seed=None) -> list[str]   (n salidas)
    generar_lote(prompts, max_new_tokens)                             -> list[str]   (greedy)

Decodificacion
    - temperatura == 0  -> greedy (do_sample=False), igual que en el Entregable 2.
    - temperatura  > 0  -> muestreo puro: temperature=T, top_p=1.0, top_k=0, con `seed` fija para que
                           la corrida sea reproducible. n muestras en una sola llamada (num_return_sequences).
    - En ambos casos queda activa la repetition_penalty=1.1 que trae generation_config.json de Qwen
      (tampoco se tocaba en el D2).
generar_lote usa padding a la izquierda: en fp16 el resultado puede diferir en algun token del que
daria cada prompt por separado.
"""
import time

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"   # mismo hash fijado en el Entregable 2


class QwenLLM:
    def __init__(self, nombre=MODEL_NAME, revision=MODEL_REVISION):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        assert torch.cuda.is_available(), "No hay GPU: en Colab activa la T4 (Entorno de ejecución > Cambiar tipo)"
        self.torch = torch
        self.nombre, self.revision = nombre, revision
        self.tok = AutoTokenizer.from_pretrained(nombre, revision=revision)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            nombre, revision=revision, torch_dtype=torch.float16, device_map="auto")   # la T4 no soporta bf16
        self.model.eval()
        self.llamadas = 0          # numero de llamadas a generate()
        self.segundos = 0.0

    # -------------------------------------------------------------- utilidades
    def _texto(self, prompt):
        return self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                            tokenize=False, add_generation_prompt=True)

    def _generate(self, textos, max_new_tokens, n, temperatura, seed):
        torch = self.torch
        inputs = self.tok(textos, return_tensors="pt", padding=True).to(self.model.device)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.eos_token_id)
        if temperatura and temperatura > 0:
            kw.update(do_sample=True, temperature=float(temperatura), top_p=1.0, top_k=0, num_return_sequences=n)
            if seed is not None:
                torch.manual_seed(int(seed) % (2**31))
                torch.cuda.manual_seed_all(int(seed) % (2**31))
        else:
            kw.update(do_sample=False)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**inputs, **kw)
        self.llamadas += 1
        self.segundos += time.time() - t0
        largo = inputs.input_ids.shape[1]
        return [self.tok.decode(o[largo:], skip_special_tokens=True) for o in out]

    # -------------------------------------------------------------- interfaz
    def generar(self, prompt, max_new_tokens=60, n=1, temperatura=0.0, seed=None):
        if not temperatura:
            n = 1                                  # greedy: n salidas serian identicas
        return self._generate([self._texto(prompt)], max_new_tokens, n, temperatura, seed)

    def generar_lote(self, prompts, max_new_tokens=60, lote_max=4):
        if len(prompts) == 1:
            return self.generar(prompts[0], max_new_tokens)
        salidas = []
        for i in range(0, len(prompts), lote_max):
            textos = [self._texto(p) for p in prompts[i:i + lote_max]]
            salidas += self._generate(textos, max_new_tokens, 1, 0.0, None)
        return salidas

    def como_funcion_d2(self):
        """Adaptador `generar(prompt, max_new_tokens) -> str` para el codigo del D2 (baseline y agente)."""
        return lambda prompt, max_new_tokens: self.generar(prompt, max_new_tokens)[0]
