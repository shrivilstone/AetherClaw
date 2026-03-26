import os
import warnings
from typing import Optional


class HuggingFaceClient:
    """Lightweight Hugging Face client supporting optional 4-bit loading.

    Usage:
        c = HuggingFaceClient('Tesslate/OmniCoder-9B', quantize=True)
        c.run('Say hi')
    """
    def __init__(self, model: str, quantize: bool = False, hf_token: Optional[str] = None):
        self.model = model
        self.quantize = quantize
        self.hf_token = hf_token or os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_HUB_TOKEN')
        self.local_dir = None
        self.tokenizer = None
        self.model_obj = None
        self.device = 'cpu'

        # Defer heavy imports until needed to avoid import-time side effects
        try:
            from huggingface_hub import snapshot_download
            try:
                self.local_dir = snapshot_download(repo_id=self.model, repo_type='model', use_auth_token=(self.hf_token if self.hf_token else None))
            except TypeError:
                # older hf hub
                self.local_dir = snapshot_download(repo_id=self.model, repo_type='model', token=(self.hf_token if self.hf_token else None))
        except Exception:
            # best-effort; will still attempt transformers direct load
            self.local_dir = None

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except Exception as e:
            raise RuntimeError("Install 'transformers' and 'torch' to use HuggingFaceClient") from e

        # determine device (prefer CUDA, then MPS, then CPU)
        try:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        except Exception:
            self.device = 'cpu'

        # load tokenizer (try fast tokenizer first, then fallback to slow tokenizer)
        try:
            if self.local_dir:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir, use_fast=True, local_files_only=True, trust_remote_code=True)
                except Exception:
                    # some tokenizers require the slow (python) backend or remote code
                    self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir, use_fast=False, local_files_only=True, trust_remote_code=True)
            else:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model, use_fast=True, use_auth_token=(self.hf_token if self.hf_token else None), trust_remote_code=True)
                except Exception:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model, use_fast=False, use_auth_token=(self.hf_token if self.hf_token else None), trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer for {self.model}: {e}") from e

        # load model with optional 4-bit quantization if bitsandbytes available
        try:
            import importlib
            bnb = importlib.import_module('bitsandbytes')
            bnb_available = True
        except Exception:
            bnb_available = False

        load_kwargs = dict()
        try:
            if self.quantize and bnb_available:
                # load-in-4bit using bitsandbytes (if supported by environment)
                self.model_obj = AutoModelForCausalLM.from_pretrained(
                    self.local_dir or self.model,
                    load_in_4bit=True,
                    device_map='auto',
                    trust_remote_code=True,
                    use_auth_token=(self.hf_token if self.hf_token else None),
                )
            else:
                # fallback: low-memory CPU/GPU load
                self.model_obj = AutoModelForCausalLM.from_pretrained(
                    self.local_dir or self.model,
                    device_map='auto',
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    use_auth_token=(self.hf_token if self.hf_token else None),
                )
        except Exception as e:
            warnings.warn(f"Model load failed for {self.model} ({e}), attempting CPU-only load.")
            try:
                import torch
                self.model_obj = AutoModelForCausalLM.from_pretrained(self.local_dir or self.model, torch_dtype=torch.float32, low_cpu_mem_usage=True)
            except Exception as e2:
                raise RuntimeError(f"Failed to load model {self.model}: {e}; {e2}") from e2

        # move to device if appropriate (device_map may already place weights)
        try:
            if hasattr(self.model_obj, 'to') and self.device == 'cpu':
                self.model_obj.to('cpu')
        except Exception:
            pass

    def run(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7, timeout: int = 600) -> str:
        """Generate text for prompt using the loaded HF model.

        This is a best-effort wrapper; large models may be slow or fail on CPU.
        """
        try:
            import torch
        except Exception:
            raise RuntimeError("Torch is required to run the HuggingFaceClient")

        # prepare inputs
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs.get('input_ids')
        attention_mask = inputs.get('attention_mask')

        if input_ids is None:
            raise RuntimeError('Failed to tokenize input')

        # move inputs to device if model expects it
        try:
            device = next(self.model_obj.parameters()).device
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
        except Exception:
            # best-effort; keep on CPU
            pass

        gen_kwargs = dict(max_new_tokens=int(max_tokens), do_sample=(temperature > 0), temperature=float(temperature), top_p=0.95)

        with torch.no_grad():
            outputs = self.model_obj.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text


if __name__ == '__main__':
    # quick smoke: do not auto-download heavy models here
    print('HuggingFaceClient available')
