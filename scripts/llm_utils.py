import os
import groq
from langchain_groq import ChatGroq

class RotatingChatGroq(ChatGroq):
    """A wrapper for ChatGroq that automatically rotates API keys on 429 RateLimitError."""
    api_keys: list[str] = []
    current_key_idx: int = 0

    def rotate_key(self):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        new_key = self.api_keys[self.current_key_idx]
        print(f"\n[API Key Manager] 🔄 Rate Limit Hit! Auto-switching to Groq API Key #{self.current_key_idx + 1}...\n")
        
        # Override the private groq clients completely
        client_params = {
            "api_key": new_key,
            "base_url": self.groq_api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": {"User-Agent": "langchain/custom-rotator"},
            "default_query": self.default_query,
        }
        self.client = groq.Groq(**client_params).chat.completions
        self.async_client = groq.AsyncGroq(**client_params).chat.completions

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Try all keys before finally failing
        for _ in range(len(self.api_keys)):
            try:
                return super()._generate(messages, stop, run_manager, **kwargs)
            except groq.RateLimitError:
                self.rotate_key()
        # If all keys are exhausted, let it crash or retry outer loop
        return super()._generate(messages, stop, run_manager, **kwargs)

def get_rotating_llm(model_name: str, temperature: float = 0, llm_cls=RotatingChatGroq, **kwargs):
    raw_keys = os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", ""))
    api_key_list = [k.strip() for k in raw_keys.split(",") if k.strip()]
    if not api_key_list:
        raise ValueError("No GROQ_API_KEYS found in environment!")
    
    return llm_cls(
        groq_api_key=api_key_list[0],
        api_keys=api_key_list,
        model_name=model_name,
        temperature=temperature,
        **kwargs
    )
