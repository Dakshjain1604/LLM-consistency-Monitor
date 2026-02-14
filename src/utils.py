import logging
import sys
import time
from functools import wraps
from typing import Dict, Any

def setup_logging(log_file: str = "consistency_monitor.log") -> logging.Logger:
    logger = logging.getLogger("consistency_monitor")
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def estimate_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
    pricing = {
        "claude": {
            "input": 3.0 / 1_000_000,
            "output": 15.0 / 1_000_000
        },
        "gpt4": {
            "input": 30.0 / 1_000_000,
            "output": 60.0 / 1_000_000
        }
    }
    
    if provider not in pricing:
        return 0.0
    
    cost = (input_tokens * pricing[provider]["input"]) + (output_tokens * pricing[provider]["output"])
    return round(cost, 6)

def parse_token_count(response: Any, provider: str) -> Dict[str, int]:
    try:
        if provider == "claude":
            if hasattr(response, 'usage'):
                return {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
        elif provider == "gpt4":
            if hasattr(response, 'usage'):
                return {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens
                }
    except Exception:
        pass
    
    return {"input_tokens": 0, "output_tokens": 0}

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = (time.time() - start) * 1000
        return result, elapsed
    return wrapper

def async_timer(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        elapsed = (time.time() - start) * 1000
        return result, elapsed
    return wrapper