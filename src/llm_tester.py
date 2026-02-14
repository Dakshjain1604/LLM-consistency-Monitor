import asyncio
import time
import os
from typing import List, Dict, Any
import aiohttp
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from huggingface_hub import AsyncInferenceClient
from src.utils import setup_logging, estimate_cost, parse_token_count

logger = setup_logging()

class BaseLLMTester:
    async def test_paraphrases(self, paraphrases: List[str]) -> List[Dict[str, Any]]:
        logger.info(f"Testing {len(paraphrases)} paraphrases on {self.__class__.__name__}")
        
        tasks = [self._test_single_paraphrase(p, idx) for idx, p in enumerate(paraphrases)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Paraphrase {i} failed: {result}")
                valid_results.append({
                    "paraphrase": paraphrases[i],
                    "response": f"ERROR: {str(result)}",
                    "latency_ms": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0
                })
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _test_single_paraphrase(self, paraphrase: str, idx: int) -> Dict[str, Any]:
        raise NotImplementedError

class ClaudeTester(BaseLLMTester):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        self.client = AsyncAnthropic(api_key=self.api_key)
    
    async def _test_single_paraphrase(self, paraphrase: str, idx: int) -> Dict[str, Any]:
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                response = await self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": paraphrase}]
                )
                
                latency_ms = (time.time() - start_time) * 1000
                response_text = response.content[0].text
                tokens = parse_token_count(response, "claude")
                cost = estimate_cost("claude", tokens["input_tokens"], tokens["output_tokens"])
                
                return {
                    "paraphrase": paraphrase,
                    "response": response_text,
                    "latency_ms": round(latency_ms, 2),
                    "input_tokens": tokens["input_tokens"],
                    "output_tokens": tokens["output_tokens"],
                    "cost": cost
                }
            
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Retry {attempt + 1}/{max_retries} for paraphrase {idx} after {delay}s")
                    await asyncio.sleep(delay)
                else:
                    raise e
        
        raise Exception(f"Failed after {max_retries} retries")

class GPT4Tester(BaseLLMTester):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found")
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    async def _test_single_paraphrase(self, paraphrase: str, idx: int) -> Dict[str, Any]:
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                response = await self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": paraphrase}],
                    max_tokens=1024
                )
                
                latency_ms = (time.time() - start_time) * 1000
                response_text = response.choices[0].message.content
                tokens = parse_token_count(response, "gpt4")
                cost = estimate_cost("gpt4", tokens["input_tokens"], tokens["output_tokens"])
                
                return {
                    "paraphrase": paraphrase,
                    "response": response_text,
                    "latency_ms": round(latency_ms, 2),
                    "input_tokens": tokens["input_tokens"],
                    "output_tokens": tokens["output_tokens"],
                    "cost": cost
                }
            
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Retry {attempt + 1}/{max_retries} for paraphrase {idx} after {delay}s")
                    await asyncio.sleep(delay)
                else:
                    raise e
        
        raise Exception(f"Failed after {max_retries} retries")

class SafeHuggingFaceTester(BaseLLMTester):
    def __init__(self, model: str = "meta-llama/Llama-3.3-70B-Instruct", api_key: str = None):
        self.model = model
        token_path = "/root/.huggingface/token"
        
        if api_key:
            self.token = api_key
        elif os.path.exists(token_path):
            with open(token_path, 'r') as f:
                self.token = f.read().strip()
        else:
            self.token = os.getenv("HUGGINGFACE_TOKEN")
        
        if not self.token:
            raise ValueError("HuggingFace token not found")
        
        self.client = AsyncInferenceClient(model=self.model, token=self.token)
    
    async def _test_single_paraphrase(self, paraphrase: str, idx: int) -> Dict[str, Any]:
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                response = await self.client.text_generation(
                    prompt=paraphrase,
                    max_new_tokens=1024,
                    return_full_text=False
                )
                
                latency_ms = (time.time() - start_time) * 1000
                response_text = response if isinstance(response, str) else response.generated_text
                
                input_tokens = len(paraphrase.split()) * 1.3
                output_tokens = len(response_text.split()) * 1.3
                
                return {
                    "paraphrase": paraphrase,
                    "response": response_text,
                    "latency_ms": round(latency_ms, 2),
                    "input_tokens": int(input_tokens),
                    "output_tokens": int(output_tokens),
                    "cost": 0.0
                }
            
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Retry {attempt + 1}/{max_retries} for paraphrase {idx} after {delay}s")
                    await asyncio.sleep(delay)
                else:
                    raise e
        
        raise Exception(f"Failed after {max_retries} retries")

class GenericOpenAITester(BaseLLMTester):
    def __init__(self, base_url: str, model: str = "gpt-3.5-turbo", api_key: str = "not-needed"):
        self.base_url = base_url
        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    
    async def _test_single_paraphrase(self, paraphrase: str, idx: int) -> Dict[str, Any]:
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": paraphrase}],
                    max_tokens=1024
                )
                
                latency_ms = (time.time() - start_time) * 1000
                response_text = response.choices[0].message.content
                
                input_tokens = getattr(response.usage, 'prompt_tokens', 0) if hasattr(response, 'usage') else 0
                output_tokens = getattr(response.usage, 'completion_tokens', 0) if hasattr(response, 'usage') else 0
                
                return {
                    "paraphrase": paraphrase,
                    "response": response_text,
                    "latency_ms": round(latency_ms, 2),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": 0.0
                }
            
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Retry {attempt + 1}/{max_retries} for paraphrase {idx} after {delay}s")
                    await asyncio.sleep(delay)
                else:
                    raise e
        
        raise Exception(f"Failed after {max_retries} retries")

class CustomHTTPTester(BaseLLMTester):
    def __init__(self, endpoint: str, api_key: str = None):
        if not endpoint:
            raise ValueError("Custom endpoint URL required")
        self.endpoint = endpoint
        self.api_key = api_key
    
    async def _test_single_paraphrase(self, paraphrase: str, idx: int) -> Dict[str, Any]:
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                async with aiohttp.ClientSession() as session:
                    headers = {}
                    if self.api_key:
                        headers["Authorization"] = f"Bearer {self.api_key}"
                    
                    async with session.post(
                        self.endpoint,
                        json={"prompt": paraphrase},
                        headers=headers
                    ) as resp:
                        latency_ms = (time.time() - start_time) * 1000
                        data = await resp.json()
                        
                        response_text = data.get("response", data.get("text", str(data)))
                        
                        return {
                            "paraphrase": paraphrase,
                            "response": response_text,
                            "latency_ms": round(latency_ms, 2),
                            "input_tokens": data.get("input_tokens", 0),
                            "output_tokens": data.get("output_tokens", 0),
                            "cost": 0.0
                        }
            
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Retry {attempt + 1}/{max_retries} for paraphrase {idx} after {delay}s")
                    await asyncio.sleep(delay)
                else:
                    raise e
        
        raise Exception(f"Failed after {max_retries} retries")

def create_llm_tester(provider: str, **kwargs) -> BaseLLMTester:
    if provider == "claude":
        return ClaudeTester(api_key=kwargs.get("api_key"))
    elif provider == "gpt4":
        return GPT4Tester(api_key=kwargs.get("api_key"))
    elif provider == "huggingface":
        return SafeHuggingFaceTester(
            model=kwargs.get("model", "meta-llama/Llama-3.3-70B-Instruct"),
            api_key=kwargs.get("api_key")
        )
    elif provider == "local":
        return GenericOpenAITester(
            base_url=kwargs.get("base_url"),
            model=kwargs.get("model", "gpt-3.5-turbo"),
            api_key=kwargs.get("api_key", "not-needed")
        )
    elif provider == "custom":
        return CustomHTTPTester(
            endpoint=kwargs.get("endpoint"),
            api_key=kwargs.get("api_key")
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

class LLMTester:
    def __init__(self, model_type: str, api_key: str = None, custom_endpoint: str = None):
        self.tester = create_llm_tester(
            provider=model_type,
            api_key=api_key,
            endpoint=custom_endpoint
        )
    
    async def test_paraphrases(self, paraphrases: List[str]) -> List[Dict[str, Any]]:
        return await self.tester.test_paraphrases(paraphrases)