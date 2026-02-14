import json
import os
from typing import Dict, List
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
import numpy as np
from src.utils import setup_logging

logger = setup_logging()

class ParaphraseGenerator:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        
        self.client = Anthropic(api_key=self.api_key)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def generate_paraphrases(self, original_question: str) -> Dict[str, List[str]]:
        logger.info(f"Generating paraphrases for: {original_question}")
        
        styles = ["formal", "casual", "short", "statement"]
        paraphrases = {}
        
        for style in styles:
            paraphrases[style] = self._generate_style_variations(original_question, style)
        
        all_variations = [original_question]
        for style_list in paraphrases.values():
            all_variations.extend(style_list)
        
        self._validate_uniqueness(all_variations)
        self._validate_semantic_similarity(all_variations)
        
        logger.info(f"Generated {sum(len(v) for v in paraphrases.values())} paraphrases")
        return paraphrases
    
    def _generate_style_variations(self, original: str, style: str) -> List[str]:
        style_descriptions = {
            "formal": "formal, professional, and polite variations",
            "casual": "casual, conversational, and friendly variations",
            "short": "short, concise, and brief variations (preferably under 10 words)",
            "statement": "statement-form variations (as if declaring a need, not asking)"
        }
        
        prompt = f"""Generate 5 {style_descriptions[style]} of this question: {original}

Requirements:
- All variations must ask the SAME thing with different wording
- Each variation must be distinct from the others
- Maintain the core intent and meaning
- Return ONLY a valid JSON array of 5 strings, nothing else

Example format: ["variation 1", "variation 2", "variation 3", "variation 4", "variation 5"]"""
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text.strip()
            
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            variations = json.loads(content)
            
            if not isinstance(variations, list) or len(variations) != 5:
                logger.warning(f"Invalid format for {style}, retrying...")
                return self._generate_style_variations(original, style)
            
            return variations
            
        except Exception as e:
            logger.error(f"Error generating {style} variations: {e}")
            return [f"{original} ({style} variation {i+1})" for i in range(5)]
    
    def _validate_uniqueness(self, variations: List[str]) -> None:
        unique_variations = set(v.lower().strip() for v in variations)
        if len(unique_variations) < len(variations):
            logger.warning("Some duplicate variations detected")
    
    def _validate_semantic_similarity(self, variations: List[str]) -> None:
        embeddings = self.model.encode(variations)
        
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        logger.info(f"Average semantic similarity: {avg_similarity:.3f}")
        
        if avg_similarity < 0.70:
            logger.warning(f"Low semantic similarity detected: {avg_similarity:.3f}")