import json
import os
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from anthropic import Anthropic
from src.utils import setup_logging

logger = setup_logging()

class ConsistencyAnalyzer:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        
        self.client = Anthropic(api_key=self.api_key)
        logger.info("Loading sentence-transformers model: all-MiniLM-L6-v2")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def analyze(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info(f"Analyzing consistency for {len(test_results)} responses")
        
        responses = [r["response"] for r in test_results]
        
        embeddings = self.model.encode(responses)
        
        similarity_matrix = self._compute_similarity_matrix(embeddings)
        
        clusters = self._cluster_responses(embeddings)
        
        facts_by_cluster = self._extract_facts_by_cluster(responses, clusters)
        
        contradictions = self._identify_contradictions(facts_by_cluster, clusters)
        
        consistency_score = self._calculate_consistency_score(similarity_matrix)
        
        num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        
        return {
            "consistency_score": consistency_score,
            "num_clusters": num_clusters,
            "cluster_assignments": clusters.tolist(),
            "similarity_matrix": similarity_matrix.tolist(),
            "facts_by_cluster": facts_by_cluster,
            "contradictions": contradictions,
            "embeddings": embeddings.tolist()
        }
    
    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarity_matrix[i][j] = sim
        
        return similarity_matrix
    
    def _cluster_responses(self, embeddings: np.ndarray) -> np.ndarray:
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        clusters = clustering.fit_predict(embeddings)
        
        logger.info(f"DBSCAN clustering: {len(set(clusters))} clusters found")
        return clusters
    
    def _extract_facts_by_cluster(self, responses: List[str], clusters: np.ndarray) -> Dict[str, List[str]]:
        unique_clusters = set(clusters)
        facts_by_cluster = {}
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue
            
            cluster_responses = [responses[i] for i, c in enumerate(clusters) if c == cluster_id]
            
            facts = self._extract_facts_from_responses(cluster_responses)
            facts_by_cluster[str(cluster_id)] = facts
        
        return facts_by_cluster
    
    def _extract_facts_from_responses(self, responses: List[str]) -> List[str]:
        combined_responses = "\n\n---\n\n".join(responses[:5])
        
        prompt = f"""Analyze these similar responses and extract 3-5 key facts or conclusions they share:

{combined_responses}

Return ONLY a JSON array of 3-5 concise fact strings, nothing else.
Example: ["Fact 1", "Fact 2", "Fact 3"]"""
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
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
            
            facts = json.loads(content)
            
            if isinstance(facts, list):
                return facts[:5]
            
        except Exception as e:
            logger.error(f"Error extracting facts: {e}")
        
        return [f"Response pattern {i+1}" for i in range(3)]
    
    def _identify_contradictions(self, facts_by_cluster: Dict[str, List[str]], clusters: np.ndarray) -> List[Dict[str, Any]]:
        contradictions = []
        cluster_ids = list(facts_by_cluster.keys())
        
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                cluster_a = cluster_ids[i]
                cluster_b = cluster_ids[j]
                
                facts_a = facts_by_cluster[cluster_a]
                facts_b = facts_by_cluster[cluster_b]
                
                contradiction = self._compare_fact_sets(facts_a, facts_b, cluster_a, cluster_b)
                
                if contradiction:
                    size_a = sum(1 for c in clusters if c == int(cluster_a))
                    size_b = sum(1 for c in clusters if c == int(cluster_b))
                    
                    contradictions.append({
                        "cluster_a": int(cluster_a),
                        "cluster_b": int(cluster_b),
                        "description": contradiction,
                        "size_a": size_a,
                        "size_b": size_b,
                        "percentage_a": round(size_a / len(clusters) * 100, 1),
                        "percentage_b": round(size_b / len(clusters) * 100, 1)
                    })
        
        return contradictions
    
    def _compare_fact_sets(self, facts_a: List[str], facts_b: List[str], cluster_a: str, cluster_b: str) -> str:
        prompt = f"""Compare these two sets of facts and identify if they contradict each other:

Cluster {cluster_a} facts:
{chr(10).join(f"- {fact}" for fact in facts_a)}

Cluster {cluster_b} facts:
{chr(10).join(f"- {fact}" for fact in facts_b)}

If they contradict, return a brief description of the contradiction (one sentence).
If they don't contradict, return an empty string.
Return ONLY the contradiction description or empty string, nothing else."""
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}]
            )
            
            contradiction = response.content[0].text.strip()
            
            if len(contradiction) > 10 and not contradiction.lower().startswith("no contradiction"):
                return contradiction
            
        except Exception as e:
            logger.error(f"Error comparing facts: {e}")
        
        return ""
    
    def _calculate_consistency_score(self, similarity_matrix: np.ndarray) -> int:
        n = len(similarity_matrix)
        if n <= 1:
            return 100
        
        upper_triangle = []
        for i in range(n):
            for j in range(i + 1, n):
                upper_triangle.append(similarity_matrix[i][j])
        
        avg_similarity = np.mean(upper_triangle)
        
        score = int(round(avg_similarity * 100))
        return max(0, min(100, score))