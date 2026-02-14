import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
from jinja2 import Template
from src.utils import setup_logging

logger = setup_logging()

class ReportBuilder:
    def __init__(self, template_path: str = "./templates/report.html"):
        self.template_path = template_path
        
        with open(template_path, 'r') as f:
            self.template = Template(f.read())
    
    def generate_report(
        self,
        question: str,
        test_results: List[Dict[str, Any]],
        analysis: Dict[str, Any],
        output_path: str
    ) -> str:
        logger.info(f"Generating report: {output_path}")
        
        chart_data = self._prepare_chart_data(test_results, analysis)
        recommendations = self._generate_recommendations(analysis, test_results)
        
        html_content = self.template.render(
            question=question,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            consistency_score=analysis["consistency_score"],
            paraphrase_count=len(test_results),
            num_clusters=analysis["num_clusters"],
            contradiction_count=len(analysis["contradictions"]),
            contradictions=analysis["contradictions"],
            test_results=test_results,
            chart_data=chart_data,
            recommendations=recommendations
        )
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Report saved: {output_path}")
        return output_path
    
    def _prepare_chart_data(self, test_results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> Dict[str, Any]:
        clusters = analysis["cluster_assignments"]
        unique_clusters = sorted(set(c for c in clusters if c != -1))
        
        cluster_counts = {c: clusters.count(c) for c in unique_clusters}
        
        cluster_labels = [f"Cluster {c}" for c in unique_clusters]
        cluster_sizes = [cluster_counts[c] for c in unique_clusters]
        
        if -1 in clusters:
            cluster_labels.append("Noise")
            cluster_sizes.append(clusters.count(-1))
        
        paraphrase_indices = [f"P{i+1}" for i in range(len(test_results))]
        latencies = [r["latency_ms"] for r in test_results]
        
        return {
            "cluster_labels": cluster_labels,
            "cluster_sizes": cluster_sizes,
            "paraphrase_indices": paraphrase_indices,
            "latencies": latencies,
            "similarity_matrix": analysis["similarity_matrix"]
        }
    
    def _generate_recommendations(self, analysis: Dict[str, Any], test_results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        recommendations = []
        score = analysis["consistency_score"]
        num_clusters = analysis["num_clusters"]
        contradictions = analysis["contradictions"]
        
        if score < 70:
            recommendations.append({
                "title": "Low Consistency Detected",
                "description": "The model produces significantly different responses to the same question. Consider adding explicit instructions to your system prompt to ensure consistent answers."
            })
        
        if num_clusters > 3:
            recommendations.append({
                "title": "Multiple Response Patterns",
                "description": f"Detected {num_clusters} distinct response clusters. Review your prompt to reduce ambiguity and guide the model toward a single, consistent answer pattern."
            })
        
        if contradictions:
            facts_summary = []
            for c in contradictions[:2]:
                facts_summary.append(f"Cluster {c['cluster_a']} vs {c['cluster_b']}: {c['description']}")
            
            recommendations.append({
                "title": "Contradictory Information",
                "description": f"Found contradictions between response groups. Key conflicts: {' | '.join(facts_summary)}. Add fact-checking guidelines to your system prompt."
            })
        
        avg_latency = sum(r["latency_ms"] for r in test_results) / len(test_results)
        if avg_latency > 3000:
            recommendations.append({
                "title": "High Response Latency",
                "description": f"Average response time: {avg_latency:.0f}ms. Consider using shorter prompts or a faster model variant for production."
            })
        
        total_cost = sum(r["cost"] for r in test_results)
        if total_cost > 0.01:
            recommendations.append({
                "title": "Cost Optimization",
                "description": f"Total cost for 20 variations: ${total_cost:.4f}. At scale, consider caching common responses or using a smaller model for simple queries."
            })
        
        if score >= 80 and num_clusters <= 2:
            recommendations.append({
                "title": "Excellent Consistency ✓",
                "description": "Your model maintains high consistency across different phrasings. Current prompt configuration is working well."
            })
        
        return recommendations