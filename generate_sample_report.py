import sys
import os
sys.path.insert(0, '/root/consistencyMonitor')

from datetime import datetime
from src.report_builder import ReportBuilder
import numpy as np

print("Generating sample HTML report with mock data...")

question = "How do I reset my password?"

test_results = []
for i in range(20):
    paraphrases = [
        f"What is the procedure for resetting one's password? (variation {i})",
        f"How can I reset my login credentials? (variation {i})",
        f"Password reset process? (variation {i})",
        f"I need to reset my password (variation {i})"
    ]
    
    responses = [
        "To reset your password, click on 'Forgot Password' on the login page, enter your email, and follow the instructions sent to your inbox.",
        "You can reset your password by visiting the Settings page, selecting Security, and clicking 'Change Password'.",
        "To reset your password, navigate to your account settings and select the password reset option.",
    ]
    
    test_results.append({
        "paraphrase": paraphrases[i % 4] if i < 12 else f"Generic variation {i}",
        "response": responses[i % 3] if i < 15 else responses[0],
        "latency_ms": 1200 + (i * 50) + np.random.randint(-100, 100),
        "input_tokens": 15 + i,
        "output_tokens": 45 + (i * 2),
        "cost": 0.0001 * (1 + i * 0.1)
    })

similarity_matrix = []
for i in range(20):
    row = []
    for j in range(20):
        if i == j:
            sim = 1.0
        elif abs(i - j) <= 5:
            sim = 0.85 + np.random.random() * 0.1
        else:
            sim = 0.65 + np.random.random() * 0.15
        row.append(sim)
    similarity_matrix.append(row)

cluster_assignments = [0] * 8 + [1] * 7 + [2] * 5

analysis = {
    "consistency_score": 78,
    "num_clusters": 3,
    "cluster_assignments": cluster_assignments,
    "similarity_matrix": similarity_matrix,
    "facts_by_cluster": {
        "0": [
            "Self-service password reset via 'Forgot Password' link",
            "Email-based verification process",
            "Instructions sent to registered inbox"
        ],
        "1": [
            "Manual reset through Settings > Security",
            "Requires current login access",
            "Direct password change option"
        ],
        "2": [
            "Account settings navigation required",
            "Password reset option in profile",
            "Generic security settings approach"
        ]
    },
    "contradictions": [
        {
            "cluster_a": 0,
            "cluster_b": 1,
            "description": "Cluster 0 suggests email-based reset for forgotten passwords, while Cluster 1 requires existing login access",
            "size_a": 8,
            "size_b": 7,
            "percentage_a": 40.0,
            "percentage_b": 35.0
        }
    ],
    "embeddings": [[0.1] * 384 for _ in range(20)]
}

output_path = "./results/sample_report_demo.html"
os.makedirs("./results", exist_ok=True)

builder = ReportBuilder()
report_path = builder.generate_report(question, test_results, analysis, output_path)

print(f"\n✓ Sample report generated: {report_path}")
print(f"✓ File size: {os.path.getsize(report_path)} bytes")
print("\nReport includes:")
print("  ✓ Summary card with consistency score (78%)")
print("  ✓ Cluster distribution pie chart (3 clusters)")
print("  ✓ 20x20 similarity heatmap")
print("  ✓ Response latency bar chart")
print("  ✓ Contradiction analysis (1 contradiction detected)")
print("  ✓ Response gallery (20 paraphrases)")
print("  ✓ Actionable recommendations")
print("\nOpen in browser:")
print(f"  file://{os.path.abspath(report_path)}")