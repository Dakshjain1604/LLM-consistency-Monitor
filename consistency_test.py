import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

from src.paraphrase_generator import ParaphraseGenerator
from src.llm_tester import LLMTester, create_llm_tester
from src.consistency_analyzer import ConsistencyAnalyzer
from src.report_builder import ReportBuilder
from src.utils import setup_logging

load_dotenv()
console = Console()
logger = setup_logging()

@click.command()
@click.option('--question', '-q', type=str, help='Question to test')
@click.option('--model', '-m', type=click.Choice(['claude', 'gpt4', 'custom']), help='LLM model to test (legacy)')
@click.option('--provider', '-p', type=click.Choice(['claude', 'gpt4', 'custom', 'huggingface', 'local']), help='LLM provider (extended options)')
@click.option('--batch', '-b', type=click.Path(exists=True), help='Path to JSON file with multiple questions')
@click.option('--custom-endpoint', type=str, help='Custom LLM endpoint URL (for custom model)')
@click.option('--base-url', type=str, help='Base URL for OpenAI-compatible endpoints (for local provider)')
@click.option('--hf-model', type=str, default='meta-llama/Llama-3.3-70B-Instruct', help='HuggingFace model name')
@click.option('--output-dir', '-o', type=str, default='./results', help='Output directory for reports')
@click.option('--test-mode', type=click.Choice(['standard', 'stress']), default='standard', help='Test mode: standard paraphrases or stress test prompts')
def main(question, model, provider, batch, custom_endpoint, base_url, hf_model, output_dir, test_mode):
    console.print(Panel.fit(
        "[bold blue]🔍 NEO Consistency Monitor[/bold blue]\n"
        "LLM Response Consistency Testing Tool",
        border_style="blue"
    ))
    
    active_provider = provider or model
    
    if batch:
        run_batch_mode(batch, active_provider, custom_endpoint, base_url, hf_model, output_dir, test_mode)
    else:
        if not question:
            question = click.prompt("Enter your question", type=str)
        
        if not active_provider:
            console.print("\n[bold]Select LLM Provider:[/bold]")
            console.print("1. Claude (Anthropic)")
            console.print("2. GPT-4 (OpenAI)")
            console.print("3. HuggingFace Inference API")
            console.print("4. Local OpenAI-compatible (Ollama, vLLM, LM Studio)")
            console.print("5. Custom HTTP endpoint")
            choice = click.prompt("Choice", type=int, default=1)
            active_provider = ['claude', 'gpt4', 'huggingface', 'local', 'custom'][choice - 1]
        
        if active_provider == 'custom' and not custom_endpoint:
            custom_endpoint = click.prompt("Enter custom endpoint URL", type=str)
        
        if active_provider == 'local' and not base_url:
            base_url = click.prompt("Enter OpenAI-compatible base URL", type=str, default="http://localhost:8000/v1")
        
        run_single_test(question, active_provider, custom_endpoint, base_url, hf_model, output_dir, test_mode)

def run_single_test(question: str, model_type: str, custom_endpoint: str, base_url: str, hf_model: str, output_dir: str, test_mode: str):
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            if test_mode == 'stress':
                task1 = progress.add_task("[cyan]Generating stress test prompts (25 variations)...", total=1)
                
                from src.prompt_engineer import generate_stress_prompts
                stress_dict = asyncio.run(generate_stress_prompts(question))
                
                all_paraphrases = []
                for category_list in stress_dict.values():
                    all_paraphrases.extend(category_list)
                
                progress.update(task1, completed=1)
            else:
                task1 = progress.add_task("[cyan]Generating 20 paraphrases...", total=1)
                
                generator = ParaphraseGenerator()
                paraphrases_dict = generator.generate_paraphrases(question)
                
                all_paraphrases = []
                for style_list in paraphrases_dict.values():
                    all_paraphrases.extend(style_list)
                
                progress.update(task1, completed=1)
            
            task2 = progress.add_task(f"[cyan]Testing on {model_type}...", total=len(all_paraphrases))
            
            if model_type == 'huggingface':
                tester = create_llm_tester('huggingface', model=hf_model)
            elif model_type == 'local':
                tester = create_llm_tester('local', base_url=base_url)
            elif model_type == 'custom':
                tester = create_llm_tester('custom', endpoint=custom_endpoint)
            else:
                tester = create_llm_tester(model_type)
            
            test_results = asyncio.run(tester.test_paraphrases(all_paraphrases))
            
            progress.update(task2, completed=len(all_paraphrases))
            
            task3 = progress.add_task("[cyan]Analyzing consistency...", total=1)
            
            analyzer = ConsistencyAnalyzer()
            analysis = analyzer.analyze(test_results)
            
            progress.update(task3, completed=1)
            
            task4 = progress.add_task("[cyan]Generating report...", total=1)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_question = "".join(c if c.isalnum() else "_" for c in question[:30])
            mode_suffix = "_stress" if test_mode == 'stress' else ""
            report_filename = f"consistency_report_{safe_question}{mode_suffix}_{timestamp}.html"
            report_path = os.path.join(output_dir, report_filename)
            
            builder = ReportBuilder()
            report_path = builder.generate_report(question, test_results, analysis, report_path)
            
            progress.update(task4, completed=1)
        
        display_results(question, analysis, report_path, test_mode)
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)

def run_batch_mode(batch_file: str, model_type: str, custom_endpoint: str, base_url: str, hf_model: str, output_dir: str, test_mode: str):
    with open(batch_file, 'r') as f:
        questions = json.load(f)
    
    mode_label = "stress test" if test_mode == 'stress' else "standard"
    console.print(f"\n[bold]Testing {len(questions)} questions in batch mode ({mode_label})[/bold]\n")
    
    if not model_type:
        console.print("Select LLM Provider:")
        console.print("1. Claude (Anthropic)")
        console.print("2. GPT-4 (OpenAI)")
        console.print("3. HuggingFace Inference API")
        console.print("4. Local OpenAI-compatible")
        console.print("5. Custom HTTP endpoint")
        choice = click.prompt("Choice", type=int, default=1)
        model_type = ['claude', 'gpt4', 'huggingface', 'local', 'custom'][choice - 1]
    
    if model_type == 'custom' and not custom_endpoint:
        custom_endpoint = click.prompt("Enter custom endpoint URL", type=str)
    
    if model_type == 'local' and not base_url:
        base_url = click.prompt("Enter OpenAI-compatible base URL", type=str, default="http://localhost:8000/v1")
    
    results = []
    
    for i, q_obj in enumerate(questions, 1):
        question = q_obj['question']
        category = q_obj.get('category', 'general')
        
        console.print(f"\n[bold cyan]Question {i}/{len(questions)}:[/bold cyan] {question}")
        
        try:
            if test_mode == 'stress':
                from src.prompt_engineer import generate_stress_prompts
                stress_dict = asyncio.run(generate_stress_prompts(question))
                all_paraphrases = []
                for category_list in stress_dict.values():
                    all_paraphrases.extend(category_list)
            else:
                generator = ParaphraseGenerator()
                paraphrases_dict = generator.generate_paraphrases(question)
                all_paraphrases = []
                for style_list in paraphrases_dict.values():
                    all_paraphrases.extend(style_list)
            
            if model_type == 'huggingface':
                tester = create_llm_tester('huggingface', model=hf_model)
            elif model_type == 'local':
                tester = create_llm_tester('local', base_url=base_url)
            elif model_type == 'custom':
                tester = create_llm_tester('custom', endpoint=custom_endpoint)
            else:
                tester = create_llm_tester(model_type)
            
            test_results = asyncio.run(tester.test_paraphrases(all_paraphrases))
            
            analyzer = ConsistencyAnalyzer()
            analysis = analyzer.analyze(test_results)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_question = "".join(c if c.isalnum() else "_" for c in question[:30])
            mode_suffix = "_stress" if test_mode == 'stress' else ""
            report_filename = f"batch_{i}_{safe_question}{mode_suffix}_{timestamp}.html"
            report_path = os.path.join(output_dir, report_filename)
            
            builder = ReportBuilder()
            builder.generate_report(question, test_results, analysis, report_path)
            
            results.append({
                'question': question,
                'category': category,
                'score': analysis['consistency_score'],
                'clusters': analysis['num_clusters'],
                'contradictions': len(analysis['contradictions']),
                'report': report_path
            })
            
            console.print(f"[green]✓[/green] Score: {analysis['consistency_score']}%")
            
        except Exception as e:
            console.print(f"[red]✗ Failed:[/red] {str(e)}")
            logger.error(f"Batch question {i} failed: {e}")
    
    display_batch_results(results)

def display_results(question: str, analysis: dict, report_path: str, test_mode: str = 'standard'):
    score = analysis['consistency_score']
    
    if score >= 80:
        emoji = "✓"
        color = "green"
        status = "GOOD"
    elif score >= 60:
        emoji = "⚠️"
        color = "yellow"
        status = "MEDIUM"
    else:
        emoji = "❌"
        color = "red"
        status = "POOR"
    
    mode_label = " (STRESS TEST)" if test_mode == 'stress' else ""
    console.print(f"\n[bold {color}]{emoji} CONSISTENCY SCORE: {score}% ({status}){mode_label}[/bold {color}]")
    
    table = Table(title="Analysis Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Response Clusters", str(analysis['num_clusters']))
    table.add_row("Contradictions Found", str(len(analysis['contradictions'])))
    table.add_row("Test Mode", test_mode.upper())
    table.add_row("Report Location", report_path)
    
    console.print(table)
    
    if analysis['contradictions']:
        console.print("\n[bold yellow]⚠️  Contradictions Detected:[/bold yellow]")
        for c in analysis['contradictions'][:3]:
            console.print(f"  • Cluster {c['cluster_a']} vs {c['cluster_b']}: {c['description'][:80]}...")

def display_batch_results(results: list):
    console.print("\n[bold]Batch Test Summary[/bold]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Question", style="cyan", width=40)
    table.add_column("Category", style="white")
    table.add_column("Score", style="white", justify="right")
    table.add_column("Status", style="white")
    table.add_column("Clusters", justify="right")
    table.add_column("Contradictions", justify="right")
    
    for r in sorted(results, key=lambda x: x['score']):
        score = r['score']
        if score >= 80:
            status = "[green]✓ GOOD[/green]"
        elif score >= 60:
            status = "[yellow]⚠️ MEDIUM[/yellow]"
        else:
            status = "[red]❌ POOR[/red]"
        
        table.add_row(
            r['question'][:37] + "..." if len(r['question']) > 40 else r['question'],
            r['category'],
            f"{score}%",
            status,
            str(r['clusters']),
            str(r['contradictions'])
        )
    
    console.print(table)
    
    critical = [r for r in results if r['score'] < 60]
    if critical:
        console.print(f"\n[bold red]⚠️  {len(critical)} Critical Issues Requiring Attention:[/bold red]")
        for r in critical:
            console.print(f"  • {r['question'][:60]}... (Score: {r['score']}%)")

if __name__ == '__main__':
    main()