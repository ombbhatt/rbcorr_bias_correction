import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATE = "Jan-21-2025"
SHOT = 'fewshot'

def extract_metrics(file_path):
    """Extract detailed metrics from the last line of CSV files."""
    try:
        with open(file_path, 'r') as file:
            last_line = file.readlines()[-1].strip()
            items = [item.strip().strip('"') for item in last_line.split(',')]
            
            metrics = {}
            for item in items:
                if "TP:" in item:
                    metrics['tp'] = int(item.split("TP:")[1].strip())
                elif "TN:" in item:
                    metrics['tn'] = int(item.split("TN:")[1].strip())
                elif "FP:" in item:
                    metrics['fp'] = int(item.split("FP:")[1].strip())
                elif "FN:" in item:
                    metrics['fn'] = int(item.split("FN:")[1].strip())
                elif "Bias score:" in item:
                    metrics['bias'] = float(item.split("Bias score:")[1].strip())
                elif "Overall bias score:" in item:
                    metrics['bias'] = float(item.split("Overall bias score:")[1].strip())
            return metrics
    except Exception as e:
        logger.warning(f"Could not extract metrics from {file_path}: {str(e)}")
        return None

def calculate_overall_metrics(base_dir, model_family, model, procedure, domains):
    """Calculate overall metrics across all domains for a specific model."""
    total_tp = total_tn = total_fp = total_fn = 0
    bias_scores = []
    
    for domain in domains:
        file_path = Path(base_dir) / SHOT / "EWOK" / procedure / model_family / domain / f"{model}_results.csv"
        metrics = extract_metrics(file_path)
        
        if metrics and all(k in metrics for k in ['tp', 'tn', 'fp', 'fn', 'bias']):
            total_tp += metrics['tp']
            total_tn += metrics['tn']
            total_fp += metrics['fp']
            total_fn += metrics['fn']
            bias_scores.append(metrics['bias'])
    
    total = total_tp + total_tn + total_fp + total_fn
    if total > 0:  # If we have any valid data
        overall_accuracy = (total_tp + total_tn) / total
        # Calculate bias using the same formula with total counts
        overall_bias = (total_tp + total_fp - total_tn - total_fn) / total
        return overall_accuracy, overall_bias
    return None, None

def setup_subplot(ax, plot_type, family):
    """Configure subplot settings."""
    ax.set_ylabel('Accuracy' if plot_type == 'accuracy' else 'Bias Score')
    ax.set_title(f'{family} Model Family Overall EWOK {plot_type.title()} {SHOT}')
    ax.title.set_fontsize(16)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if plot_type == 'accuracy':
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axhspan(0.5, 1, facecolor='gray', alpha=0.1)
        ax.axhspan(0, 0.5, facecolor='gray', alpha=0.05)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    else:
        ax.set_ylim(-1, 1)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        ax.axhspan(0, 1, facecolor='gray', alpha=0.1)
        ax.axhspan(-1, 0, facecolor='gray', alpha=0.05)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))

def create_overall_plots(base_dir=f"outputs/{DATE}", 
                        procedures=None,
                        model_families=None):
    """Create overall accuracy and bias plots for EWOK dataset, showing each model variation."""
    
    EWOK_DOMAINS = ["social_interactions", "social_properties", "material_dynamics", "social_relations", "quantitative_properties", "physical_dynamics", "agent_properties", "physical_interactions", "material_properties", "physical_relations", "spatial_relations"]
    
    ALL_MODEL_FAMILIES = {
        "GPT2": ["gpt2-xl"],
        "Falcon": ["Falcon3-10B-Base", "Falcon3-10B-Instruct", 
                  "Falcon3-Mamba-7B-Base", "Falcon3-Mamba-7B-Instruct"],
        "MPT": ["mpt-7b", "mpt-7b-instruct", "mpt-7b-chat", "mpt-7b-8k",  "mpt-7b-8k-chat"],
        "Phi": ["phi-1", "phi-1_5", "phi-2"],
        "Qwen": ["Qwen1.5-7B", "Qwen1.5-7B-Chat", "Qwen1.5-14B", "Qwen1.5-14B-Chat"],
        "Olmo": ["Olmo-2-1124-7B", "Olmo-2-1124-7B-Instruct", "Olmo-2-1124-13B", "Olmo-2-1124-13B-Instruct"]
    }
    
    procedures = procedures or ["plain", "bos", "kfold"]
    model_families = [f for f in (model_families or ALL_MODEL_FAMILIES.keys())
                     if f in ALL_MODEL_FAMILIES and f != "Pythia"]
    
    # Create plots for each model family
    for family in model_families:
        for plot_type in ['accuracy', 'bias']:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            models = ALL_MODEL_FAMILIES[family]
            x = np.arange(len(models)) * 1.25
            width = 1 / len(procedures)
            colors = plt.cm.tab10(np.linspace(0, 1, len(procedures)))
            
            has_valid_data = False
            
            for proc_idx, procedure in enumerate(procedures):
                values = []
                for model in models:
                    accuracy, bias = calculate_overall_metrics(
                        base_dir, family, model, procedure, EWOK_DOMAINS)
                    if accuracy is not None and bias is not None:
                        value = accuracy if plot_type == 'accuracy' else bias
                        values.append(value)
                        has_valid_data = True
                    else:
                        values.append(0)
                
                if has_valid_data:
                    bars = ax.bar(x + (proc_idx - len(procedures)/2 + 0.5)*width,
                                values, width, label=procedure, color=colors[proc_idx])
                    
                    for bar in bars:
                        height = bar.get_height()
                        if height != 0:
                            if plot_type == 'bias':
                                text_height = height - 0.06 if height >= 0 else height + 0.06
                                va = 'bottom' if height >= 0 else 'top'
                            else:
                                text_height = height - 0.02
                                va = 'bottom'
                            ax.text(bar.get_x() + bar.get_width()/2, text_height,
                                   f'{height:.3f}', ha='center', va=va,
                                   rotation=0, fontsize=6)
            
            if has_valid_data:
                setup_subplot(ax, plot_type, family)
                ax.set_xticks(x)
                ax.set_xticklabels([model.upper() for model in models],
                                 rotation=30, ha='right', fontsize=8)
                ax.legend(title="Procedure", bbox_to_anchor=(1.05, 1), loc='upper left')
                
                plt.tight_layout()
                output_path = Path(base_dir) / SHOT / f'{family.lower()}_EWOKov_{SHOT}_{plot_type}.png'
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Successfully saved overall {plot_type} plot for {family}")
            else:
                logger.warning(f"No valid data found for {family} {plot_type} plot")
            
            plt.close()

if __name__ == "__main__":
    create_overall_plots(
        model_families=["Falcon", "MPT", "Qwen", "Olmo"],
        procedures=["plain", "bos", "kfold"]
    )