import wandb
import yaml
import torch
import numpy as np
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from typing import Dict, Any, List, Tuple
from datasets import load_dataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import SquadDataset
from metrics.metrics import compute_exact_match, compute_f1

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse configuration from a YAML file.
    
    Args:
        config_path (str): Path to the configuration YAML file
        
    Returns:
        Dict[str, Any]: Parsed configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def get_predictions(
    model: AutoModelForQuestionAnswering,
    dataset: SquadDataset,
    device: torch.device
) -> Tuple[List[str], List[List[str]]]:
    """
    Get predictions from the model for the entire dataset.
    
    Args:
        model: The QA model
        dataset: Dataset to evaluate on
        device: Device to run inference on
        
    Returns:
        Tuple[List[str], List[List[str]]]: Predicted answers and lists of ground truth answers
    """
    model.eval()
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        from tqdm import tqdm  
        for i in tqdm(range(len(dataset)), desc="Evaluating", unit="example"):
            # Get input tensors
            input_ids = torch.tensor(dataset[i]["input_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(dataset[i]["attention_mask"]).unsqueeze(0).to(device)
            token_type_ids = torch.tensor(dataset[i]["token_type_ids"]).unsqueeze(0).to(device)            

            # Get model predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
            # Convert logits to predictions
            start_pred = torch.argmax(start_logits).item()
            end_pred = torch.argmax(end_logits).item()
            if start_pred > end_pred:
                end_pred = start_pred  # Ensure valid span
            
            # Convert token indices to text
            pred_tokens = input_ids[0][start_pred:end_pred + 1]
            pred_text = dataset.tokenizer.decode(pred_tokens, skip_special_tokens=True)
            
            # Get all ground truth answers
            if "answers" in dataset[i] and len(dataset[i]["answers"]["text"]) > 0:
                gold_texts = dataset[i]["answers"]["text"]
            else:
                gold_texts = [""]  # Unanswerable question
            
            predictions.append(pred_text)
            ground_truths.append(gold_texts)

    # print(predictions)
    # print("##########################")
    # print(ground_truths)
    
    return predictions, ground_truths

def evaluate_predictions(
    predictions: List[str],
    ground_truths: List[List[str]]
) -> Dict[str, float]:
    """
    Evaluate predictions using EM and F1 metrics, considering multiple ground truths.
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of lists of ground truth answers
        
    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics
    """
    exact_matches = []
    f1_scores = []
    
    for pred, truths in zip(predictions, ground_truths):
        # Compute metrics against all answers and take the maximum
        em = max(compute_exact_match(pred, gt) for gt in truths)
        f1 = max(compute_f1(pred, gt) for gt in truths)
        
        exact_matches.append(em)
        f1_scores.append(f1)
    
    return {
        "exact_match": 100 * float(np.mean(exact_matches)),
        "f1": 100 * float(np.mean(f1_scores))
    } 

def main() -> None:
    """
    Main inference function that:
    - Sets up device
    - Loads configuration
    - Downloads model from wandb
    - Runs inference on validation set
    - Computes and displays metrics
    """
    # Device setup
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load configuration
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "config.yaml"
    )
    config = load_config(config_path)
    
    
    model_name = config["inference"]["model_name"]
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
    
    # Load validation dataset
    print("Loading validation dataset...")
    squad = load_dataset(config["data"]["dataset_path"])
    val_dataset = SquadDataset(
        squad["validation"],#.select(range(100)),  # Sample 500 examples from validation dataset
        tokenizer,
        config["training"]["max_length"]
    )
    
    # Get predictions
    print("Running inference...")
    predictions, ground_truths = get_predictions(model, val_dataset, device)
    
    # Compute metrics
    print("Computing metrics...")
    metrics = evaluate_predictions(predictions, ground_truths)
    
    # Log results
    print("\nValidation Results:")
    print(f"Exact Match: {metrics['exact_match']:.2f}")
    print(f"F1 Score: {metrics['f1']:.2f}")
    
    # # Log metrics to wandb
    # print("\nValidation Results:")
    # print(f"Exact Match: {metrics['exact_match']:.2f}")
    # print(f"F1 Score: {metrics['f1']:.2f}")
    
    # # Log metrics to wandb
    # wandb.log(metrics)
    # wandb.finish()

if __name__ == "__main__":
    main() 