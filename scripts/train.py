from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import sys
import os
import yaml
import wandb
from typing import Dict, Any
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.trainer import get_trainer
from data.dataset import SquadDataset

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

def main() -> None:
    """
    Main training function that handles:
    - Device setup
    - Configuration loading
    - Dataset preparation
    - Model initialization
    - Training execution
    - Model saving
    - Weights & Biases logging
    """
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
    
    # Load data and tokenizer
    print("Loading dataset...")
    squad = load_dataset(config["data"]["dataset_path"])
    tokenizer = AutoTokenizer.from_pretrained(config["training"]["model_name"])
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SquadDataset(
        squad["train"], 
        tokenizer, 
        max_length=config["data"]["max_length"],
        is_training=True,
    )
    eval_dataset = SquadDataset(
        squad["validation"],#.select(range(100)), 
        tokenizer, 
        max_length=config["data"]["max_length"],
    )

    # Load model
    print("Loading model...")
    model = AutoModelForQuestionAnswering.from_pretrained(
        config["training"]["model_name"]
    ).to(device)

    # Setup and run training
    trainer = get_trainer(
        model, 
        tokenizer, 
        train_dataset,
        eval_dataset,
        config,
    )
    
    try:
        trainer.train()
        
        # Save final model
        final_output_dir = os.path.join(
            config["training"]["output_dir"],
            "final_model"
        )
        os.makedirs(final_output_dir, exist_ok=True)
        trainer.save_model(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        
        # Log model artifacts to wandb
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}", 
            type="model",
            description="Final trained model"
        )
        artifact.add_dir(final_output_dir)
        wandb.log_artifact(artifact)
        
    except Exception as e:
        wandb.finish(exit_code=1)
        raise e
    
    wandb.finish()

if __name__ == "__main__":
    main()