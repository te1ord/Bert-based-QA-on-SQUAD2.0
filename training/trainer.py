import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import os
import wandb
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import DataCollatorWithPadding
from transformers.trainer_utils import EvalPrediction
from metrics.metrics import compute_exact_match, compute_f1


class QATrainer(Trainer):
    """
    Custom Trainer for Question Answering that handles multiple gold answers during evaluation.
    """
    
    def __init__(
        self,
        model: Any = None,
        args: TrainingArguments = None,
        data_collator: Any = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        tokenizer: Any = None,
        **kwargs,
    ):
        """
        Initialize the QATrainer with a custom compute_metrics.
        """
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,  # Assign custom compute_metrics
            **kwargs,
        )
    
    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute EM and F1 by comparing predictions against all gold answers.

        Args:
            eval_pred (EvalPrediction): Predictions and label_ids.

        Returns:
            Dict[str, float]: EM and F1 scores.
        """
        # Unpack predictions and labels
        start_logits, end_logits = eval_pred.predictions
        start_positions, end_positions = eval_pred.label_ids

        # Convert logits to predicted start/end positions
        start_preds = np.argmax(start_logits, axis=1)
        end_preds = np.argmax(end_logits, axis=1)

        exact_matches = []
        f1_scores = []

        for i in range(len(start_preds)):
            example = self.eval_dataset[i]
            input_ids = example["input_ids"]

            # Decode predicted span
            pred_start = int(start_preds[i])
            pred_end = int(end_preds[i])
            if pred_start > pred_end:
                pred_end = pred_start  # Ensure valid span

            pred_tokens = input_ids[pred_start:pred_end+1]
            pred_text = self.processing_class.decode(pred_tokens, skip_special_tokens=True)

            # Retrieve all gold answers for this example
            # Assuming 'answers' field is retained in the eval_dataset
            if "answers" in example and len(example["answers"]["text"]) > 0:
                gold_texts = example["answers"]["text"]
            else:
                # Unanswerable question
                gold_texts = [""]

            # Compute EM and F1 for all gold answers and take the maximum
            em = max(compute_exact_match(pred_text, gt) for gt in gold_texts)
            f1 = max(compute_f1(pred_text, gt) for gt in gold_texts)

            exact_matches.append(em)
            f1_scores.append(f1)

        # Calculate average metrics
        avg_em = 100.0 * np.mean(exact_matches)
        avg_f1 = 100.0 * np.mean(f1_scores)

        return {
            "exact_match": avg_em,
            "f1": avg_f1
        }

def get_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    config: Dict[str, Any]
) -> QATrainer:
    """
    Initialize and configure a QATrainer instance with the specified parameters.

    Args:
        model: The model to train (usually a HuggingFace transformer model).
        tokenizer: Tokenizer instance for processing text.
        train_dataset: Dataset for training.
        eval_dataset: Dataset for evaluation.
        config (Dict[str, Any]): Configuration dictionary containing training parameters.

    Returns:
        QATrainer: Configured trainer instance ready for training.
    """
    # Initialize Weights & Biases
    wandb.login(key=config['wandb']['API_TOKEN'])
    
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        name=config["wandb"]["run_name"],
        config=config
    )

    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        learning_rate=float(config["training"]["learning_rate"]),
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        num_train_epochs=config["training"]["num_train_epochs"],
        weight_decay=config["training"]["weight_decay"],
        warmup_ratio=config["training"]["warmup_ratio"],

        # Use mixed precision if specified
        # fp16=config["training"]["fp16"],
        
        # Logging
        report_to=["wandb"],
        logging_steps=config["training"]["logging_steps"],
        
        # Evaluation
        evaluation_strategy="steps",
        eval_steps=config["training"]["eval_steps"],
        
        # Checkpoints
        save_strategy="steps",
        save_steps=config["training"]["save_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        
        # Best model
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        
        # Seed for reproducibility
        seed=config["training"]["seed"],
    )

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer)

    # Initialize the custom QATrainer
    trainer = QATrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,   # Single-span training dataset
        eval_dataset=eval_dataset,     # Multi-span evaluation dataset
        tokenizer=tokenizer,
        # callbacks=[EarlyStoppingCallback(
        #     early_stopping_patience=config["training"]["early_stopping_patience"],
        #     early_stopping_threshold=config["training"]["early_stopping_threshold"]
        # )]
    )

    return trainer
