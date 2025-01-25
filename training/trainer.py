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
    Custom trainer class for Question Answering tasks, extending HuggingFace's Trainer.
    Implements custom metric computation for QA evaluation.
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
        We override __init__ to pass compute_metrics=self.compute_metrics
        so that Hugging Face's Trainer will call our custom method.
        """
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,  # Critical line!
            **kwargs,
        )

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute evaluation metrics (Exact Match and F1 score) for QA predictions.

        Args:
            eval_pred (EvalPrediction): 
                - predicitons: (start_logits, end_logits)
                - label_ids: (start_positions, end_positions)

        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics:
                - 'eval_exact_match': Average exact match score
                - 'eval_f1': Average F1 score
        """
        # Unpack predictions and labels
        predictions, labels = eval_pred
        start_logits, end_logits = predictions
        start_positions, end_positions = labels

        # Convert logits to argmax predictions
        start_preds = np.argmax(start_logits, axis=1)
        end_preds = np.argmax(end_logits, axis=1)

        exact_match_scores: List[int] = []
        f1_scores: List[float] = []

        # Go through each example to compute EM and F1
        for i in range(len(start_preds)):
            # Predicted text
            pred_text = self.processing_class.decode(
                self.eval_dataset[i]["input_ids"][start_preds[i] : end_preds[i] + 1],
                skip_special_tokens=True
            )
            # Ground truth text
            true_text = self.processing_class.decode(
                self.eval_dataset[i]["input_ids"][start_positions[i] : end_positions[i] + 1],
                skip_special_tokens=True
            )

            # Calculate Exact Match and F1 for each example
            exact_match = compute_exact_match(pred_text, true_text)
            f1 = compute_f1(pred_text, true_text)

            exact_match_scores.append(exact_match)
            f1_scores.append(f1)

        # Return the average metrics
        return {
            "eval_exact_match": 100 * float(np.mean(exact_match_scores)),
            "eval_f1": 100 * float(np.mean(f1_scores))
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
        model: The model to train (usually a HuggingFace transformer model)
        tokenizer: Tokenizer instance for processing text
        train_dataset: Dataset for training
        eval_dataset: Dataset for evaluation
        config (Dict[str, Any]): Configuration dictionary containing training parameters

    Returns:
        QATrainer: Configured trainer instance ready for training
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

        # if use cuda
        fp16=config["training"]["fp16"],
        
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
        
        # Seed
        seed=config["training"]["seed"],
    )

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer)

    # early_stopping_callback = EarlyStoppingCallback(
    #     early_stopping_patience=config["training"]["early_stopping_patience"],  
    #     early_stopping_threshold=config["training"]["early_stopping_threshold"] 
    # )
    # Initialize our custom QATrainer (which has compute_metrics built-in)
    trainer = QATrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # callbacks=[early_stopping_callback]
    )

    return trainer
