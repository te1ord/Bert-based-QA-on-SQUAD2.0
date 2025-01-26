from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import DefaultDataCollator, PreTrainedTokenizer
from typing import Dict, List, Any, Union, Optional
import torch
import numpy as np

class SquadDataset(Dataset):
    """
    Enhanced PyTorch Dataset for SQuAD with batched preprocessing and multi-answer handling.
    
    Attributes:
        tokenizer (PreTrainedTokenizer): Tokenizer for processing text
        max_length (int): Maximum sequence length for tokenization
        data_collator (DefaultDataCollator): Collator for batch processing
        dataset: Processed dataset containing tokenized examples
    """
    
    def __init__(self, 
                 dataset: Any,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 384,
                 is_training: bool = True) -> None:
        """
        Initialize the SQuAD dataset.
        
        Args:
            dataset: Raw dataset containing questions and contexts
            tokenizer (PreTrainedTokenizer): Tokenizer for processing text
            max_length (int, optional): Maximum sequence length. Defaults to 384.
            is_training (bool): Whether this is for training (affects answer selection)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_collator = DefaultDataCollator()
        
        # Load dataset and preprocess
        self.dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=[col for col in dataset.column_names if col != "answers"],
            fn_kwargs={
                'tokenizer': tokenizer, 
                'max_length': max_length,
                'is_training': is_training
            }
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.dataset[idx]

    @staticmethod
    def preprocess_function(
        examples: Dict[str, List[Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        is_training: bool
    ) -> Dict[str, torch.Tensor]:
        """
        Batched preprocessing function with proper context truncation and answer mapping.
        
        Args:
            examples (Dict[str, List[Any]]): Batch of examples to process
            tokenizer (PreTrainedTokenizer): Tokenizer for processing text
            max_length (int): Maximum sequence length
            is_training (bool): Whether to select random answers for training
            
        Returns:
            Dict[str, torch.Tensor]: Processed examples with tokenized inputs and answer positions
        """
        questions = [q.strip() for q in examples["question"]]
        contexts = examples["context"]
        
        # Tokenize with context-only truncation
        inputs = tokenizer(
            questions,
            contexts,
            max_length=max_length,
            truncation="only_second",
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        
        offset_mapping = inputs.pop("offset_mapping").numpy().tolist()
        answers = examples["answers"]
        start_positions: List[int] = []
        end_positions: List[int] = []

        for i, offsets in enumerate(offset_mapping):
            answer = answers[i]
            sequence_ids = inputs.sequence_ids(i)
            
            # Find context boundaries
            context_start = 0
            while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
                context_start += 1
                
            context_end = len(sequence_ids) - 1
            while context_end >= 0 and sequence_ids[context_end] != 1:
                context_end -= 1

            # Initialize positions
            start_char = end_char = 0
            if answer["text"]:
                # Select answer strategy
                num_answers = len(answer["text"])
                if is_training and num_answers > 1:
                    # Random selection for training
                    selected_idx = np.random.randint(0, num_answers)
                else:
                    # First answer for evaluation
                    selected_idx = 0

                start_char = answer["answer_start"][selected_idx]
                end_char = start_char + len(answer["text"][selected_idx])

            # Find token positions
            if start_char == end_char == 0:  # No answer
                start_positions.append(0)
                end_positions.append(0)
            elif (offsets[context_start][0] > end_char or 
                  offsets[context_end][1] < start_char):
                # Answer is outside the context
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Find start token
                idx = context_start
                while idx <= context_end and offsets[idx][0] <= start_char:
                    idx += 1
                start_pos = idx - 1

                # Find end token
                idx = context_end
                while idx >= context_start and offsets[idx][1] >= end_char:
                    idx -= 1
                end_pos = idx + 1

                start_positions.append(start_pos)
                end_positions.append(end_pos)

        inputs["start_positions"] = torch.tensor(start_positions)
        inputs["end_positions"] = torch.tensor(end_positions)
        return inputs