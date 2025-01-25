from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import DefaultDataCollator, PreTrainedTokenizer
from typing import Dict, List, Any, Union
import torch

class SquadDataset(Dataset):
    """
    Enhanced PyTorch Dataset for SQuAD with batched preprocessing.
    
    Attributes:
        tokenizer (PreTrainedTokenizer): Tokenizer for processing text
        max_length (int): Maximum sequence length for tokenization
        data_collator (DefaultDataCollator): Collator for batch processing
        dataset: Processed dataset containing tokenized examples
    """
    
    def __init__(self, 
                 dataset: Any,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 384) -> None:
        """
        Initialize the SQuAD dataset.
        
        Args:
            dataset: Raw dataset containing questions and contexts
            tokenizer (PreTrainedTokenizer): Tokenizer for processing text
            max_length (int, optional): Maximum sequence length. Defaults to 384.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_collator = DefaultDataCollator()
        
        # Load dataset using Hugging Face datasets library
        self.dataset = dataset
        
        # Preprocess with batched operations
        self.dataset = self.dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.dataset.column_names,
            fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length}
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.dataset[idx]

    @staticmethod
    def preprocess_function(
        examples: Dict[str, List[Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int
    ) -> Dict[str, torch.Tensor]:
        """
        Batched preprocessing function with proper context truncation and answer mapping.
        
        Args:
            examples (Dict[str, List[Any]]): Batch of examples to process
            tokenizer (PreTrainedTokenizer): Tokenizer for processing text
            max_length (int): Maximum sequence length
            
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
            if not answer["text"]:
                start_positions.append(0)
                end_positions.append(0)
                continue

            # Use only first answer for training
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find context boundaries
            context_start = 0
            while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
                context_start += 1
                
            context_end = len(sequence_ids) - 1
            while context_end >= 0 and sequence_ids[context_end] != 1:
                context_end -= 1

            # Check if answer is fully inside the context
            if (offsets[context_start][0] > end_char or 
                offsets[context_end][1] < start_char):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Find start token
                idx = context_start
                while idx <= context_end and offsets[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                # Find end token
                idx = context_end
                while idx >= context_start and offsets[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = torch.tensor(start_positions)
        inputs["end_positions"] = torch.tensor(end_positions)

        # inputs["input_ids"] = torch.tensor(inputs["input_ids"])
        # inputs["attention_mask"] = torch.tensor(inputs["attention_mask"])
        return inputs
    
