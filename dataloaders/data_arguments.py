from typing import Optional
from dataclasses import dataclass, field

from dataloaders.enumerated_file_utils import DATA_KEYS, TRANSLATION_KEYS


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    max_total_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of total examples to this "
                "value if set."
            )
        },
    )

    block_size: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )

    eval_split_percentage: Optional[int] = field(
        default=4,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )

    refactored_base_path: Optional[str] = field(
        default=None,
        metadata={"help": "The base path for refactored data"},
    )

    refactored_style: Optional[str] = field(
        default="base_original",
        # choices=DATA_KEYS.keys(),
        metadata={"help": "The data style for training."},
    )

    final_style: Optional[str] = field(
        default=None,
        # choices=TRANSLATION_KEYS.keys(),
        metadata={"help": "The data style for filtering"},
    )

    filter_on_passed: bool = field(
        default=False,
        metadata={"help": "Whether to filter to only keep passed solutions"},
    )

    plan_pad_merge_count: int = field(
        default=2,
        metadata={"help": "If training only on plans, how many questions to merge"},
    )

    subfunction_pad_merge_count: Optional[int] = field(
        default=1,
        metadata={
            "help": "If training only on subfunctions, how many questions to merge"
        },
    )
