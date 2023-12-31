from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )

    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    use_flash_attn: bool = field(
        default=False,
        metadata={
            "help": (
                "Use flash attention implementation."
                "This is set_true argument, do not change default"
            )
        },
    )

    use_xformer_attn: bool = field(
        default=False,
        metadata={
            "help": (
                "Use xformer memory efficient attention implementation."
                "This is set_true argument, do not change default"
            )
        },
    )


@dataclass
class ModelSpecificArguments:
    """
    Arguments pertaining to the specific model to be used.
    """

    scale_attention_softmax_in_fp32: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Scale the attention softmax in fp32. This argument is only relevant for GPTBigCodeForCausalLM."
            )
        },
    )

    attention_softmax_in_fp32: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Compute the attention softmax in fp32. This argument is only relevant for GPTBigCodeForCausalLM."
            )
        },
    )

    alibi: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Use alibi positional embeddings. This argument is only relevant for MPTForCausalLM."
            )
        },
    )

    attn_impl: Optional[str] = field(
        default="triton",
        metadata={
            "help": (
                "Attention implementation. This argument is only relevant for MPTForCausalLM."
            )
        },
    )
