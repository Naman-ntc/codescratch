import sys
import logging
import pathlib

import torch
import datasets
import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from dataloaders import DataArguments, build_refactored_datasets, build_oai_datasets
from model_arguments import ModelArguments, ModelSpecificArguments
from utils.monkey_patches import replace_attn_with_xformer, replace_attn_with_flash_attn


logger = logging.getLogger(__name__)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def setup_logging(training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


def setup_monkey_patches():
    if any(arg in sys.argv for arg in ["--use_flash_attn"]):
        print(f"Using flash attention")
        replace_attn_with_flash_attn()
    if any(arg in sys.argv for arg in ["--use_xformer_attn"]):
        print(f"Using xformer mem-efficient attention")
        replace_attn_with_xformer()
    return


def main():
    setup_monkey_patches()

    all_argument_classes = (
        ModelArguments,
        ModelSpecificArguments,
        DataArguments,
        TrainingArguments,
    )

    parser = HfArgumentParser(all_argument_classes)

    (
        model_args,
        model_specific_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    setup_logging(training_args)
    # Set seed before initializing model.
    set_seed(training_args.seed)
    setattr(data_args, "seed", training_args.seed)
    setattr(data_args, "cache_dir", model_args.cache_dir)

    # Load the tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": True,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": True,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, **tokenizer_kwargs
    )

    # Load the dataset
    if data_args.oai_mode:
        train_dataset, eval_dataset = build_oai_datasets(tokenizer, data_args)
    else:
        train_dataset, eval_dataset = build_refactored_datasets(tokenizer, data_args)

    # Load the model
    config_kwargs = {
        "use_cache": False,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": True,
    }

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    if config.architectures[0] == "MPTForCausalLM":
        config.attn_config["attn_impl"] = model_specific_args.attn_impl
        config.attn_config["alibi"] = model_specific_args.alibi
    if config.architectures[0] == "GPTBigCodeForCausalLM":
        config.scale_attention_softmax_in_fp32 = (
            model_specific_args.scale_attention_softmax_in_fp32
        )
        config.attention_softmax_in_fp32 = model_specific_args.attention_softmax_in_fp32

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
    )

    # Setup metrics
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=training_args.output_dir + "/final_model"
    )


if __name__ == "__main__":
    main()
