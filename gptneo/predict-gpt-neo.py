import logging
import math
import os
import json
import sys
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

import datasets
from datasets import load_dataset

import torch
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.10.0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the perplexity on (a text file)."
        },
    )
    write_generation_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional path file to write the generations (json)"
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    max_input: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum number of input sequence length after tokenization for generation. "
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True,
        metadata={"help": "Whether to keep line breaks when using TXT files or not."},
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation/test file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "jsonl",
                    "txt",
                ], "`train_file` should be a csv, a json, a jsonl or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "jsonl",
                    "txt",
                ], "`validation_file` should be a csv, a json, a jsonl or a txt file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "jsonl",
                    "txt",
                ], "`test_file` should be a csv, a json, a jsonl or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()
    training_args = TrainingArguments("test-trainer")

    data_files = {}
    dataset_args = {}
    extension = 'json'
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
    else:
        raise Exception("what is test file ?")
    if extension == "txt":
        extension = "text"
        dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
    if extension == "jsonl":
        extension = "json"
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        **dataset_args,
    )

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        print("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            print(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        ).cuda()
        # model = model.half()
    else:
        model = AutoModelForCausalLM.from_config(config).cuda()
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["test"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    print(column_names, text_column_name)

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base"
    )

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
        return output

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
    if "test" not in tokenized_datasets:
        raise ValueError("do_predict requires a test dataset")
    test_dataset = tokenized_datasets["test"]
    if data_args.max_test_samples is not None:
        test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Prediction
    print("*** Predict ***")
    generations = []
    for idx, example in tqdm(enumerate(test_dataset)):

        # sample_outputs = model.generate(torch.tensor([example['input_ids']]).cuda(), do_sample=True, top_k=5, max_length=2048, num_return_sequences=1)
        sample_outputs = model.generate(torch.tensor([example['input_ids'][-data_args.max_input:]]).cuda(), do_sample=True, top_k=5,
                                        max_length=1024, top_p=0.95, temperature=1.9, num_return_sequences=1)
        input_length = len(example['input_ids'])
        generations.append({'input': raw_datasets['test'][idx]['text'],
                            'target': raw_datasets['test'][idx]['code'],
                            'generation': tokenizer.decode(sample_outputs[0][input_length:], skip_special_tokens=True)})
    print("number of generations: {} ".format(len(generations)))
    with open(data_args.write_generation_path, 'w', encoding='utf-8') as fp:
        json.dump(generations, fp, indent=1)

        # print(sample_outputs)
        # for j, sample_output in enumerate(sample_outputs):
        #     print("{}-{}: {}".format(idx, j, tokenizer.decode(sample_output, skip_special_tokens=True)))

    # max_test_samples = (
    #     data_args.max_test_samples
    #     if data_args.max_test_samples is not None
    #     else len(test_dataset)
    # )
    # metrics["test_samples"] = min(max_test_samples, len(test_dataset))
    # try:
    #     perplexity = math.exp(metrics["test_loss"])
    # except OverflowError:
    #     perplexity = float("inf")
    # metrics["perplexity"] = perplexity


if __name__ == "__main__":
    main()
