# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Text to code generation pipeline in CodeXGLUE
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from dataset import concodeDataset
from beam import Beam

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from torch.nn import CrossEntropyLoss

from bleu import _bleu
# from bleu import _bleu, _evaluation_untokenize
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
# from transformers import GPTNeoForCausalLM, GPTNeoConfig, AutoTokenizer, AutoConfig, AutoModelForCausalLM, FlaxAutoModelForCausalLM
# from transformers import GPTNeoForCausalLM, GPTNeoConfig, AutoTokenizer, AutoConfig, AutoModelForCausalLM
# from transformers import T5ForConditionalGeneration

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
}


def read_json(name):
    with open(name, 'r') as f:
        json_file = json.load(f)
    return json_file


def load_and_cache_examples(args, tokenizer, evaluate=False):
    dataset = concodeDataset(tokenizer, args, logger, file_type='dev' if evaluate else 'train',
                             block_size=args.block_size)
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def update_config(model, tokenizer):
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id


def train(args, train_dataset, model, tokenizer, pool):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        args.tensorboard_dir = os.path.join(args.output_dir, 'tensorboard')
        if not os.path.exists(args.tensorboard_dir):
            os.makedirs(args.tensorboard_dir)
        tb_writer = SummaryWriter(args.tensorboard_dir)

    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=True)
    total_examples = len(train_dataset) * (
        torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    batch_size = args.batch_size * args.gradient_accumulation_steps * (
        torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    # if args.max_steps > 0:
    #     t_total = args.max_steps
    #     args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    if args.num_train_epochs > 0:
        t_total = total_examples // batch_size * args.num_train_epochs
    args.max_steps = t_total
    model.to(args.device)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last, map_location="cpu"))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last, map_location="cpu"))
    if args.local_rank == 0:
        torch.distributed.barrier()
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank % args.gpu_per_node],
                                                          output_device=args.local_rank % args.gpu_per_node,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", total_examples)
    logger.info("  Num epoch = %d", t_total * batch_size // total_examples)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb = 0.0, 0.0, 0.0, 0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    best_bleu = 0.0
    best_ppl = 0.0
    if args.evaluate_during_training:
        args.eval_output_dir = args.output_dir
        if not os.path.exists(args.eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.eval_output_dir)
        eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        for step, (batch, token_labels) in enumerate(train_dataloader):
            inputs = batch.to(args.device)
            attn_mask = torch.tensor(token_labels.clone().detach() != 0, dtype=torch.uint8, device=args.device)
            loss_mask = torch.tensor(token_labels.clone().detach() == 2, dtype=torch.uint8, device=args.device)
            model.train()
            # outputs = model(inputs, attention_mask=attn_mask, labels=inputs, loss_mask=loss_mask)
            # loss = outputs[0]
            outputs = model(inputs, attention_mask=attn_mask)
            logits = outputs[0]
            labels = inputs
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            flatten_shift_loss_mask = loss_mask[..., :-1].contiguous().view(-1)
            ids = torch.nonzero(flatten_shift_loss_mask).view(-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[ids], shift_labels.view(-1)[ids])

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if global_step % args.logging_steps == 0:
                    logger.info("epoch {} steps: {} / {}, ppl: {}".format(idx, global_step, len(train_dataloader), avg_loss))
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    logger.info('Evaluation step is left out here.')
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        logger.info(torch.cuda.memory_summary())
                        logger.info(torch.cuda.memory_allocated())
                        results = evaluate(args, model, tokenizer, eval_dataloader, eval_when_training=True)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                            logger.info("  %s = %s", key, round(value, 4))
                        if results["perplexity"] > best_ppl:
                            best_ppl = results["perplexity"]
                            best_output_dir = os.path.join(args.output_dir, 'checkpoint-best')
                            if not os.path.exists(best_output_dir):
                                os.makedirs(best_output_dir)
                            logger.info(f"best ppl updated. saved in {best_output_dir}")
                            logger.info(f"best ppl: {best_ppl}")
                            model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(best_output_dir)
                            tokenizer.save_pretrained(best_output_dir)
                            torch.save(args, os.path.join(best_output_dir, "training_args.bin"))
                            logger.info("Saving model checkpoint to %s", best_output_dir)
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    # _rotate_checkpoints(args, checkpoint_prefix)
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save.save_pretrained(last_output_dir)
                    tokenizer.save_pretrained(last_output_dir)
                    idx_file = os.path.join(last_output_dir, 'idx_file.txt')
                    with open(idx_file, 'w', encoding='utf-8') as idxf:
                        idxf.write(str(0) + '\n')

                    torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", last_output_dir)

                    step_file = os.path.join(last_output_dir, 'step_file.txt')
                    with open(step_file, 'w', encoding='utf-8') as stepf:
                        stepf.write(str(global_step) + '\n')

                    model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save.save_pretrained(last_output_dir)
                    tokenizer.save_pretrained(last_output_dir)
                    torch.save(args, os.path.join(last_output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", last_output_dir)
                    logger.info(torch.cuda.memory_summary())
                    logger.info(torch.cuda.memory_allocated())
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.info(torch.cuda.memory_summary())
                    logger.info(torch.cuda.memory_allocated())

            if args.max_steps > 0 and global_step > args.max_steps:
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, eval_dataloader, prefix="", eval_when_training=False):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    # logger.info("***** Running evaluation {} *****".format(prefix))
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for step, (batch, token_labels) in enumerate(eval_dataloader):
        inputs = batch.to(args.device)
        attn_mask = torch.tensor(token_labels.clone().detach() != 0, dtype=torch.uint8, device=args.device)
        loss_mask = torch.tensor(token_labels.clone().detach() == 2, dtype=torch.uint8, device=args.device)
        with torch.no_grad():
            outputs = model(inputs, attention_mask=attn_mask)
            logits = outputs[0]
            labels = inputs
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            flatten_shift_loss_mask = loss_mask[..., :-1].contiguous().view(-1)
            ids = torch.nonzero(flatten_shift_loss_mask).view(-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[ids], shift_labels.view(-1)[ids])
            eval_loss += loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    del batch
    del token_labels
    del inputs
    del attn_mask
    del loss_mask
    del outputs
    del logits
    del labels
    del shift_logits
    del shift_labels
    del flatten_shift_loss_mask
    del ids
    del loss
    del eval_loss
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


    result = {
        "perplexity": float(perplexity)
    }

    output_eval_file = os.path.join(args.eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        # logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            # logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def eval_bleu(args, model, tokenizer, file_type='test', num=2000):
    dataset = concodeDataset(tokenizer, args, logger, file_type=file_type, block_size=args.block_size, mode='test')
    test_sampler = SequentialSampler(dataset)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=1)
    model.to(args.device)
    model.zero_grad()
    model.eval()

    preds = []
    # max_gen_len = 100
    for step, (batch, token_labels) in enumerate(test_dataloader):
        if step >= num:
            break
        inputs = batch.to(args.device)

        with torch.no_grad():
            beam_size = 5
            m = torch.nn.LogSoftmax(dim=-1)
            outputs = model(inputs)[1]
            p = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(inputs.shape[0]):
                past_hidden = [x[:, i:i + 1].expand(-1, beam_size, -1, -1, -1) for x in outputs]
                # past_hidden = [x[:, i:i+1].expand(-1, beam_size, -1, -1) for x in outputs]
                # past_hidden = [x[:, i:i+1].expand(-1, beam_size, -1, -1) for x in outputs]
                # context_mask=source_mask[i:i+1,:].expand(beam_size,-1)
                beam = Beam(beam_size, tokenizer.bos_token_id, tokenizer.eos_token_id)
                input_ids = None
                for _ in range(args.max_gen_len):
                    if beam.done():
                        break
                    input_ids = beam.getCurrentState()
                    # context_mask=torch.cat((context_mask,input_ids*0+1),-1)
                    # mask=context_mask.unsqueeze(0).unsqueeze(-2).unsqueeze(-2).expand(self.config.n_layer, -1, -1, -1, -1)
                    transformer_outputs = model(input_ids, past=past_hidden)
                    out = m(transformer_outputs[0][:, -1, :]).data
                    # out = self.lsm(self.lm_head(transformer_outputs[0][:,-1,:])).data
                    beam.advance(out)
                    past_hidden = [x.data.index_select(1, beam.getCurrentOrigin()) for x in transformer_outputs[1]]
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:beam_size]

                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (args.max_gen_len - len(p))).view(1, -1) for p in pred]
                p.append(torch.cat(pred, 0).unsqueeze(0))
            p = torch.cat(p, 0)
            for pred in p:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                # print(text)
                preds.append(text)

        if step % args.logging_steps == 0:
            logger.info(f"{step} are done!")

    golds = []
    datafile = os.path.join(args.data_dir, f"{file_type}.json")
    # datas = open(datafile).readlines()
    datas = read_json(datafile)
    for x in datas[:num]:
        # x = json.loads(x)
        golds.append(x["code"])

    assert len(preds) == len(golds)
    print("length preds: {}, golds: {}".format(len(preds), len(golds)))

    EM = []
    with open(os.path.join(args.prediction_dir, f"{file_type}.output"), 'w', encoding='utf-8') as f, open(
            os.path.join(args.prediction_dir, f"{file_type}.gold"), 'w', encoding='utf-8') as f1:
        for pred, gold in zip(preds, golds):
            f.write(pred + '##########\n')
            f1.write(gold + '##########\n')
            EM.append(pred.split() == gold.split())

    bleu_score = round(
        _bleu(os.path.join(args.prediction_dir, f"{file_type}.gold"), os.path.join(args.prediction_dir, f"{file_type}.output")),
        2)
    EM = round(np.mean(EM) * 100, 2)
    Results = {'pl': 0}
    # Results = _evaluation_untokenize(os.path.join(args.prediction_dir, f"{file_type}.gold"), os.path.join(args.prediction_dir, f"{file_type}.output"))
    return bleu_score, EM, Results


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data path.")
    parser.add_argument("--langs", default=None, type=str, required=True,
                        help="Languages to train, if all, train all languages in data_dir")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--prediction_dir", default=None, type=str,
                        help="The output directory where the model predictions will be written.")

    ## Other parameters
    parser.add_argument("--model_type", default="gpt2", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--pretrain_dir", default="", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--config_dir", type=str,
                        help="config name. Required when training from scratch")
    parser.add_argument("--tokenizer_dir", type=str,
                        help="Pre-trained tokenizer dir. Required when training from scratch")
    parser.add_argument("--load_name", type=str, default="pretrained",
                        help="Load pretrained model name")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=1024, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--dev_file_type", default="dev", type=str,)
    parser.add_argument("--eval_output_dir", default="dev", type=str,)
    parser.add_argument("--test_file_type", default="dev", type=str,)
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_infer", action='store_true',
                        help="Whether to run inference on test set.")
    parser.add_argument("--evaluate_during_training", action='store_true', default=False,
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_gen_len", default=100, type=int,
                        help="Maximum length for generation.")

    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--node_index", type=int, default=-1,
                        help="node index if multi-node running")
    parser.add_argument("--gpu_per_node", type=int, default=-1,
                        help="num of gpus per node")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    # parser.add_argument('--log_file', type=str, default='')
    parser.add_argument('--tensorboard_dir', type=str)

    pool = None
    args = parser.parse_args()

    # args.output_dir = os.path.join(args.output_dir, args.dataset)

    if args.model_type in ["bert", "roberta", "distilbert"] and not args.mlm:
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
                         "flag (masked language modeling).")

    if not args.prediction_dir:
        args.prediction_dir = args.output_dir
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    logger.warning(
        "local_rank: %d, node_index: %d, gpu_per_node: %d" % (args.local_rank, args.node_index, args.gpu_per_node))
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.local_rank += args.node_index * args.gpu_per_node
        args.n_gpu = 1
    args.device = device
    # args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s, world size: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16,
        torch.distributed.get_world_size() if args.local_rank != -1 else 1)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if args.do_train and os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.pretrain_dir = os.path.join(checkpoint_last)
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    # Load pre-trained model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    pretrained = args.pretrain_dir
    if pretrained:
        if args.model_type == 'gpt2':
            tokenizer = tokenizer_class.from_pretrained(pretrained, do_lower_case=args.do_lower_case, bos_token='<s>',
                                                        eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>',
                                                        sep_token='juice_elem_sep')
            logger.info(tokenizer.encode("<s> hello world <pad> </s>"))
        else:
            tokenizer = tokenizer_class.from_pretrained(pretrained, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(pretrained)
        model.resize_token_embeddings(len(tokenizer))
        update_config(model, tokenizer)
        logger.info(model.config)
    else:
        if args.model_type == 'gpt2':
            tokenizer = tokenizer_class.from_pretrained(args.tokenizer_dir, bos_token='<s>', eos_token='</s>',
                                                        pad_token='<pad>', unk_token='<|UNKNOWN|>',
                                                        sep_token='juice_elem_sep')
            args.vocab_size = tokenizer.vocab_size
        else:
            tokenizer = tokenizer_class.from_pretrained(pretrained, do_lower_case=args.do_lower_case)
        config = config_class.from_pretrained(args.config_dir)
        model = model_class(config)
        model.resize_token_embeddings(len(tokenizer))
        update_config(model, tokenizer)

    model_parameters = model.parameters()
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f"Model has a total of {num_params} trainable parameters")

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, pool)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_eval:  # only works on 1 GPU
        dev_bleu, dev_EM, dev_Result = eval_bleu(args, model, tokenizer, file_type='dev', num=4000)
        logger.info("dev bleu: {}, dev EM: {}, dev Result: {}".format(dev_bleu, dev_EM, dev_Result))

    if args.do_infer:  # only works on 1 GPU
        test_bleu, test_EM, test_Result = eval_bleu(args, model, tokenizer, file_type='test', num=534)
        logger.info("534 test bleu: {}, 534 test EM: {}, 534 test Result: {}".format(test_bleu, test_EM, test_Result))


if __name__ == "__main__":
    main()
