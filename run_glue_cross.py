""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

import argparse
from log import logger
from sklearn.model_selection import KFold
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from prior_wd_optim import PriorWD
import pandas as pd
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertModel,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    FlaubertConfig,
    FlaubertForSequenceClassification,
    FlaubertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
    BertForSequenceClassification
)
from models import RobertaForSequenceClassification
from utils import write_to_csv, set_seed
from data import convert_examples_to_features, processors, output_modes

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}



def convert_examples(args, task, tokenizer, label_list, examples):
    output_mode = output_modes[task]
    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=args.max_seq_length,
        pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        output_mode=output_mode,
        no_label=False
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_dropout", action="store_true", help="If specified, uses the information bottleneck to reduce\
            the dimensions.")
    parser.add_argument("--mixout", type=float, default=0.0, help="mixout probability (default: 0)")
    parser.add_argument(
        "--prior_weight_decay", action="store_true", help="Weight Decaying toward the bert params",
    )
    parser.add_argument("--kl_annealing", choices=[None, "linear"], default=None)
    parser.add_argument("--evaluate_after_each_epoch", action="store_true", help="Eveluates the model after\
            each epoch and saves the best model.")
    parser.add_argument("--deterministic", action="store_true", help="If specified, learns the reduced dimensions\
            through mlp in a deterministic manner.")
    parser.add_argument("--activation", type=str, choices=["tanh", "sigmoid", "relu"], \
                        default="relu")
    parser.add_argument("--eval_types", nargs="+", type=str, default=["train", "test"], \
                        choices=["train", "test", "dev"], help="Specifies the types to evaluate on,\
                            can be dev, test, train.")
    parser.add_argument("--binarize_eval", action="store_true", help="If specified, binarize the predictions, and\
            labels during the evaluations in case of binary-class datasets.")
    # Ib parameters.
    parser.add_argument("--beta", type=float, default=1.0, help="Defines the weight for the information bottleneck\
            loss.")
    parser.add_argument("--ib", action="store_true", help="If specified, uses the information bottleneck to reduce\
            the dimensions.")
    parser.add_argument("--sample_size", type=int, default=5, help="Defines the number of samples for the ib method.")
    parser.add_argument("--ib_dim", default=128, type=int,
                        help="Specifies the dimension of the information bottleneck.")

    # Required parameter
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--eval_tasks", nargs="+", default=[], type=str, help="Specifies a list of evaluation tasks.")
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=64, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--n_splits",
        default=None,
        type=int,
        required=True,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--cuda",
        default="cuda",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument('--seeds', nargs='+', type=int, required=False)
    args = parser.parse_args()

    args.device = torch.device(args.cuda)

    args.task_to_data_dir = {
        "smell": "./data/smell/",
        "complexity": "./data/complexity/",
        "read": "./data/read/",
    }
    return args


def collection(examples, indices):
    temp = []
    for i in indices:
        temp.append(examples[i])
    return temp


def get_config(args):
    processor = processors[args.task_name](args.task_to_data_dir[args.task_name])
    label_list = processor.get_labels()
    num_labels = len(label_list)
    config_class, _, _ = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
    )
    if args.model_type in ["bert", "roberta"]:
        # bert dim is 768.
        args.hidden_dim = (768 + args.ib_dim) // 2
    # sets the parameters of IB or MLP baseline.
    config.ib = args.ib
    config.activation = args.activation
    config.hidden_dim = args.hidden_dim
    config.ib_dim = args.ib_dim
    config.beta = args.beta
    config.sample_size = args.sample_size
    config.kl_annealing = args.kl_annealing
    config.deterministic = args.deterministic
    config.use_dropout = args.use_dropout
    return config


def get_train_loader(train_examples, tokenizer, labels, args):
    train_dataset = convert_examples(args, args.task_name, tokenizer, labels, train_examples)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    return train_dataloader


def get_test_loader(test_examples, tokenizer, labels, args):
    test_dataset = convert_examples(args, args.task_name, tokenizer, labels, test_examples)
    test_sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

    return test_dataloader


def load_model(model_class, args, config, total_step):
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    if args.prior_weight_decay:  # I am just addding this because revisiting bert few-sample added it. should be checked.
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
                          correct_bias=True, weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.prior_weight_decay:
        optimizer = PriorWD(optimizer, use_prior_wd=args.prior_weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_step)
    if args.mixout > 0:
        from mixout import MixLinear
        for sup_module in model.modules():
            for name, module in sup_module.named_children():
                if isinstance(module, nn.Dropout):
                    module.p = 0.0
                if isinstance(module, nn.Linear) and not ('output' in name and 'attention' not in name):
                    target_state_dict = module.state_dict()
                    bias = True if module.bias is not None else False
                    new_module = MixLinear(
                        module.in_features, module.out_features, bias, target_state_dict["weight"], args.mixout
                    ).to(args.cuda)
                    new_module.load_state_dict(target_state_dict)
                    setattr(sup_module, name, new_module)
    return model, optimizer, scheduler


def compute_metrics(out_label_ids, preds_label):
    result = {
        'f1': f1_score(y_true=out_label_ids, y_pred=preds_label, average='macro'),
        'recall': recall_score(y_true=out_label_ids, y_pred=preds_label, average='macro'),
        'precision': precision_score(out_label_ids, preds_label, average='macro'),
        'acc': accuracy_score(out_label_ids, preds_label)
    }
    return result


def main(args, results, seed):
    args.output_mode = output_modes[args.task_name]
    args.model_type = args.model_type.lower()
    args.train_batch_size = args.per_gpu_train_batch_size
    args.eval_batch_size = args.per_gpu_eval_batch_size
    processor = processors[args.task_name](args.task_to_data_dir[args.task_name])
    label_list = processor.get_labels()
    _, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    )
    config = get_config(args)
    kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=seed)
    examples = processor.get_examples()
    splits = kfold.split(examples)
    for n_split, (train_indices, valid_indices) in enumerate(splits):
        logger.info("run fold {}".format(n_split))
        train_examples = collection(examples, train_indices)
        val_examples = collection(examples, valid_indices)
        train_dataloader = get_train_loader(train_examples, tokenizer, processor.get_labels(), args)
        model, optimizer, scheduler = load_model(model_class, args, config,
                                                 len(train_dataloader) * args.num_train_epochs)
        model.to(args.device)
        model.zero_grad()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0.0
            train_step = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs, epoch=epoch)
                loss = outputs["loss"]["loss"]  # model outputs are always tuple in transformers (see doc)
                loss.backward()
                tr_loss += loss.item()
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                train_step += 1

            eval_dataloader = get_test_loader(val_examples, tokenizer, label_list, args)
            # Eval!
            eval_loss = 0.0
            nb_eval_steps = 0
            ce_loss = 0.0
            preds = None
            out_label_ids = None
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                with torch.no_grad():
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                    if args.model_type != "distilbert":
                        inputs["token_type_ids"] = (
                            batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                        )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                    outputs = model(**inputs, sampling_type="argmax")
                    tmp_eval_loss, logits = outputs["loss"]['loss'], outputs["logits"]
                    if 'ce_loss' in outputs["loss"]:
                        ce_loss += outputs["loss"]['ce_loss'].mean().item()
                    else:
                        ce_loss += tmp_eval_loss.mean().item()
                    eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            preds_label = np.argmax(preds, axis=1)
            metrics_result = compute_metrics(out_label_ids, preds_label)
            metrics_result['train_loss'] = tr_loss / train_step
            metrics_result['eval_loss'] = eval_loss
            metrics_result['epoch'] = epoch
            metrics_result['fold'] = n_split
            metrics_result['seed'] = seed
            metrics_result['ce_loss'] = ce_loss / nb_eval_steps
            logger.info(metrics_result)
            if results is None:
                results = pd.DataFrame(metrics_result, columns=metrics_result.keys(), index=[0])
            else:
                results = results.append(metrics_result, ignore_index=True)
        results.to_csv(args.output, index=False)
    return results


if __name__ == "__main__":
    args = get_args()

    results = None
    for seed in args.seeds:
        set_seed(seed)
        results = main(args, results, seed)
