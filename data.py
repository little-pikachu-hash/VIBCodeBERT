import json
import copy
import csv
import os
import random
from os.path import join
from log import logger
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import defaultdict
import pandas as pd
import time


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None, guid=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.guid = guid

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def tfds_map(self, example):
        """Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are.
        This method converts examples to the correct format."""
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None, encoding="utf-8-sig"):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding=encoding) as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))


def convert_examples_to_features(
        examples,
        tokenizer,
        max_length=512,
        task=None,
        label_list=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        output_mode=None,
        mask_padding_with_zero=True,
        no_label=False,
        add_guid=False):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """

    def get_padded(input_ids, token_type_ids, attention_mask, max_length, pad_token,
                   pad_token_segment_id, mask_padding_with_zero):
        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )
        return input_ids, attention_mask, token_type_ids

    if task is not None:
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample):
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    features = []
    for (ex_index, example) in enumerate(examples):
        if example.text_b:
            inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True,
                                           max_length=max_length, )
        else:
            inputs = tokenizer.encode_plus(example.text_a, add_special_tokens=True, max_length=max_length, )
        input_ids = inputs["input_ids"]
        token_type_ids = [0] * len(input_ids)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_ids, attention_mask, token_type_ids = get_padded(input_ids, token_type_ids, \
                                                               attention_mask, max_length, pad_token,
                                                               pad_token_segment_id, mask_padding_with_zero)

        label = label_from_example(example) if not no_label else -1
        if add_guid:
            features.append(
                InputFeatures(
                    input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label,
                    guid=example.guid
                )
            )
        else:
            features.append(
                InputFeatures(
                    input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
                )
            )
    return features


class YelpProcessor(DataProcessor):
    """Processor for the Yelp dataset."""

    def __init__(self):
        self.num_classes = 5

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_validation_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            label = line[0]
            text_a = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class PojProcessor(DataProcessor):
    """Processor for the Yelp dataset."""

    def __init__(self, data_dir, per=1000000):
        data = json.load(open(data_dir + 'train.jsonl'))
        self.labels = list(set([i['label'] for i in data]))
        self.num_classes = len(self.labels)
        self.per = per

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "test")

    def get_validation_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "val")

    def get_labels(self):
        """See base class."""
        return self.labels

    def _create_examples(self, data_dir, set_type):
        """Creates examples for the training and dev sets."""
        logger.info(data_dir + '{}.jsonl'.format(set_type))
        data = json.load(open(data_dir + '{}.jsonl'.format(set_type)))
        counter = defaultdict(int)
        d = []
        if set_type == 'train':
            random.shuffle(data)
            for item in data:
                if counter[item['label']] < self.per:
                    d.append(item)
                    counter[item['label']] += 1
            data = d
        examples = []
        for (i, example) in enumerate(data):
            guid = i
            examples.append(
                InputExample(guid=guid, text_a=example['code'], text_b=None, label=example['label']))
        return examples


class BuggyProcessor(DataProcessor):

    def __init__(self, data_dir):
        # We assume there is a training file there and we read labels from there.
        test_df = pd.read_csv(data_dir + 'test.csv')
        labels = test_df.label.values
        self.labels = list(set(labels))
        self.num_classes = len(self.labels)
        print(self.labels)

    def get_dev_labels(self, data_dir):
        labels = pd.read_csv(data_dir + 'test.csv').label.values
        return np.array(labels)

    def get_validation_examples(self, data_dir):
        return self._create_examples(data_dir, "test")

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "test")

    def get_labels(self):
        """See base class."""
        return self.labels

    def _create_examples(self, data_dir, set_type):
        """Creates examples for the training and dev sets."""
        logger.info(data_dir + '{}.csv'.format(set_type))
        df = pd.read_csv(data_dir + '{}.csv'.format(set_type))
        examples = []
        for index, row in df.iterrows():
            guid = int(row['id'])
            examples.append(
                InputExample(guid=guid, text_a=row['code'], label=row['label']))
        return examples


class SmellProcessor(object):
    def __init__(self, data_dir):
        # We assume there is a training file there and we read labels from there.
        self.data = json.load(open(data_dir + 'train_validate.jsonl'))
        labels = [item['label'] for item in self.data]
        self.labels = list(set(labels))
        self.num_classes = len(self.labels)

    def get_examples(self):
        examples = []
        for (i, example) in enumerate(self.data):
            guid = "%s-%s" % ('data', i)
            examples.append(
                InputExample(guid=guid, text_a=example['code'], label=example['label']))
        return examples

    def get_labels(self):
        """See base class."""
        return self.labels


class ComplexityProcessor(object):
    def __init__(self, data_dir):
        # We assume there is a training file there and we read labels from there.
        data = json.load(open(data_dir + 'train.jsonl'))
        # random.shuffle(data)
        labels = [item['label'] for item in data]
        self.labels = list(set(labels))
        self.num_classes = len(self.labels)

    def get_dev_labels(self, data_dir):
        examples = self._create_examples(data_dir, "dev")
        return np.array([item.label for item in examples])

    def get_validation_examples(self, data_dir):
        return self._create_examples(data_dir, "test")

    def get_train_examples(self, data_dir, shot=-1):
        """See base class."""
        return self._create_examples(data_dir, "train", shot)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "test")

    def get_labels(self):
        """See base class."""
        return self.labels

    def _create_examples(self, data_dir, set_type, shot=-1):
        """Creates examples for the training and dev sets."""
        data = json.load(open(data_dir + '{}.jsonl'.format(set_type)))
        examples = []
        for (i, example) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            examples.append(
                InputExample(guid=guid, text_a=example['code'], label=example['label']))
        return examples

class SmellProcessor2(object):
    def __init__(self, data_dir):
        # We assume there is a training file there and we read labels from there.
        data = json.load(open(data_dir + 'train_validate.jsonl'))
        # random.shuffle(data)
        labels = [item['label'] for item in data]
        self.data = data
        self.labels = list(set(labels))
        self.num_classes = len(self.labels)

    def get_dev_labels(self, data_dir):
        examples = self._create_examples(data_dir, "dev")
        return np.array([item.label for item in examples])

    def get_validation_examples(self, data_dir):
        return self._create_examples(data_dir, "test")

    def get_train_examples(self, data_dir, shot=-1):
        """See base class."""
        return self._create_examples(data_dir, "train", shot)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "test")

    def get_labels(self):
        """See base class."""
        return self.labels

    def _create_examples(self, data_dir, set_type, shot=-1):
        """Creates examples for the training and dev sets."""
        label_data = {i: [] for i in self.labels}
        for item in self.data:
            label_data[item['label']].append(item)

        data = []
        if set_type == 'test':
            for key in label_data:
                d = label_data[key]
                data += d[int(len(d) * 0.80):]
        elif shot != -1:
            for key in label_data:
                d = label_data[key]
                data += d[:shot]
        else:
            for key in label_data:
                d = label_data[key]
                data += d[:int(len(d) * 0.8)]

        examples = []
        for (i, example) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            examples.append(
                InputExample(guid=guid, text_a=example['code'], label=example['label']))
        return examples


class CoherenceProcessor(object):

    def __init__(self, data_dir):
        # We assume there is a training file there and we read labels from there.
        self.data = json.load(open(data_dir + 'train.jsonl'))
        labels = [item['label'] for item in self.data]
        self.labels = list(set(labels))
        self.num_classes = len(self.labels)

    def get_dev_labels(self, data_dir):
        examples = self._create_examples(data_dir, "dev")
        return np.array([item.label for item in examples])

    def get_validation_examples(self, data_dir):
        return self._create_examples(data_dir, "test")

    def get_train_examples(self, data_dir, shot=-1):
        """See base class."""
        return self._create_examples(data_dir, "train", shot)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "test")

    def get_labels(self):
        """See base class."""
        return self.labels

    def _create_examples(self, data_dir, set_type, shot=-1):
        """Creates examples for the training and dev sets."""
        logger.info(data_dir + '{}.jsonl'.format(set_type))
        data = json.load(open(data_dir + '{}.jsonl'.format(set_type)))
        random.shuffle(data)
        counter = defaultdict(int)
        d = []
        if set_type == 'train':
            for item in data:
                if counter[item['label']] < shot or shot == -1:
                    d.append(item)
                    counter[item['label']] += 1
            data = d
        examples = []
        for (i, example) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            examples.append(
                InputExample(guid=guid, text_a=example['comment'], text_b=example['code'], label=example['label']))
        return examples


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def all_metric(preds, labels, num_classes):
    acc = simple_accuracy(preds, labels)
    average = 'binary'
    if num_classes > 2:
        average = 'macro'
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    precision = precision_score(labels, preds, average=average)
    recall = recall_score(labels, preds, average=average)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
        "precision": precision,
        "recall": recall,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def glue_compute_metrics(task_name, preds, labels, num_classes):
    assert len(preds) == len(labels)
    if task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name in ["mnli", "mnli-mm", "rte", "snli", \
                       "addonerte", "dpr", "spr", "fnplus", "joci", "mpe", \
                       "scitail", "sick", "QQP", "snlihard", "imdb", "yelp"]:
        return {"acc": simple_accuracy(preds, labels)}
    else:
        return all_metric(preds, labels, num_classes)


processors = {
    "yelp": YelpProcessor,
    "poj": PojProcessor,
    "py800": PojProcessor,
    "py20": PojProcessor,
    "jv250": PojProcessor,
    "smell": SmellProcessor,
    "smell2": SmellProcessor2,
    "read2": SmellProcessor2,
    "complexity2": SmellProcessor2,
    "py100": PojProcessor,
    "jv100": PojProcessor,
    "jv50": PojProcessor,
    "complexity": SmellProcessor,
    "coherence": CoherenceProcessor,
    "buggy": BuggyProcessor,
    "sbabi": BuggyProcessor,
    "read": SmellProcessor,
    "juliet": BuggyProcessor,
}

output_modes = {
    "sts-b": "regression",
    "poj": "classification",
    "py800": "classification",
    "py20": "classification",
    "jv250": "classification",
    "smell": "classification",
    "smell2": "classification",
    "complexity2": "classification",
    "read2": "classification",
    "py100": "classification",
    "jv100": "classification",
    "jv50": "classification",
    "complexity": "classification",
    "coherence": "classification",
    "buggy": "classification",
    "sbabi": "classification",
    "read": "classification",
    "juliet": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "yelp": 5,
    "poj": 104,
    "py800": 800,
    "py20": 20,
    "jv250": 250,
    "smell": 2,
    "smell2": 2,
    'py100': 100,
    'jv100': 100,
    'jv50': 50,
    'complexity': 5,
    'complexity2': 5,
    'coherence': 2,
    'buggy': 2,
    'read': 2,
    'read2': 2,
    'sbabi': 2,
    'juliet': 2,
}
