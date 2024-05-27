
from lxml import etree
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
from datetime import datetime
import logging 
import pandas as pd

logging.basicConfig(level=logging.INFO) 
dirname = os.path.dirname(__file__)
logger = logging.getLogger(__name__)

def _get_node_localname(node) -> str:
    return etree.QName(node).localname


def get_stats(bpi_raw_path:str):
    tree = etree.parse(bpi_raw_path)
    root = tree.getroot()
    act_lens = []
    events = defaultdict(int)
    ids = defaultdict(int)
    case_attribs = defaultdict(int)
    for trace in root:
        if not _get_node_localname(trace) == 'trace':
            continue
        # Count Rfp_id
        for string in trace:
            if _get_node_localname(string) != 'string':
                continue
            if string.attrib['key'] == 'concept:name':
                ids[string.attrib['value']] += 1

        # Check if each case contains the same attributes
        for node in trace:
            if _get_node_localname(node) != 'event':
                case_attribs[node.attrib['key']] += 1

        # Count events
        act_len = 0
        for event in trace:
            if _get_node_localname(event) != 'event':
                continue
            act_len += 1
            for string in event:
                if _get_node_localname(string) != 'string':
                    continue
                if string.attrib['key'] == 'concept:name':
                    events[string.attrib['value']] += 1
        act_lens.append(act_len)
    assert max(ids.values()) == 1
    for attrib, cnt in case_attribs.items():
        if cnt != len(ids):
            print(f"Not all cases have attribute '{attrib}'.")
    return max(act_lens), list(events.keys())

def get_trace_start_end_time(trace):
    start_time = None
    end_time = None
    for trace_metadata in trace:
        if _get_node_localname(trace_metadata) == 'event':
            for event in trace_metadata:
                if _get_node_localname(event) == 'date' and event.attrib['key'] == 'time:timestamp':
                        ts = datetime.fromisoformat(event.attrib['value'].replace("Z", "+00:00"))
                        if not ts:
                            continue
                        if start_time is None:
                            start_time = ts
                        end_time = ts
    return start_time, end_time

def bpi_dataset_preprocessing(bpi_raw_path:str, bpi_path:str):
    tree = etree.parse(bpi_raw_path)
    root = tree.getroot()

    max_trace_len, event_categories = get_stats(bpi_raw_path)

    event_tokenization = {event: i for i, event in enumerate(event_categories)}
    TERMINATION, PAD = len(event_categories), len(event_categories) + 1

    text_metadata, numerical_metadata, activities, event_text_metadata, event_durations, event_time_since_start, event_recent_times, event_time_until_end= [], [], [], [], [], [], [], []

    for trace in root:
        start_time, end_time = get_trace_start_end_time(trace)
        if not start_time and end_time:
            continue
        if not _get_node_localname(trace) == 'trace':
            continue 
        categorical_values = {}
        for node in trace:
            if not _get_node_localname(node) == 'string':
                continue
            categorical_values[node.attrib['key']] = node.attrib['value']
        for node in trace:
            if not _get_node_localname(node) == 'int':
                continue
            categorical_values[node.attrib['key']] = str(node.attrib['value'])
        numerical_values = {}
        for node in trace:
            if not _get_node_localname(node) == 'float':
                continue
            numerical_values[node.attrib['key']] = float(node.attrib['value'])

        acts, event_text, durations, time_remaining, time_since_starts, recent_times = [], [], [], [], [], []

        ts = None
        second_to_last_ts = None
        dates_seen = 0
        for event in trace:
            if not _get_node_localname(event) == 'event':
                continue
            # get event name and text metadata
            event_categorical_values = {}
            for node in event:
                if _get_node_localname(node) == 'string':
                    if node.attrib['key'] == 'concept:name':
                        acts.append(event_tokenization[node.attrib['value']])
                        continue
                    event_categorical_values[node.attrib['key']] = node.attrib['value']
                elif _get_node_localname(node) == 'boolean':
                    event_categorical_values[node.attrib['key']] = node.attrib['value']
                elif _get_node_localname(node) == 'date' and node.attrib['key'] == 'time:timestamp':
                    new_ts = datetime.fromisoformat(node.attrib['value'].replace("Z", "+00:00"))
                    if not new_ts:
                        continue
                    if dates_seen == 0:
                        second_to_last_ts = new_ts
                    time_remaining.append((end_time - new_ts).total_seconds() / 86400)
                    time_since_starts.append((new_ts - start_time).total_seconds() / 86400)
                    if ts is not None:
                        durations.append((new_ts - ts).total_seconds() / 86400)
                    if dates_seen > 1:
                        recent_times.append((new_ts - second_to_last_ts).total_seconds() / 86400)
                    second_to_last_ts = ts       
                    ts = new_ts
                    dates_seen += 1
                elif _get_node_localname(node) == 'float': # Add
                    event_categorical_values[node.attrib['key']] = node.attrib['value']

            event_text.append(', '.join([f'{k}:{v}' for k, v in event_categorical_values.items()]))

        assert len(acts) == len(time_since_starts) == len(time_remaining)
        assert len(durations) == len(acts) - 1
        assert len(recent_times) == max(len(acts) - 2, 0)

        acts.append(TERMINATION)
        acts += [PAD] * (max_trace_len + 1 - len(acts))
        assert len(acts) == max_trace_len + 1

        durations += [0] * (max_trace_len + 1 - len(durations))
        assert len(durations) == len(acts)

        recent_times = [0,0] + recent_times + [0] * (max_trace_len - len(recent_times) - 1)
        assert len(recent_times) == len(acts)

        time_remaining += [0] * (max_trace_len + 1 - len(time_remaining))
        time_since_starts += [0] * (max_trace_len + 1 - len(time_since_starts))
        assert(len(time_remaining) == len(time_since_starts) == len(acts))

        event_text += [''] * (max_trace_len + 1 - len(event_text))
        assert len(event_text) == len(acts)

        text_metadata.append(', '.join([f'{k}:{v}' for k, v in categorical_values.items()]))
        numerical_metadata.append(numerical_values)
        activities.append(acts)
        event_text_metadata.append(event_text)
        event_durations.append(durations)
        event_time_since_start.append(time_since_starts)
        event_recent_times.append(recent_times)
        event_time_until_end.append(time_remaining)
    numerical_metadata = pd.DataFrame(numerical_metadata).to_numpy(dtype='float32')

    assert len(text_metadata) == len(numerical_metadata) == len(activities)
    assert len(event_text_metadata) == len(event_durations) == len(activities) == len(event_time_since_start) == len(event_recent_times) == len(event_time_until_end)

    max_event_text_length = max([len(text) for text in event_text for event_text in event_text_metadata])

    dataset = {
        'text_metadata': text_metadata,
        'numerical_metadata': torch.tensor(numerical_metadata),
        'activities': torch.tensor(activities),
        'durations': torch.tensor(event_durations),
        'time_remaining': torch.tensor(event_time_until_end),
        'recent_times': torch.tensor(event_recent_times),
        'times_since_start': torch.tensor(event_time_since_start),
        'event_text_metadata': event_text_metadata,
        'activity_categories': event_categories,
        'TERMINATION_CODE': TERMINATION, 'PAD_CODE': PAD,
        'max_sequence_length': max_trace_len + 1,
        'max_event_text_length': max_event_text_length,
    }
    dataset['event_numerical_metadata'] = torch.cat(
        [dataset['durations'].unsqueeze(-1),
        dataset['recent_times'].unsqueeze(-1),
        dataset['times_since_start'].unsqueeze(-1)],
        dim=2
    )
	# Adding an extra space for the metadata token.
    batch_size, _, channels = dataset['event_numerical_metadata'].shape
    # Create a tensor of zeros to add as the first row
    zeros_row = torch.zeros((batch_size, 1, channels))

    dataset['event_numerical_metadata_spaced'] = torch.cat((zeros_row, dataset['event_numerical_metadata']), dim=1)

    torch.save(dataset, bpi_path)


class BPIDataset(Dataset):
    def __init__(self, path: str, case_metadata_space:bool=False, total_duration:bool=False):
        self.tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
        self.dataset = torch.load(path)
        self.case_metadata_space = case_metadata_space
        self.total_duration = total_duration
        assert len(self.dataset['text_metadata']) == len(self.dataset['numerical_metadata']) == len(self.dataset['activities'])
        if not 'event_numerical_metadata_spaced' in self.dataset:
            # Adding an extra space for the metadata token.
            batch_size, _, channels = self.dataset['event_numerical_metadata'].shape
            # Create a tensor of zeros to add as the first row
            zeros_row = torch.zeros((batch_size, 1, channels))
            self.dataset['event_numerical_metadata_spaced'] = torch.cat((zeros_row, self.dataset['event_numerical_metadata']), dim=1)

        # Setting instance variables
        self.numerical_metadata_dim = self.dataset["numerical_metadata"].shape[1]
        self.embedding_dim = self.dataset["activities"].shape[1]
        self.activity_vocab_size = len(self.dataset["activity_categories"]) + 2
        self.max_sequence_length = self.dataset["max_sequence_length"]
        self.pad_code = self.dataset["PAD_CODE"]
        self.terminal_code = self.dataset['TERMINATION_CODE']
        
        # Get the longest text metadata length
        longest_text = max(self.dataset["text_metadata"], key=len)
        self.padding_size = len(self.tokenizer(longest_text)["input_ids"])

        # Calculate the longest length of the tokenized text metadata
        longest_event_text = max(
            [text for event_text in self.dataset["event_text_metadata"] for text in event_text], key=len
        )
        self.event_padding_size = len(self.tokenizer(longest_event_text)["input_ids"])
        self.event_numerical_metadata_dim = self.dataset["event_numerical_metadata"].shape[2]

    def __len__(self):
        return len(self.dataset["text_metadata"])
    
    def __getitem__(self, idx):
        # activities, text_metadata, numerical_metadata, durations, event_text_metadata
        # where event_text_metadata is a 2-d vector of tokens
        if self.total_duration:
            return dict(
                activities=self.dataset["activities"][idx],
                case_text_metadata=self.tokenizer(
                    self.dataset["text_metadata"][idx], 
                    return_tensors='pt', 
                    padding='max_length', 
                    max_length=self.padding_size, 
                    truncation=True
                ),
                case_numerical_metadata=self.dataset["numerical_metadata"][idx], 
                event_numerical_metadata=self.dataset["event_numerical_metadata_spaced"][idx],
                event_text_metadata=self.tokenizer(
                    # Because the input data is always cut off by 1 less
                    # Due to inputs vs targets 
                    [""] + self.dataset["event_text_metadata"][idx][:-1], 
                    return_tensors='pt', 
                    padding='max_length', 
                    max_length=self.event_padding_size, 
                    truncation=True
                ),
                time_remaining=self.dataset['time_remaining'][idx],
                durations = self.dataset['durations'][idx],

            )
        if self.case_metadata_space:
            return (
                self.dataset["activities"][idx],
                self.tokenizer(
                    self.dataset["text_metadata"][idx], 
                    return_tensors='pt', 
                    padding='max_length', 
                    max_length=self.padding_size, 
                    truncation=True
                ),
                self.dataset["numerical_metadata"][idx], 
                torch.cat((torch.tensor([0]),self.dataset["durations"][idx])),
                self.tokenizer(
                    [""] + self.dataset["event_text_metadata"][idx], 
                    return_tensors='pt', 
                    padding='max_length', 
                    max_length=self.event_padding_size, 
                    truncation=True
                )
            )
        else:
            return (
                self.dataset["activities"][idx],
                self.tokenizer(
                    self.dataset["text_metadata"][idx], 
                    return_tensors='pt', 
                    padding='max_length', 
                    max_length=self.padding_size, 
                    truncation=True
                ),
                self.dataset["numerical_metadata"][idx], 
                self.dataset["durations"][idx],
                self.tokenizer(
                    self.dataset["event_text_metadata"][idx], 
                    return_tensors='pt', 
                    padding='max_length', 
                    max_length=self.event_padding_size, 
                    truncation=True
                )
            )

def load_bpi_dataset(bpi_dataset_name: str, parent_dir: str="", case_metadata_space:bool=False, total_duration:bool=False):
    """Load preprocessed BPI dataset. If not exist, process BPI dataset and load it.

    Args:
        bpi_dataset_name (str): Name of the BPI dataset. "PrepaidTravelCost" or "InternationalDeclarations".
        parent_dir (str): Parent directory of the dataset. Default is "".
        case_metadata_space (bool): Whether to include the case metadata space. Default is False.
    """
    if parent_dir:
        bpi_raw_path = os.path.join(parent_dir, f'data/{bpi_dataset_name}.xes')
        bpi_path = os.path.join(parent_dir, f'data/bpi_{bpi_dataset_name}.pt')
    else:
        bpi_raw_path = os.path.join(dirname, f'../data/{bpi_dataset_name}.xes') 
        bpi_path = os.path.join(dirname, f'../data/bpi_{bpi_dataset_name}.pt')
    if not os.path.exists(bpi_path):
        bpi_dataset_preprocessing(bpi_raw_path, bpi_path)
    assert os.path.exists(bpi_path)
    return BPIDataset(bpi_path, case_metadata_space, total_duration)
