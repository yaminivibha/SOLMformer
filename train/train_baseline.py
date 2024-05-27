import argparse
import time 
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm

from utils.BPIDataset import load_bpi_dataset
from models.model import MetadataSequenceModel
from models.gpt import GPTLanguageModel

# Taking command line arguments
parser = argparse.ArgumentParser(description="Train a GPT model with or without metadata on the invoice or BPI dataset.")
parser.add_argument("--include-metadata", action='store_true', help="Include metadata (default: False)")
parser.add_argument("--dataset", default="invoice", type=str, help="bpi or invoice, default: invoice")
parser.add_argument(
    "--dataset-name", 
    default="PrepaidTravelCost", 
    type=str, 
    help='name of the bpi dataset. "PrepaidTravelCost" or "InternationalDeclarations"'
)
parser.add_argument("--use-weights", action='store_true', help="Use weights for the loss function (default: False)")
parser.add_argument("--n-epochs", default=4, type=int, help="Number of epochs (default: 4)")
args = parser.parse_args()
dataset = args.dataset
include_metadata = True if args.include_metadata else False
use_weights = True if args.use_weights else False


# hyperparameters
n_epochs = args.n_epochs
if args.dataset_name in ["BPIC15_1"]:
    batch_size = 24
else:
    batch_size = 128
if args.dataset_name in ["2013_incidents", "BPIC15_1", "InternationalDeclarations"]:
    eval_interval = 10
elif args.dataset_name in [ "BPI2012"]:
    eval_interval = 30
else:
    eval_interval = 100
learning_rate = 1e-4
hidden_dim = 1024
duration_embd = 256
n_embd = 256
n_head = 4
n_layer = 2
dropout = 0.1
device="cpu"
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

model_dir = f"models/{args.dataset_name}_{'metadata' if include_metadata else 'mini'}{'_weights' if use_weights else ''}_gpt_{current_time}"
os.mkdir(model_dir)
model_filename = f"{model_dir}/model.pt"

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"

# Data setup
data = load_bpi_dataset(args.dataset_name)

# Define the sizes for train, validation, and test sets
train_size = int(0.8 * len(data))
val_size = int(0.1 * len(data))
test_size = len(data) - train_size - val_size

# Use random_split to split the dataset
torch.manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(data, [train_size, val_size, test_size])

# Create separate DataLoaders for each split
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if include_metadata:
# Initializing model
    model = MetadataSequenceModel(
        data.numerical_metadata_dim,
        hidden_dim, 
        data.activity_vocab_size, 
        n_embd, 
        n_head, 
        n_layer, 
        data.max_sequence_length+1 # plus one because the metadata is prepended as the first token
    )
    model.to(device)
else:
    model = GPTLanguageModel(
        data.activity_vocab_size,
        n_embd,
        n_head,
        n_layer,
        data.max_sequence_length+1
    )
    model.to(device)

if use_weights:
    if dataset == "bpi":
        weights = torch.load("data/log_counts_bpi.pt")
        weights = weights.to(device)
    else:
        weights = torch.load("data/log_counts_invoice.pt")
        weights = weights.to(device)

model_params = {
    "numerical_metadata_dim": data.numerical_metadata_dim if include_metadata else None,
    "hidden_dim": hidden_dim if include_metadata else None,
    "activity_vocab_size": data.activity_vocab_size,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "block_size": data.max_sequence_length+1,
    "pad": data.pad_code,
    "terminal_code": data.terminal_code,
    "include_metadata": include_metadata,
    "device": device,
    "train_size": train_size,
    "val_size": val_size,
    "test_size": test_size,
    'dataset': dataset,
    'use_weights': use_weights
}
torch.save(model_params, f"{model_dir}/model_params.pt")

print("=========================================")
print(f"Config:\n Dataset:{dataset}\n Include Metadata: {include_metadata}\n Use Weights: {use_weights}\n")
print(f"Device: {device}\nMax sequence length: {data.max_sequence_length}\nVocabulary size: {data.activity_vocab_size}")
print("Initialized model, parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, "M parameters ...")
print("# iterations per epoch:", len(train_dataloader))
print("=========================================")

# Training block
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss(model, data_loader):
    model.eval()
    validation_loss = 0
    for batch in data_loader:
        xb, text, numbers, d, _ = batch
        xb, text, numbers = xb.to(device), text.to(device), numbers.to(device)
        if include_metadata:
            logits = model(text, numbers, xb)
            logits = logits[:, :-1].contiguous()
            targets = xb.contiguous()
            if use_weights:
                loss = F.cross_entropy(
                    logits.view(-1, data.activity_vocab_size), targets.view(-1), ignore_index=data.pad_code, weight=weights
                )
            else:
                loss = F.cross_entropy(
                    logits.view(-1, data.activity_vocab_size), targets.view(-1), ignore_index=data.pad_code
                )
        else:
            logits = model(xb)
            logits = logits.contiguous()
            targets = xb.contiguous()
            if use_weights:
                loss = F.cross_entropy(
                    logits.view(-1, data.activity_vocab_size), targets.view(-1), ignore_index=data.pad_code, weight=weights
                )
            else:
                loss = F.cross_entropy(
                    logits.view(-1, data.activity_vocab_size), targets.view(-1), ignore_index=data.pad_code
                )
        validation_loss += loss.item()
    validation_loss /= len(val_dataloader)
    model.train()
    return validation_loss

start = time.time()

training_loss = []
best_validation_loss = float("inf")
for epoch in range(n_epochs):
    for i, batch in enumerate(train_dataloader):
        # sample a batch of data
        xb, text, numbers, d, _ = batch
        xb, text, numbers = xb.to(device), text.to(device), numbers.to(device)
        if include_metadata:
            logits = model(text, numbers, xb)
            logits = logits[:, :-1].contiguous()
            targets = xb.contiguous()
            if use_weights:
                loss = F.cross_entropy(logits.view(-1, data.activity_vocab_size), targets.view(-1), ignore_index=data.pad_code, weight=weights)
            else:
                loss = F.cross_entropy(logits.view(-1, data.activity_vocab_size), targets.view(-1), ignore_index=data.pad_code)
        else:
            # logits -> B x T x |A|
            # targets -> B x T
            x_in = xb[:, :-1]
            x_trgt = xb[:, 1:]
            logits = model(x_in)
            logits = logits.contiguous()
            targets = x_trgt.contiguous()
            if use_weights:
                loss = F.cross_entropy(
                    logits.view(-1, data.activity_vocab_size), targets.view(-1), ignore_index=data.pad_code, weight=weights
                )
            else:
                loss = F.cross_entropy(
                    logits.view(-1, data.activity_vocab_size), targets.view(-1), ignore_index=data.pad_code
                )
        training_loss.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (i+1) % eval_interval == 0:
            # evaluate the model on the validation set
            validation_loss = estimate_loss(model, val_dataloader)
            print(f"Epoch {epoch+1}, iteration {i+1}: validation loss={validation_loss:.4f}")
    if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), model_filename)
            torch.save(optimizer.state_dict(), f"{model_dir}/optimizer.pt")
            print(f"Saved model to `{model_filename}`")

print(f"Time elapsed: {time.time() - start:.2f} s.")

torch.save(model.state_dict(), model_filename)
torch.save(optimizer.state_dict(), f"{model_dir}/optimizer.pt")

print(f"Saved model to `{model_filename}`")

# Evaluation on test set
if include_metadata:
# Initializing model
    model = MetadataSequenceModel(
        data.numerical_metadata_dim,
        hidden_dim, 
        data.activity_vocab_size, 
        n_embd, 
        n_head, 
        n_layer, 
        data.max_sequence_length+1 # plus one because the metadata is prepended as the first token
    )
    model.to(device)
else:
    model = GPTLanguageModel(
        data.activity_vocab_size,
        n_embd,
        n_head,
        n_layer,
        data.max_sequence_length+1
    )
    model.to(device)

model.load_state_dict(torch.load(model_filename))
initial_sequence_lengths = np.arange(1, data.max_sequence_length - 2)
accuracies = []
nums_samples = []
total = 0
total_correct = 0
correct_next_activity_preds = {l: 0 for l in initial_sequence_lengths}
total_next_activity_preds = {l: 0 for l in initial_sequence_lengths}

with torch.no_grad():
    for initial_sequence_length in tqdm(initial_sequence_lengths):
        accuracy = 0
        n_test_samples = 0
        for batch in test_dataloader:
            xb, text, numbers, d, _ = batch
            xb, text, numbers = xb.to(device), text.to(device), numbers.to(device)
            cur_seq = xb[:, :initial_sequence_length]
            if include_metadata:
                logits = model(text, numbers, cur_seq)
                probs = F.softmax(logits[:,-1], dim=-1)
            else:
                logits = model(cur_seq)
                probs = F.softmax(logits[:,-1], dim=-1)
            
            pred = torch.argmax(probs, dim=-1)
            actual = xb[:, initial_sequence_length] # we predict the NEXT token
            mask = actual != data.pad_code
            accuracy += torch.sum(pred[mask] == actual[mask]).item()
            n_test_samples += pred[mask].shape[0]
            break
        if n_test_samples == 0:
            accuracies.append("N/A")
        else:
            accuracies.append(accuracy / n_test_samples)
        nums_samples.append(n_test_samples)
        total += n_test_samples
        total_correct += accuracy
final_accuracy = total_correct / total

result = '--------------------------------------------'
result += "Sequence Length\tNext act Accuracy\t# Samples "
result += "\n".join((f"{l}\t\t{acc}\t{n}" for l, acc, n,in zip(initial_sequence_lengths, accuracies, nums_samples)))
result += "\n"
result += "Mean Accuracy of next activity: "
result += str(final_accuracy)
result += "\n"
result += "--------------------------------------------"


plt.clf()

plt.plot(accuracies)
plt.title(f'Next Activity Accuracy')
plt.savefig(f"{model_dir}/accuracy.jpg")


plt.clf()

with open(f"{model_dir}/results.txt", "w") as file:
    file.write(result)

dict_results = {
    "accuracies": accuracies,
    "nums_samples": nums_samples,
    "initial_sequence_lengths": initial_sequence_lengths,
}

torch.save(dict_results, f"{model_dir}/dict_results.pt")