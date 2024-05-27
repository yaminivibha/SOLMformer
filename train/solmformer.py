import argparse
import time 
import datetime
import os
import shutil
import math
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.data import random_split
from tqdm import tqdm
import numpy as np
from utils.Evaluator import Evaluator
from utils.BPIDataset import load_bpi_dataset
from models.model import SOLMformer

# Taking command line arguments
parser = argparse.ArgumentParser(description="Train a GPT model with or without metadata on the invoice or BPI dataset.")
parser.add_argument(
    "--dataset-name", 
    default="PrepaidTravelCost", 
    type=str, 
    help='name of the bpi dataset. "PrepaidTravelCost" or "InternationalDeclarations"'
)
parser.add_argument("--n-epochs", default=10, type=int, help="Number of epochs")
parser.add_argument("--logits-weight", default=0.33, type=float, help="Weight for the logits loss")
parser.add_argument("--durations-weight", default=0.33, type=float, help="Weight for the durations loss")

args = parser.parse_args()
dataset = args.dataset 
logits_weight = args.logits_weight
durations_weight = args.durations_weight
if dataset == "bpi" and args.dataset_name:
    dataset = args.dataset_name

# hyperparameters
n_epochs = args.n_epochs
if args.dataset_name in ["BPIC15_1"]:
    batch_size = 24
elif args.dataset_name in ["BPI2017"]:
    batch_size = 256
else:
    batch_size = 128
if args.dataset_name in ["2013_incidents", "BPIC15_1", "InternationalDeclarations"]:
    eval_interval = 20
elif args.dataset_name in ["BPI2012"]:
    eval_interval = 40
elif args.dataset_name in ["BPI2017"]:
    eval_interval = 95
else:
    eval_interval = 100
learning_rate = 1e-4
hidden_dim = 1024
duration_embd = 256
n_embd = 256
n_head = 4
n_layer = 2
dropout = 0.1
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
use_weights = False

model_dir = f"models/{dataset}_SOLMFORMER_logits_{logits_weight}_durations_{durations_weight}{current_time}"
os.mkdir(model_dir)
model_filename = f"{model_dir}/model.pt"

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

data = load_bpi_dataset(dataset)
# Define the sizes for train, validation, and test sets
train_size = int(0.8 * len(data))
val_size = int(0.1 * len(data))
test_size = len(data) - train_size - val_size

# Use random_split to split the dataset
torch.manual_seed(42)
train_dataset, val_dataset, test_dataset,  = random_split(data, [train_size, val_size, test_size])

# Create separate DataLoaders for each split
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if args.dataset_name in ["BPI2012", "2013_incidents","BPIC15_1"]:
    no_numerical_case_metadata=True
else:
    no_numerical_case_metadata=False

model = SOLMformer(
    numerical_metadata_dim=data.numerical_metadata_dim,
    duration_embedding_dim=duration_embd,
    hidden_dim=hidden_dim,
    activity_vocab_size=data.activity_vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=data.max_sequence_length+1,
    pad_code = data.pad_code,
    event_numerical_metadata_dim=data.event_numerical_metadata_dim,
    no_numerical_case_metadata=no_numerical_case_metadata,
)

model_params = {
    "numerical_metadata_dim": data.numerical_metadata_dim,
    "hidden_dim": hidden_dim,
    "activity_vocab_size": data.activity_vocab_size,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "block_size": data.max_sequence_length+1,
    "pad": data.pad_code,
    "terminal_code": data.terminal_code,
    "include_metadata": True,
    "device": device,
    "train_size": train_size,
    "val_size": val_size,
    "test_size": test_size,
    'dataset': dataset,
    'use_weights': use_weights,
    'duration_embd': duration_embd,
    'alpha': logits_weight,
    "event_numerical_metadata_dim":data.event_numerical_metadata_dim
}
torch.save(model_params, f"{model_dir}/model_params.pt")

print("=========================================")
print(f"Config:\n Dataset:{dataset}\n Use Weights: {use_weights}\n")
print(f"Device: {device}\nMax sequence length: {data.max_sequence_length}\nVocabulary size: {data.activity_vocab_size}")
print("Initialized model, parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, "M parameters ...")
print("# iterations per epoch:", len(train_dataloader))
print("=========================================")# Training block

model.to(device)
Q = math.floor(len(train_dataloader)/batch_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,  weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = Q)
def custom_loss(logits, targets, durations, durations_target, time_remaining, time_remaining_target,
                logits_weight=0.33, durations_weight = 0.33):
    time_remaining_weight = 1 - logits_weight - durations_weight
    # Mask out the padding tokens
    mask = torch.where(targets!=data.pad_code)
    logits = logits[mask]
    targets = targets[mask]
    durations = durations[mask]
    durations_target = durations_target[mask]
    time_remaining = time_remaining[mask]
    time_remaining_target = time_remaining_target[mask]

    logits_loss = F.cross_entropy(
        logits.reshape(-1, data.activity_vocab_size), targets.reshape(-1), ignore_index=data.pad_code
    )
    durations_loss = F.l1_loss(durations.squeeze(-1), durations_target,) 
    time_remaining_loss = F.l1_loss(time_remaining.squeeze(-1), time_remaining_target)
    combined_loss = (logits_loss * logits_weight) + (durations_loss * durations_weight) + (time_remaining_loss * time_remaining_weight)
    return combined_loss, logits_loss, durations_loss, time_remaining_loss

@torch.no_grad()
def estimate_loss(model, data_loader):
    model.eval()
    torch.no_grad()
    validation_loss = 0
    lloss = 0
    dloss = 0
    rloss = 0
    for batch in data_loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        # remove the metadata spacer, remove the last elemt (for input vs target)
        event_numerical_metadata_in = torch.log(batch['event_numerical_metadata'] + 0.000001)[:, :-1]
        x_in = batch['activities'][:, :-1]
        x_trgt = batch['activities'][:, 1:]
        durations_trgt = torch.log(batch['durations'] + 0.0000001)[:,1:]
        time_remaining_trgt = torch.log(batch['time_remaining'] + 0.0000001)[:,1:]
        logits, durations_pred, time_remaining_pred = model(x_in, batch['case_text_metadata'], batch['case_numerical_metadata'], event_numerical_metadata_in, batch['event_text_metadata'])
        logits = logits[:,1:]
        durations_pred = durations_pred[:,1:]
        time_remaining_pred = time_remaining_pred[:,1:]

        # Compute loss
        loss = custom_loss(logits, x_trgt, durations_pred, durations_trgt, time_remaining_pred, time_remaining_trgt, logits_weight, durations_weight)
        lloss += loss[1].item()
        dloss += loss[2].item()
        rloss += loss[3].item()
        validation_loss += loss[0].item()
    
    validation_loss /= len(val_dataloader)
    lloss /= len(val_dataloader)
    dloss /= len(val_dataloader)
    rloss /= len(val_dataloader)
    print("Validation loss {}, logits loss {}, durs loss {}, remaining loss {}".format(validation_loss, lloss, dloss, rloss))
    model.train()
    if device == "cuda":
        torch.cuda.empty_cache()
    torch.set_grad_enabled(True)
    return validation_loss
best_validation_loss = float("inf")
start = time.time()

training_loss = []
logits_loss = []
durations_loss = []
time_remaining_loss = []
learning_rates = []
for epoch in range(n_epochs):
    for i, batch in enumerate(tqdm(train_dataloader, desc="Batches", leave=False)):
        # xb, text, numbers, durations, event_text_metadata, event_numerical_metadata = xb.to(device), text.to(device), numbers.to(device), durations.to(device), event_text_metadata.to(device)
        # logits -> B x T x |A|
        # targets -> B x T
        batch = {key: value.to(device) for key, value in batch.items()}
        # remove the metadata spacer, remove the last elemt (for input vs target)
        event_numerical_metadata_in = torch.log(batch['event_numerical_metadata'] + 0.000001)[:, :-1]
        x_in = batch['activities'][:, :-1]
        x_trgt = batch['activities'][:, 1:]
        durations_trgt = torch.log(batch['durations'] + 0.0000001)[:,1:]
        time_remaining_trgt = torch.log(batch['time_remaining'] + 0.0000001)[:,1:]
        logits, durations_pred, time_remaining_pred = model(x_in, batch['case_text_metadata'], batch['case_numerical_metadata'], event_numerical_metadata_in, batch['event_text_metadata'])
        logits = logits[:,1:]
        durations_pred = durations_pred[:,1:]
        time_remaining_pred = time_remaining_pred[:,1:]

        if i % 10 == 0:
            probs = F.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)
        # Compute loss
        loss = custom_loss(logits, x_trgt, durations_pred, durations_trgt, time_remaining_pred, time_remaining_trgt, logits_weight, durations_weight)

        training_loss.append(loss[0].item())
        logits_loss.append(loss[1].item())
        durations_loss.append(loss[2].item())
        time_remaining_loss.append(loss[3].item())
        
        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss[0].backward()
        optimizer.step()
        
        # Validation
        if (i+1) % eval_interval == 0:
            # evaluate the model on the validation set
            validation_loss = estimate_loss(model, val_dataloader)
            print(f"Epoch {epoch+1}, iteration {i+1}: validation loss={validation_loss:.4f}")
            learning_rates.append(optimizer.param_groups[0]["lr"])
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        torch.save(model.state_dict(), model_filename)
        torch.save(optimizer.state_dict(), f"{model_dir}/optimizer.pt")
        print(f"Saved model to `{model_filename}`")
    plt.clf()
    plt.plot(training_loss, label="Total Loss")
    plt.plot(logits_loss, label="Logits Loss")
    plt.plot(durations_loss, label="Durations Loss")
    plt.plot(time_remaining_loss, label="Time Remaining Loss")
    plt.title(f'Training Loss, alpha={logits_weight}')
    plt.legend()
    plt.savefig(f"{model_dir}/loss.jpg")

print(f"Time elapsed: {time.time() - start:.2f} s.")

torch.save(model.state_dict(), model_filename)
print(f"Saved model to `{model_filename}`")
torch.save(optimizer.state_dict(), f"{model_dir}/optimizer.pt")
torch.save(scheduler.state_dict(), f"{model_dir}/scheduler.pt")
# Define the source files and the destination directory
source_files =['scripts/train.py','models/gpt.py','models/model.py']

for source_file in source_files:
    shutil.copy(source_file, model_dir)

new_model = SOLMformer(
    numerical_metadata_dim=model_params['numerical_metadata_dim'],
    duration_embedding_dim=model_params['duration_embd'],
    hidden_dim=model_params['hidden_dim'],
    activity_vocab_size=model_params['activity_vocab_size'],
    n_embd=model_params['n_embd'],
    n_head=model_params['n_head'],
    n_layer=model_params['n_layer'],
    block_size=model_params['block_size'],
    pad_code = data.pad_code,
    event_numerical_metadata_dim=model_params['event_numerical_metadata_dim'],
    no_numerical_case_metadata=no_numerical_case_metadata
)
new_model.load_state_dict(torch.load(model_filename))
model = new_model
model.to(device)
model.eval()

evaluator = Evaluator(dataset=dataset, model=model, device=device, data=data)
eval_results = evaluator.evaluate_solm()
plot_results = evaluator.plot_results(eval_results, model_dir)

result = '--------------------------------------------'
result += "Sequence Length\tNext act Accuracy\t# Samples \t MAE duration \t MAE Time remaining\n"
# result += "\n".join((f"{l}\t\t{acc}\t{n}\t{mse} \t {mse_remain}" for l, acc, n, mse, mse_remain in zip(initial_sequence_lengths, accuracies, nums_samples, MAEs, MAE_time_remaining)))
result += "overall next activity prediction accuracy:"
result += str(eval_results['next_activity_accuracy'])
result += "\n"
result += "Mean MAE of the durations: "
result += str(eval_results['next_act_dur_mae'])
result += "\n"
result += "Mean MAE of the time remaining: "
result += str(eval_results['time_remaining_mae'])
result += "\n"
result += "--------------------------------------------"
print(result)


with open(f"{model_dir}/results.txt", "w") as file:
    file.write(result)

torch.save(eval_results, f"{model_dir}/dict_results.pt")