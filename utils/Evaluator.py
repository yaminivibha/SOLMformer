import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from matplotlib import pyplot as plt
class Evaluator():
    def __init__(self, dataset:torch.utils.data.Dataset, test_dataloader:torch.utils.data.Dataloader, model:torch.nn.Module, device:str,) -> None:
        """
        Evaluator class for evaluating the performance of the SOLM model & baselines

        Parameters:
        dataset (torch.utils.data.Dataset): The dataset object
        test_dataloader (torch.utils.data.Dataloader): The test dataloader
        model (torch.nn.Module): The model to be evaluated
        device (str): The device to run the evaluation on

        Returns:
        None
        """
        
        self.data = dataset
        self.dataloader = test_dataloader
        self.model = model
        self.device = device
        

    def evaluate_solm(self):
        """
        Purpose: Produce evaluation metrics for SOLM model
        across all 3 tasks
        1. Next activity prediction accuracy
        2. Next Duration prediction MAE
        3. Time remaining prediction MAE

        Parameters:
        None
        Returns:
        dict: Dictionary containing the evaluation metrics
        """
        # Evaluation on test set
        # integers from 1 to data.max_sequence_length - 2 
        initial_sequence_lengths = np.arange(2, self.data.max_sequence_length - 2)
        accuracies = []
        nums_samples = []
        MAEs = []
        MAE_time_remaining = []
        predicted_activities = []
        actual_activities = []
        with torch.no_grad():
            correct_next_activity_preds = {l: 0 for l in initial_sequence_lengths}
            total_next_activity_preds = {l: 0 for l in initial_sequence_lengths}
            predicted_durations_dict = {l: [] for l in initial_sequence_lengths}
            actual_durations_dict = {l: [] for l in initial_sequence_lengths}
            predicted_time_remaining_dict = {l: [] for l in initial_sequence_lengths}
            actual_time_remaining_dict = {l: [] for l in initial_sequence_lengths}
            predicted_next_activities = {l: [] for l in initial_sequence_lengths}
            actual_next_activities = {l: [] for l in initial_sequence_lengths}
            for i, batch in enumerate(tqdm(self.dataloader)):
                batch = {key: value.to(self.device) for key, value in batch.items()}
                # remove the metadata spacer, remove the last elemt (for input vs target)
                event_numerical_metadata_in = torch.log(batch['event_numerical_metadata'] + 0.000001)[:, :-1]
                x_in = batch['activities'][:, :-1]
                x_trgt = batch['activities'][:, 1:]
                durations_trgt = torch.log(batch['durations'] + 0.0000001)[:,1:]
                time_remaining_trgt = torch.log(batch['time_remaining'] + 0.0000001)[:,1:]
                logits, durations_pred, time_remaining_pred = self.model(x_in, batch['case_text_metadata'], batch['case_numerical_metadata'], event_numerical_metadata_in, batch['event_text_metadata'])
                logits = logits[:,1:]
                durations_pred = durations_pred[:,1:]
                time_remaining_pred = time_remaining_pred[:,1:]
                # Predictions at each point in time:
                probs = F.softmax(logits, dim=-1)
                pred = torch.argmax(probs, dim=-1)

                viable_initial_seq_lens = [i for i in initial_sequence_lengths if i < x_in.shape[1] - 2 and i>0]
                # Calculate accuracy at each initiql sequence length
                for initial_sequence_length in viable_initial_seq_lens:
                    # Cut off predictions at the initial sequence length
                    predicted_activity = pred[:, initial_sequence_length]
                    actual_activity = x_trgt[:, initial_sequence_length]
                    # Saving accuracy
                    mask = ((actual_activity != self.data.pad_code))
                    result = (predicted_activity == actual_activity) & mask
                    correct_next_activity_preds[initial_sequence_length] += result.sum().item()
                    total_next_activity_preds[initial_sequence_length] += (actual_activity != self.data.pad_code).sum().item()

                    # Cut off duration predictions at the initial sequence length
                    mask = ((actual_activity != self.data.pad_code) & (actual_activity != self.data.terminal_code)) 
                    predicted_duration = torch.masked_select(durations_pred[:, initial_sequence_length].squeeze(-1), mask)
                    actual_duration = torch.masked_select(durations_trgt[:, initial_sequence_length], mask)

                    # Save the predicted and actual durations
                    predicted_durations_dict[initial_sequence_length].extend(predicted_duration.tolist())
                    actual_durations_dict[initial_sequence_length].extend(actual_duration.tolist())

                    predicted_time_remaining = torch.masked_select(time_remaining_pred[:, initial_sequence_length].squeeze(-1), mask)
                    actual_time_remaining = torch.masked_select(time_remaining_trgt[:, initial_sequence_length], mask)
                    assert predicted_time_remaining.shape == actual_time_remaining.shape

                    predicted_time_remaining_dict[initial_sequence_length].extend(predicted_time_remaining.tolist())
                    actual_time_remaining_dict[initial_sequence_length].extend(actual_time_remaining.tolist())   
                
            # Calculate accuracy at each initial sequence length
            for initial_sequence_length in initial_sequence_lengths:
                if total_next_activity_preds[initial_sequence_length] == 0:
                    accuracies.append("N/A")
                else:
                    accuracies.append(correct_next_activity_preds[initial_sequence_length] / total_next_activity_preds[initial_sequence_length])
                
                # Calculate the mean absolute error for the durations
                predicted_durations_dict[initial_sequence_length] = torch.tensor(predicted_durations_dict[initial_sequence_length])
                actual_durations_dict[initial_sequence_length] = torch.tensor(actual_durations_dict[initial_sequence_length])
                
                # Scale back up the durations from log scale
                predicted_durations_dict[initial_sequence_length] = torch.exp(predicted_durations_dict[initial_sequence_length]) - 0.0000001
                actual_durations_dict[initial_sequence_length] = torch.exp(actual_durations_dict[initial_sequence_length]) - 0.0000001
                mae_durations = F.l1_loss(predicted_durations_dict[initial_sequence_length], actual_durations_dict[initial_sequence_length])

                predicted_time_remaining_dict[initial_sequence_length] = torch.tensor(predicted_time_remaining_dict[initial_sequence_length])
                actual_time_remaining_dict[initial_sequence_length] = torch.tensor(actual_time_remaining_dict[initial_sequence_length])
                # Scale back up the durations from log scale
                predicted_time_remaining_dict[initial_sequence_length] = torch.exp(predicted_time_remaining_dict[initial_sequence_length]) - 0.0000001
                actual_time_remaining_dict[initial_sequence_length] = torch.exp(actual_time_remaining_dict[initial_sequence_length]) - 0.0000001
                mae_time_remaining = F.l1_loss(predicted_time_remaining_dict[initial_sequence_length], actual_time_remaining_dict[initial_sequence_length])
                
                MAE_time_remaining.append(mae_time_remaining.item())
                MAEs.append(mae_durations.item())
                nums_samples.append(total_next_activity_preds[initial_sequence_length])

        def flatten(d: dict):
            out = []
            for key in initial_sequence_lengths:
                if key in d and torch.numel(d[key]) > 0:  # Check if key exists and list is not empty
                    out.extend(d[key])
            return out

        # Combining the results to compute averages more fairly
        total_next_activity = sum([total_next_activity_preds[l] for l in initial_sequence_lengths]   )
        total_next_activity_correct = sum([correct_next_activity_preds[l] for l in initial_sequence_lengths]   )   

        predicted_time_remaining = flatten(predicted_time_remaining_dict)
        actual_time_remaining = flatten(actual_time_remaining_dict)

        return {"next_act_accuracy": total_next_activity_correct / total_next_activity,
                "next_act_dur_mae": sum([e for e in MAEs if not np.isnan(e)])/len([1 for e in MAEs if not np.isnan(e)]),
                "time_remaining_mae":sum([e for e in MAE_time_remaining if not np.isnan(e)])/len([1 for e in MAE_time_remaining if not np.isnan(e)]),
                "MAEs": MAEs,
                "MAE_time_remaining": MAE_time_remaining,
                "nums_samples": nums_samples,
                "accuracies": accuracies,                                                             
                }
    
    def evaluate_baseline():
        pass
    def plot_results(eval_results:dict, model_dir:str):
        # Plotting results
        plt.clf()
        plt.plot(eval_results['MAEs'])
        plt.title(f'MAE Duration')
        plt.savefig(f"{model_dir}/mae.jpg")

        plt.clf()

        plt.plot(eval_results['accuracies'])
        plt.title(f'Next Activity Accuracy')
        plt.savefig(f"{model_dir}/accuracy.jpg")

        plt.clf()

        plt.plot(eval_results['MAE_time_remaining'])
        plt.title(f'MAE Time Remaining')
        plt.savefig(f"{model_dir}/mae_time_remaining.jpg")

        plt.clf()

    

