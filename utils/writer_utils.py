from datetime import datetime
import os

from torch.utils.tensorboard.writer import SummaryWriter

def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str,
                  workspace: str) -> SummaryWriter:
    """
    Creates a SummaryWriter object for logging experiment data.
    
    Parameters:
    experiment_name (str): The name of the experiment.
    model_name (str): The name of the model.
    extra (str): Additional information for the log directory path (optional).
    workspace (str): The path to the workspace directory.
    
    Returns:
    SummaryWriter: The SummaryWriter object for logging experiment data.
    """
    
    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H") # returns current date in YYYY-MM-DD format
    print(experiment_name, model_name, extra, workspace)
    if extra:
        # Create log directory path
        log_dir = os.path.join(workspace, experiment_name,timestamp, model_name, extra)
    else:
        log_dir = os.path.join(workspace, experiment_name,timestamp, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

def create_summary_writer(log_dir):
    print(log_dir)
    return SummaryWriter(log_dir=log_dir)
