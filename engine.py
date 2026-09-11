from tqdm.auto import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

# This function from learnpytorch.py
def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str=None):
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    from datetime import datetime
    import os

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

def train_step(model,train_dataloader:torch.utils.data.DataLoader,loss_fn,optimizer,device:str='cpu'):
    correct=0
    loss_sum=0
    for batch, (X,y) in enumerate(train_dataloader):
        X,y=X.to(device),y.to(device)
        y_pred=model(X)
        loss=loss_fn(y_pred,y)
        loss_sum+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred=y_pred.argmax(dim=1)
        correct+=(y_pred==y).sum().item()
    return correct,(loss_sum/len(train_dataloader))

def test_step(model,test_dataloader:torch.utils.data.DataLoader,loss_fn,device:str):
    correct=0
    loss_sum=0
    for batch, (X,y) in enumerate(test_dataloader):
        X,y=X.to(device),y.to(device)
        y_pred=model(X)
        loss=loss_fn(y_pred,y)
        y_pred=y_pred.argmax(dim=1)
        correct+=(y_pred==y).sum().item() 
        loss_sum+=loss.item()  
        
    return correct,(loss_sum/len(test_dataloader))

def train(epochs:int,model,train_dataloader:torch.utils.data.DataLoader,test_dataloader:torch.utils.data.DataLoader,loss_fn,optimizer,device:str,writer: SummaryWriter = None,verbose:bool=False,granularity:int=1, ):
    """Function to train model

    Args:
        epochs (int): How many times the model loops through data in training
        model (nn.Module): ML model to be trained
        train_dataloader (Pytorch Dataloader): Dataloader obj for training
        test_dataloader (Pytorch Dataloader): Dataloader obj for testing
        loss_fn (nn loss function): Function used for calculating model loss
        optimizer (nn.optim): Function used for performing gradient descent on model
        device (str): Which device should everything be performed on (either cuda or cpu)
        verbose (bool): Should model print out loss and accuracy as often as granualarity specifies. Defaults to False
        granularity (int): How often model prints out info and/or appends to list (will result in total of epochs/granularity outputs). Defaults to 1

    Returns:
        dict: dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    train_accuracy=[]
    test_accuracy=[]    
    train_loss=[]
    test_loss=[]
    for i in tqdm(range(epochs)):
        model.train()
        tr_correct,tr_loss=train_step(model=model,train_dataloader=train_dataloader,loss_fn=loss_fn,optimizer=optimizer,device=device)
        if (i%granularity==0):
            with torch.inference_mode():
                model.eval()
                te_correct,te_loss=test_step(model=model,test_dataloader=test_dataloader,loss_fn=loss_fn,device=device)
            tr_accuracy=tr_correct/len(train_dataloader.dataset)
            train_accuracy.append(tr_accuracy)
            train_loss.append(tr_loss)
            te_accuracy=te_correct/len(test_dataloader.dataset)
            test_accuracy.append(te_accuracy)
            test_loss.append(te_loss)
            if verbose:
                print("Train Accuracy: ",tr_accuracy)
                print("Test Accuracy: ",te_accuracy)
            if writer:
                # Add loss results to SummaryWriter
                writer.add_scalar(tag="Loss/train_loss", scalar_value=tr_loss, global_step=i)
                writer.add_scalar(tag="Loss/test_loss", scalar_value=te_loss, global_step=i)

                # Add accuracy results to SummaryWriter
                writer.add_scalar(tag="Accuracy/train_acc", scalar_value=tr_accuracy, global_step=i)
                writer.add_scalar(tag="Accuracy/test_acc", scalar_value=te_accuracy, global_step=i)
                
                writer.flush()
        
    # Close the writer
    if writer:
        # Track the PyTorch model architecture
        writer.add_graph(model=model, 
                        # Pass in an example input
                        input_to_model=torch.randn(1, 3, 224, 224).to(device))
        writer.close()
        
    return {
        "train_loss": train_loss,
        "train_acc": train_accuracy,
        "test_loss": test_loss,
        "test_acc": test_accuracy
    }