import torch
import os
import re
def load_latest_checkpoint(model, checkpoint_dir, prefix):
    # List all files in the checkpoint directory
    checkpoint_files = os.listdir(checkpoint_dir)

    # Regular expression to find files that match the naming pattern
    pattern = re.compile(rf'{prefix}(\d+)\.pth')

    # Find the file with the largest epoch number
    latest_checkpoint_file = None
    latest_epoch = -1
    for filename in checkpoint_files:
        match = pattern.match(filename)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num > latest_epoch:
                latest_epoch = epoch_num
                latest_checkpoint_file = filename

    # Check if we found a checkpoint file
    if latest_checkpoint_file:
        # Full path to the latest checkpoint file
        latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint_file)

        # Load the latest checkpoint
        if torch.cuda.is_available():
            checkpoint = torch.load(latest_checkpoint_path)
        else:
            checkpoint = torch.load(latest_checkpoint_path, map_location=torch.device('cpu'))

        # Load the saved weights into the model
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Loaded checkpoint from epoch {latest_epoch}: {latest_checkpoint_path}")
        return model, latest_epoch
    else:
        print("No checkpoint file found. Model weights not modified.")
        return model, -1

def save_checkpoint(model, epoch, checkpoint_dir, prefix):
    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define the checkpoint filename
    filename = f"{prefix}{epoch}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    # Save the checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, checkpoint_path)

    print(f"Checkpoint saved to {checkpoint_path}")

def mkdir(path):
    isExist = os.path.exists(path)
    if not isExist:# Check whether the specified path exists or not
        os.makedirs(path)
        print(f"The new directory {path} is created!")
