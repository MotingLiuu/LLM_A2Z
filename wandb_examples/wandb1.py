import wandb

wandb.login()

run = wandb.init(
    project="my-awesome-project",  # Specify your project
    config={                        # Track hyperparameters and metadata
        "learning_rate": 0.01,
        "epochs": 10,
    },  # Set to "online" for real-time tracking,  # Set to "online" for real-time tracking
)
