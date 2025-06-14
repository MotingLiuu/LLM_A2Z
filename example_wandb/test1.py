import wandb

run = wandb.init(
    project="huggingface_example",
    entity="mutyuu",
    settings=wandb.Settings(
        init_timeout=300,
        _disable_stats=True,
        _disable_viewer=True
    )
)
print("WandB initialized:", run.name)
