def get_wandb_run_name(model_name, **kwargs):
    """
    Generate the run name for Weights & Biases logging.

    Args:
        model_name (str): The name of the model.
        **kwargs: Additional keyword arguments for other parameters.

    Returns:
        str: The run name for Weights & Biases logging.
    """
    wandb_run_name = f"{model_name}"

    # Append additional parameters
    for key, value in kwargs.items():
        wandb_run_name += f"_{key}-{value}"

    return wandb_run_name
