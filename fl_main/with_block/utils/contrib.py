def calculate_contribution(
    client_data,
    global_weights,
    initial_accuracy,
    final_accuracy,
    initial_loss,
    final_loss,
):
    local_weights = client_data["weight"]
    dataset_size = client_data["dataset_size"]
    local_epochs = client_data["local_epochs"]

    base_score = dataset_size * local_epochs
    weight_diff = sum(
        (local_weights[key] - global_weights[key]).norm() for key in local_weights
    )
    accuracy_improvement = max(0, final_accuracy - initial_accuracy)
    loss_reduction = max(0, initial_loss - final_loss)

    contribution_score = (
        (0.7 * base_score)
        + (0.2 * weight_diff.item())
        + (0.1 * accuracy_improvement)
        + (0.1 * loss_reduction)
    )
    return contribution_score
