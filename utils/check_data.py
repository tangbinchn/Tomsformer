def check_model_device(model, device):
    print("Expected device:", device)
    print("Model device:", next(model.parameters()).device)


def check_data_device(data, label, device):
    print("Expected device:", device)
    print("Data device:", data.device)
    print("Label device:", label.device)
