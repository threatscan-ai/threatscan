import torch
from tqdm import tqdm

def evaluate(model, test_loader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating"):
            inputs_dict = batch_data[0]
            labels = batch_data[1].to(device)

            pixel_values = inputs_dict['pixel_values'].to(device)
            if pixel_values.ndim == 6 and pixel_values.shape[1] == 1:
                pixel_values = pixel_values.squeeze(1)

            outputs = model(pixel_values=pixel_values) # Pass pixel_values directly
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")