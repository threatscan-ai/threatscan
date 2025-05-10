import torch
from tqdm import tqdm

def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        print(f"\n--- Starting Epoch {epoch+1}/{num_epochs} ---") # Epoch start marker
        for i, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)")):
            inputs_dict = batch_data[0]
            labels = batch_data[1].to(device)

            # 1. Get the pixel_values tensor from the batch
            pixel_values_from_loader = inputs_dict['pixel_values'].to(device)

            # 2. Attempt to squeeze the tensor
            pixel_values_for_model = pixel_values_from_loader # Initialize with the original

            if pixel_values_from_loader.ndim == 6 and pixel_values_from_loader.shape[1] == 1:
                pixel_values_for_model = pixel_values_from_loader.squeeze(1)

            # 3. Model forward pass
            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values_for_model)
            logits = outputs.logits
            
            # 4. Loss calculation and backward pass
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}")

        # Validation loop (apply similar debugging if issues persist there)
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i_val, batch_data_val in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)")):
                inputs_dict_val = batch_data_val[0]
                labels_val = batch_data_val[1].to(device)

                pixel_values_from_loader_val = inputs_dict_val['pixel_values'].to(device)
                pixel_values_for_model_val = pixel_values_from_loader_val

                if pixel_values_from_loader_val.ndim == 6 and pixel_values_from_loader_val.shape[1] == 1:
                    pixel_values_for_model_val = pixel_values_from_loader_val.squeeze(1)

                outputs_val = model(pixel_values=pixel_values_for_model_val)
                logits_val = outputs_val.logits
                loss_val = criterion(logits_val, labels_val)
                total_val_loss += loss_val.item()

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs}, Average Validation Loss: {avg_val_loss:.4f}")
