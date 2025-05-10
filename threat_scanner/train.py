import sys
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from threat_scanner.training.train import train
from threat_scanner.training.evaluate import evaluate
from threat_scanner.training.load_model import load_model
from threat_scanner.training.load_threat import load_threat
from threat_scanner.training.save_model import save_model
from threat_scanner.training.dataset import ThreatDataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 10
    learning_rate = 1e-4
    batch_size = 4
    model_path = "finetuned_videomae_threats.pth"

    threats = load_threat()
    if not threats:
        print("No data specified for training")
        sys.exit(1)

    directories = [threat.path for threat in threats]
    label2id = {threat.name: threat.id for _, threat in enumerate(threats)}
    id2label = {threat.id: threat.name for _, threat in enumerate(threats)}

    model, processor = load_model(label2id, id2label)
    
    labels = [i for i in range(len(directories))]
    dataset = ThreatDataset(directories, labels, processor)

    train_data, test_data = train_test_split(dataset.data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    train_dataset = ThreatDataset(directories, labels, processor)
    train_dataset.data = train_data

    val_dataset = ThreatDataset(directories, labels, processor)
    val_dataset.data = val_data

    test_dataset = ThreatDataset(directories, labels, processor)
    test_dataset.data = test_data
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=num_epochs)
    evaluate(model, test_loader, device)
    save_model(model, model_path)
    print(f"Model saved to {model_path} ")

if __name__ == "__main__":
    main()