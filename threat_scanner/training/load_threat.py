import json

from threat_scanner.training.threat import Threat

def load_threat(filepath="threat_scanner/training/threats.json"):
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
            threats = []
            for threat_data in list(data.values()):
                threats.append(Threat(threat_data))
            print("Threats successfully loaded!")
            return threats
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
        return None
    except Exception as e:
         print(f"An unexpected error occurred: {e}")
         return None