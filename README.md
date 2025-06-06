# ThreatScan
**ThreatScan** is an open-source Python package designed to detect potential physical threats in videos. It leverages the power of video masking autoencoder (VideoMAE) classification models, fine-tuning them to specifically identify events such as fires and other user-defined threats.

## Overview

This package provides a streamlined approach to:

* **Utilize pre-trained VideoMAE models:** Benefit from state-of-the-art video understanding capabilities.
* **Fine-tune for threat detection:** Adapt the base model to accurately classify specific threat events present in video data.
* **Scalable threat analysis:** Process video streams or individual video files for real-time or batch analysis.
* **Extensible architecture:** Easily integrate custom threat classes and extend the model's capabilities.

## Installation

You can install [ThreatScan](https://pypi.org/project/threat-scanner/) using pip:
```
pip install threat-scanner
```

# Getting Started

This document outlines the setup and usage instructions for the project.

## Setting up your Virtual Environment

It's highly recommended to use a virtual environment to isolate project dependencies. Follow these steps:

1.  **Create a virtual environment:**
    ```
    python3 -m venv .venv
    ```
    This command creates a new virtual environment in a directory named `.venv` within your project.

2.  **Activate the virtual environment:**

    * **On macOS and Linux:**
        ```
        source .venv/bin/activate
        ```
        Your terminal prompt should now be prefixed with `(.venv)`, indicating that the virtual environment is active.

    * **On Windows (Command Prompt):**
        ```
        .venv\Scripts\activate
        ```

    * **On Windows (PowerShell):**
        ```powershell
        .venv\Scripts\Activate.ps1
        ```

## Installing Dependencies

Once the virtual environment is activated, install the project's required libraries using pip:

```
pip install -r requirements.txt
```

## Running the Application
You can run the main application in different modes:

* Live Feed (Default)
To run the application using your default webcam or live video feed:

```
python3 -m main
```

* With a Video File
To run the application using a specific video file:

```
python3 -m threat_scanner.main --source threat_scanner/detector/examples/fire.mp4
```
Replace `threat_scanner/detector/examples/fire.mp4` with the actual path to your video file.

* With a Custom Model
To run the application using a specific trained model:

```
python3 -m threat_scanner.main --model mymodel/path/etc
```
Replace `mymodel/path/etc` with the correct path to your model file.


## Training your own model
To train your own data go to [Training README](training/README.md)

# Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

# License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.