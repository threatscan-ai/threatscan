# Getting Started

This document outlines the setup and usage instructions for the project.

## Setting up your Virtual Environment

It's highly recommended to use a virtual environment to isolate project dependencies. Follow these steps:

1.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    ```
    This command creates a new virtual environment in a directory named `.venv` within your project.

2.  **Activate the virtual environment:**

    * **On macOS and Linux:**
        ```bash
        source .venv/bin/activate
        ```
        Your terminal prompt should now be prefixed with `(.venv)`, indicating that the virtual environment is active.

    * **On Windows (Command Prompt):**
        ```bash
        .venv\Scripts\activate
        ```

    * **On Windows (PowerShell):**
        ```powershell
        .venv\Scripts\Activate.ps1
        ```

## Installing Dependencies

Once the virtual environment is activated, install the project's required libraries using pip:

```bash
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
python3 -m main --source detector/examples/fire.mov
```
Replace `detector/examples/fire.mov` with the actual path to your video file.

* With a Custom Model
To run the application using a specific trained model:

```
python3 -m main --model mymodel/path/etc
```
Replace `mymodel/path/etc` with the correct path to your model file.


## Training your own model
To train your own data go to [Training README](training/README.md)


