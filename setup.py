from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="threat_scanner", 
    version="0.1.2",         
    author="Ogo-oluwasubomi Popoola",
    author_email="popoolaogooluwasubomi@gmail.com",
    description="ThreatScan is an open-source Python package designed to detect potential physical threats in videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/threatscan-ai/threatscan",
    packages=find_packages(), 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
