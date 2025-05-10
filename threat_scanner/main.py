import os 
import sys
import argparse

from threat_scanner.detector.threat import ThreatDetector

def main():
    parser = argparse.ArgumentParser(description="Discover threats from video files or live feed.")
    parser.add_argument(
        "--source",
        "-s",
        help="Path to the video file. If not provided, defaults to live feed.",
        default=0,
        dest="source", #store in args.source
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Model to classify video. If not provided, defaults to videomae-base-finetuned-kinetics",
        default="MCG-NJU/videomae-base-finetuned-kinetics",
        dest="model", #store in args.model
    )

    args = parser.parse_args()
    source = args.source
    model = args.model

    if source != 0 and not os.path.exists(source):
        print(f"Error: Video file not found at path: {source}")
        sys.exit(1)
    
    detector = ThreatDetector(model)
    detector.scan(source=source)

if __name__ == "__main__":
    main()