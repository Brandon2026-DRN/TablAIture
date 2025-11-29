"""
Command-line interface for TablAIture.
"""

import argparse
import json
import sys
from pathlib import Path

from .pipeline import TablAIturePipeline
from .features.f0_detection import YINF0Detector, AutocorrelationF0Detector


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TablAIture: Convert guitar audio to tablature"
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input audio file"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (default: print to stdout)"
    )
    
    parser.add_argument(
        "-f", "--format",
        type=str,
        choices=["json", "plain_text", "musicxml", "all"],
        default="json",
        help="Output format (default: json)"
    )
    
    parser.add_argument(
        "--f0-method",
        type=str,
        choices=["yin", "autocorr"],
        default="yin",
        help="F0 detection method (default: yin)"
    )
    
    parser.add_argument(
        "--use-neural",
        action="store_true",
        help="Use neural note classifier (requires trained model)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for neural models (default: cpu)"
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize F0 detector
    if args.f0_method == "yin":
        f0_detector = YINF0Detector()
    else:
        f0_detector = AutocorrelationF0Detector()
    
    # Initialize pipeline
    pipeline = TablAIturePipeline(
        use_neural_model=args.use_neural,
        f0_detector=f0_detector,
        device=args.device
    )
    
    # Process audio
    try:
        result = pipeline.process_audio_file(args.input_file, args.format)
    except Exception as e:
        print(f"Error processing audio: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        if args.format == "json":
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
        elif args.format == "plain_text":
            with open(output_path, "w") as f:
                f.write(result)
        elif args.format == "musicxml":
            with open(output_path, "w") as f:
                f.write(result)
        else:  # all
            if "json" in result:
                json_path = output_path.with_suffix(".json")
                with open(json_path, "w") as f:
                    json.dump(result["json"], f, indent=2)
            if "plain_text" in result:
                txt_path = output_path.with_suffix(".txt")
                with open(txt_path, "w") as f:
                    f.write(result["plain_text"])
            if "musicxml" in result:
                xml_path = output_path.with_suffix(".musicxml")
                with open(xml_path, "w") as f:
                    f.write(result["musicxml"])
        print(f"Output written to {args.output}")
    else:
        # Print to stdout
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(result)


if __name__ == "__main__":
    main()

