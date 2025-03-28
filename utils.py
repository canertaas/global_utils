import os 
from datetime import datetime


def create_output_directories(output_dir):
    """Create necessary directories for outputs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir