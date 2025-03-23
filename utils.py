import os 
from datetime import datetime


def create_output_directories(self):
    """Create necessary directories for outputs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    self.results_dir = os.path.join(self.output_dir, f"run_{timestamp}")
    os.makedirs(self.results_dir, exist_ok=True)
    os.makedirs(os.path.join(self.results_dir, "models"), exist_ok=True)