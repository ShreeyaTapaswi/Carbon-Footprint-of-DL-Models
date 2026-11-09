"""Energy / CO2 measurement (codecarbon) and model complexity (thop)."""
import copy
from contextlib import contextmanager

import torch
from codecarbon import EmissionsTracker
from thop import profile


def count_flops_params(model, img_size=224, device="cpu"):
    """Returns (GFLOPs, millions of params). thop reports MACs; FLOPs ~= 2 x MACs."""
    m = copy.deepcopy(model).to(device).eval()  # deepcopy: thop attaches hooks/buffers
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    macs, params = profile(m, inputs=(dummy,), verbose=False)
    return 2 * macs / 1e9, params / 1e6


@contextmanager
def track_energy(label: str, out_dir: str = "results"):
    """Usage:
        with track_energy("train") as r:
            ...train...
        r["energy_kwh"], r["co2_kg"]
    """
    result = {}
    tracker = EmissionsTracker(project_name=label, output_dir=out_dir, log_level="error",
                               save_to_file=False, measure_power_secs=1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    tracker.start()
    try:
        yield result
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        co2 = tracker.stop()
        data = getattr(tracker, "final_emissions_data", None)
        energy = getattr(data, "energy_consumed", None)
        if energy is None:  # fallback for older codecarbon versions
            energy = tracker._total_energy.kWh
        result["energy_kwh"] = float(energy)
        result["co2_kg"] = float(co2 or 0.0)
