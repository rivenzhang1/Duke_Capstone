"""Wall time and sampled process RSS (includes native tensors, not accelerator memory)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class ResourceUse:
    seconds: float = 0.0
    rss_start_bytes: int = 0
    rss_peak_bytes: int = 0


class Measure:
    """Sample RSS every 10 ms. Peak is approximate and includes resident model weights."""

    def __enter__(self) -> ResourceUse:
        import psutil

        self.process = psutil.Process()
        self.result = ResourceUse(rss_start_bytes=self.process.memory_info().rss)
        self.result.rss_peak_bytes = self.result.rss_start_bytes
        self.stop = threading.Event()
        self.start = time.perf_counter()

        def sample():
            while not self.stop.wait(0.01):
                self.result.rss_peak_bytes = max(
                    self.result.rss_peak_bytes, self.process.memory_info().rss
                )

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self.result

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.result.seconds = time.perf_counter() - self.start
        self.result.rss_peak_bytes = max(self.result.rss_peak_bytes, self.process.memory_info().rss)
