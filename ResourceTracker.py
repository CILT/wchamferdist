import time
import psutil
import torch
import os

class ResourceTracker:
    def __init__(self, device):
        self.device = device
        self.process = psutil.Process(os.getpid())

    def start(self):
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        self.t0 = time.perf_counter()
        self.cpu_mem0 = self.process.memory_info().rss
        self.cpu_times0 = self.process.cpu_times()

    def stop(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1024**2
            gpu_mem_reserved = torch.cuda.max_memory_reserved() / 1024**2
        else:
            gpu_mem_alloc = None
            gpu_mem_reserved = None

        t1 = time.perf_counter()
        cpu_mem1 = self.process.memory_info().rss
        cpu_times1 = self.process.cpu_times()

        return {
            "wall_time_sec": t1 - self.t0,
            "cpu_mem_mb": (cpu_mem1 - self.cpu_mem0) / 1024**2,
            "cpu_user_time_sec": cpu_times1.user - self.cpu_times0.user,
            "cpu_system_time_sec": cpu_times1.system - self.cpu_times0.system,
            "gpu_mem_alloc_mb": gpu_mem_alloc,
            "gpu_mem_reserved_mb": gpu_mem_reserved,
        }
