"""
This file is part of nataili ("Homepage" = "https://github.com/Sygil-Dev/nataili").

Copyright 2022 hlky and Sygil-Dev
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import threading
import time

import psutil
import pynvml


class VRAMMonitor:
    def __init__(self, device_id=0, interval=1, unit="MB"):
        self.is_running = False
        self.monitor_thread = None
        self.peak_usage = 0
        self.start_usage = 0
        self.end_usage = 0
        self.usage_history = {}
        self.named_history = {}
        self.interval = interval
        self.step = 0
        self.unit = unit
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        self.total = self.get_total()

    def __del__(self):
        self.is_running = False
        if self.monitor_thread is not None:
            self.monitor_thread.join()
        self.monitor_thread = None
        pynvml.nvmlShutdown()

    def to_unit(self, value, unit):
        # B, KB, MB, GB
        if unit == "B":
            return value
        elif unit == "KB":
            return round(value / 1024, 2)
        elif unit == "MB":
            return round(value / 1024 / 1024, 2)
        elif unit == "GB":
            return round(value / 1024 / 1024 / 1024, 2)
        else:
            raise ValueError(f"Unsupported unit: {unit}")

    def get_total(self, unit=None):
        if unit is None:
            unit = self.unit
        return self.to_unit(pynvml.nvmlDeviceGetMemoryInfo(self.handle).total, unit)

    def get_in_use(self, unit=None):
        if unit is None:
            unit = self.unit
        return self.to_unit(pynvml.nvmlDeviceGetMemoryInfo(self.handle).used, unit)

    def get_available(self, unit=None):
        if unit is None:
            unit = self.unit
        return self.to_unit(pynvml.nvmlDeviceGetMemoryInfo(self.handle).free, unit)

    def get_peak(self, unit=None):
        if unit is None:
            unit = self.unit
        return self.to_unit(self.peak_usage, unit)

    def get_history_average(self):
        """
        History average uses bytes, so we can't pass unit into this function.
        """
        return self.to_unit(sum(self.usage_history.values()) / len(self.usage_history), self.unit)

    def record_event(self, name):
        self.named_history[name] = self.get_in_use()

    def get_event(self, name):
        return self.named_history[name]

    def get_events(self):
        return self.named_history

    def log_events(self):
        for name, value in self.named_history.items():
            print(f"{name}: {value} {self.unit}")

    def reset(self):
        self.peak_usage = 0
        self.start_usage = 0
        self.end_usage = 0
        self.usage_history = {}
        self.step = 0

    def start(self):
        self.reset()
        self.is_running = True
        self.start_usage = self.get_in_use()

        # Start the monitoring in the background
        self.monitor_thread = threading.Thread(target=self.monitor)
        self.monitor_thread.start()

    def stop(self):
        self.is_running = False
        self.end_usage = self.get_in_use()
        return {
            "start": self.start_usage,
            "end": self.end_usage,
            "peak": self.get_peak(),
            "average": self.get_history_average(),
        }

    def monitor(self):
        while self.is_running:
            in_use = self.get_in_use(unit="B")  # need to use B for the max function
            self.peak_usage = max(self.peak_usage, in_use)
            self.usage_history[self.step] = in_use
            self.step += 1
            time.sleep(self.interval)


class RAMMonitor:
    def __init__(self, interval=1, unit="MB"):
        self.is_running = False
        self.monitor_thread = None
        self.peak_usage = 0
        self.start_usage = 0
        self.end_usage = 0
        self.usage_history = {}
        self.interval = interval
        self.step = 0
        self.unit = unit
        self.total = self.get_total()

    def __del__(self):
        self.is_running = False
        if self.monitor_thread is not None:
            self.monitor_thread.join()
        self.monitor_thread = None

    def to_unit(self, value, unit):
        # B, KB, MB, GB
        if unit == "B":
            return value
        elif unit == "KB":
            return round(value / 1024, 2)
        elif unit == "MB":
            return round(value / 1024 / 1024, 2)
        elif unit == "GB":
            return round(value / 1024 / 1024 / 1024, 2)
        else:
            raise ValueError(f"Unsupported unit: {unit}")

    def get_total(self, unit=None):
        if unit is None:
            unit = self.unit
        return self.to_unit(psutil.virtual_memory().total, unit)

    def get_in_use(self, unit=None):
        if unit is None:
            unit = self.unit
        return self.to_unit(psutil.virtual_memory().used, unit)

    def get_available(self, unit=None):
        if unit is None:
            unit = self.unit
        return self.to_unit(psutil.virtual_memory().available, unit)

    def get_peak(self, unit=None):
        if unit is None:
            unit = self.unit
        return self.to_unit(self.peak_usage, unit)

    def get_history_average(self):
        """
        History average uses bytes, so we can't pass unit into this function.
        """
        return self.to_unit(sum(self.usage_history.values()) / len(self.usage_history), self.unit)

    def reset(self):
        self.peak_usage = 0
        self.start_usage = 0
        self.end_usage = 0
        self.usage_history = {}
        self.step = 0

    def start(self):
        self.reset()
        self.is_running = True
        self.start_usage = self.get_in_use()

        # Start the monitoring in the background
        self.monitor_thread = threading.Thread(target=self.monitor)
        self.monitor_thread.start()

    def stop(self):
        self.is_running = False
        self.end_usage = self.get_in_use()
        return {
            "start": self.start_usage,
            "end": self.end_usage,
            "peak": self.get_peak(),
            "average": self.get_history_average(),
        }

    def monitor(self):
        while self.is_running:
            in_use = self.get_in_use(unit="B")  # need to use B for the max function
            self.peak_usage = max(self.peak_usage, in_use)
            self.usage_history[self.step] = in_use
            self.step += 1
            time.sleep(self.interval)
