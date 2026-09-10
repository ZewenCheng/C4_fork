"""线程安全阶段记录；主机墙钟不冒充异步设备计算时间。"""
from contextlib import contextmanager
from pathlib import Path
import threading
import time
from report_contract import atomic_json, preserve_attempt


class RuntimeMetrics:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()
        self.begin_time = time.time()

    @contextmanager
    def span(self, stage, **metadata):
        started = time.perf_counter()
        state = 'ok'
        try:
            yield
        except BaseException:
            state = 'failed'
            raise
        finally:
            event = {'stage': stage, 'host_wall_seconds': time.perf_counter() - started,
                     'device_seconds': None, 'status': state, **metadata}
            with self.lock:
                self.events.append(event)

    def save(self, path):
        with self.lock:
            events = list(self.events)
        path = Path(path)
        preserve_attempt(path, path.parent / 'metrics_history')
        atomic_json(Path(path), {'infer_time': time.time() - self.begin_time, 'events': events,
                                '说明': '本入口墙钟秒；各阶段可能重叠，不相加充当总时间；设备时间未知'})
