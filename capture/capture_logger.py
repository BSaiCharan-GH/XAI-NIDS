"""
capture_logger.py
-----------------
Consumes completed Flow objects from the flow tracker, extracts features,
writes them to CSV, and optionally sends every completed flow to the live
machine-learning inference pipeline.
"""

import csv
import queue
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from capture.constants import CAPTURE_CSV_DIR, FEATURE_COLUMNS, MAX_CSV_ROWS
from capture.feature_extractor import extract_features
from capture.flow import Flow


class CaptureLogger:
    def __init__(
        self,
        completed_queue: queue.Queue,
        output_dir: str = CAPTURE_CSV_DIR,
        session_name: Optional[str] = None,
        on_flow_completed: Optional[Callable] = None,
        verbose: bool = True,
    ) -> None:
        self._queue = completed_queue
        self._output_dir = Path(output_dir)
        self._session = session_name or _make_session_name()
        self._on_flow = on_flow_completed
        self._verbose = verbose
        self._rows_written = 0
        self._file_index = 0
        self._csv_path: Optional[Path] = None
        self._csv_file = None
        self._csv_writer = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._open_csv()
        self._thread = threading.Thread(target=self._consume_loop, daemon=True, name="capture-logger")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        self._close_csv()

    @property
    def csv_path(self) -> Optional[Path]:
        return self._csv_path

    @property
    def rows_written(self) -> int:
        return self._rows_written

    def _consume_loop(self) -> None:
        while self._running or not self._queue.empty():
            try:
                flow: Flow = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                features = extract_features(flow)
                self._write_row(features)

                inference_result = None
                if self._on_flow:
                    try:
                        inference_result = self._on_flow(features)
                    except Exception as exc:
                        print(f"[INFERENCE ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)

                if self._verbose:
                    self._print_flow_summary(features, inference_result)
            except Exception as exc:
                print(f"[LOGGER] Feature extraction error: {type(exc).__name__}: {exc}", file=sys.stderr)
            finally:
                self._queue.task_done()

    def _write_row(self, features: dict) -> None:
        if self._rows_written > 0 and self._rows_written % MAX_CSV_ROWS == 0:
            self._rotate_csv()
        if self._csv_writer is None:
            self._open_csv()
        self._csv_writer.writerow({col: features.get(col, "") for col in FEATURE_COLUMNS})
        self._csv_file.flush()
        self._rows_written += 1

    def _open_csv(self) -> None:
        suffix = f"_{self._file_index}" if self._file_index > 0 else ""
        self._csv_path = self._output_dir / f"{self._session}{suffix}.csv"
        is_new = not self._csv_path.exists()
        self._csv_file = open(self._csv_path, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=FEATURE_COLUMNS)
        if is_new:
            self._csv_writer.writeheader()

    def _close_csv(self) -> None:
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.flush()
            self._csv_file.close()

    def _rotate_csv(self) -> None:
        self._close_csv()
        self._file_index += 1
        self._open_csv()

    def _print_flow_summary(self, features: dict, inference_result=None) -> None:
        proto_name = {6: "TCP", 17: "UDP", 1: "ICMP"}.get(features.get("protocol", 0), "???")
        dur_ms = int(features.get("Flow Duration", 0) / 1000)
        ml_label = inference_result.get("ml_prediction") if isinstance(inference_result, dict) else None
        final_label = inference_result.get("final_decision") if isinstance(inference_result, dict) else None
        behaviour = inference_result.get("behavioural_detection") if isinstance(inference_result, dict) else None
        suffix = ""
        if ml_label is not None:
            suffix = f" | ML: {ml_label} | Behaviour: {behaviour} | Final: {final_label}"
        print(
            f"  [FLOW] {features.get('src_ip','?')}:{features.get('src_port','?')} -> "
            f"{features.get('dst_ip','?')}:{features.get('dst_port','?')} "
            f"| {proto_name} | {dur_ms} ms "
            f"| pkts: {int(features.get('Total Fwd Packets', 0))}+{int(features.get('Total Backward Packets', 0))}"
            f"{suffix}"
        )


def _make_session_name() -> str:
    return "capture_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
