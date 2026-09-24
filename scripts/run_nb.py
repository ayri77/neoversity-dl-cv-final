"""Execute a notebook in place and stream its output to a log file in real time.

Same result as ``jupyter nbconvert --execute --inplace`` (outputs are saved
into the notebook), but every stdout/stderr line is also appended to
``logs/<notebook>.log`` as soon as the kernel prints it, so a long run can be
followed with::

    Get-Content logs/05_text_models.log -Wait        # PowerShell
    tail -f logs/05_text_models.log                  # bash

Usage: ``uv run python scripts/run_nb.py notebooks/05_text_models.ipynb [--timeout 7200]``
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"


class LoggingClient(NotebookClient):
    """NotebookClient that mirrors kernel output messages to a log file."""

    def __init__(self, nb, log, **kwargs):
        super().__init__(nb, **kwargs)
        self.logfile = log
        self.t0 = time.time()
        self.on_cell_execute = self._cell_started

    def _stamp(self) -> str:
        return time.strftime("%H:%M:%S") + f" +{(time.time() - self.t0) / 60:.1f}m"

    def _cell_started(self, cell, cell_index: int) -> None:
        first_line = cell.source.strip().splitlines()[0] if cell.source.strip() else ""
        self.logfile.write(f"\n[{self._stamp()}] ── cell {cell_index}: {first_line[:80]}\n")
        self.logfile.flush()

    def output(self, outs, msg, display_id, cell_index):
        content, msg_type = msg["content"], msg["msg_type"]
        if msg_type == "stream":
            # tqdm redraws with \r: turn every redraw into its own log line
            self.logfile.write(content["text"].replace("\r", "\n"))
        elif msg_type == "error":
            self.logfile.write(f"ERROR {content['ename']}: {content['evalue']}\n")
        elif msg_type in ("execute_result", "display_data") and "text/plain" in content.get("data", {}):
            if "image/png" not in content["data"]:
                self.logfile.write(content["data"]["text/plain"] + "\n")
        self.logfile.flush()
        return super().output(outs, msg, display_id, cell_index)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=7200, help="per-cell timeout, seconds")
    parser.add_argument("--kernel", default="neoversity-dl")
    args = parser.parse_args()

    nb_path = args.notebook.resolve()
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{nb_path.stem}.log"
    nb = nbformat.read(nb_path, as_version=4)

    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] run {nb_path.name}\n")
        client = LoggingClient(
            nb, log, timeout=args.timeout, kernel_name=args.kernel,
            resources={"metadata": {"path": str(nb_path.parent)}},
        )
        status = 0
        try:
            client.execute()
        except CellExecutionError:
            status = 1  # the error itself is already in the log via output()
        finally:
            nbformat.write(nb, nb_path)  # save outputs even after a failure
            log.write(f"\n[{client._stamp()}] {'FAILED' if status else 'done'} → saved {nb_path.name}\n")
    print(f"{'FAILED' if status else 'done'}: {nb_path.name}, log: {log_path}")
    return status


if __name__ == "__main__":
    sys.exit(main())
