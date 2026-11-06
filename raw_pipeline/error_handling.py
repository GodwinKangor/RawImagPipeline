# Author: Godwin Kangor
# Date: 2026-09-28
# File name: error_handling.py

from datetime import datetime
from pathlib import Path
import sys


ERROR_LOG = Path(__file__).resolve().parent.parent / "outputs" / "errors.log"


# Log the error and stop execution so invalid data cannot reach the next stage.
def log_error(message):
    message = f"ERROR: {message}"
    print(message, file=sys.stderr)
    try:
        path = Path(ERROR_LOG)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as log:
            timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
            log.write(f"{timestamp} | {message}\n")
        print(f"Pipeline stopped. Error saved to {path}", file=sys.stderr)
    except OSError as error:
        print(f"Pipeline stopped. Could not write error log: {error}", file=sys.stderr)
    sys.exit(1)
