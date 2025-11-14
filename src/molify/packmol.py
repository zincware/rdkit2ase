"""Python wrapper for packmol binary."""

import subprocess
from pathlib import Path
import sys


def get_packmol_binary() -> Path:
    """Get the path to the packmol binary.

    Returns
    -------
    Path
        Path to the packmol binary.

    Raises
    ------
    RuntimeError
        If the binary is not found.
    """
    binary_path = Path(__file__).parent / "binaries" / "packmol"

    if not binary_path.exists():
        raise RuntimeError(
            f"Packmol binary not found at: {binary_path}\n"
            f"This may indicate an incomplete installation."
        )

    return binary_path


def run_packmol(input_file: str | Path, *, timeout: int | None = None) -> str:
    """Run packmol with the given input file.

    Parameters
    ----------
    input_file : str or Path
        Path to the packmol input file.
    timeout : int, optional
        Timeout in seconds for the packmol process.

    Returns
    -------
    str
        Combined stdout and stderr from packmol.

    Raises
    ------
    RuntimeError
        If packmol execution fails.
    FileNotFoundError
        If the input file doesn't exist.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    binary = get_packmol_binary()

    try:
        result = subprocess.run(
            [str(binary)],
            stdin=open(input_path),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        return result.stdout + result.stderr
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Packmol execution failed with return code {e.returncode}\n"
            f"stdout: {e.stdout}\n"
            f"stderr: {e.stderr}"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"Packmol execution timed out after {timeout} seconds"
        ) from e


def packmol_cli():
    """CLI entry point for packmol command.

    This function executes the bundled packmol binary, forwarding all
    command-line arguments and stdin/stdout. This makes the `packmol`
    command available after installing molify.
    """
    binary = get_packmol_binary()

    # Execute packmol with the same stdin/stdout as this process
    # This allows users to run: packmol < input.inp
    result = subprocess.run(
        [str(binary)],
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    sys.exit(result.returncode)


__all__ = ["get_packmol_binary", "run_packmol", "packmol_cli"]
