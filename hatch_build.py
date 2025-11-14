"""Custom Hatchling build hook to compile packmol binary."""

import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Build hook that compiles packmol and includes it in the wheel."""

    def initialize(self, version, build_data):
        """Compile packmol before building the wheel."""
        # Paths
        project_root = Path(self.root)
        packmol_source = project_root / "external" / "packmol"
        binaries_dir = project_root / "src" / "molify" / "binaries"
        target_binary = binaries_dir / "packmol"

        # Ensure binaries directory exists
        binaries_dir.mkdir(parents=True, exist_ok=True)

        # Verify packmol source exists
        if not packmol_source.exists():
            raise RuntimeError(
                f"Packmol source not found at {packmol_source}. "
                "Run: git submodule update --init --recursive"
            )

        # Compile packmol
        print("Compiling packmol...")
        try:
            # Clean previous build artifacts
            subprocess.run(
                ["make", "clean"],
                cwd=packmol_source,
                check=False,  # Don't fail if clean fails
                capture_output=True,
            )

            # Override FORTRAN variable to use gfortran from PATH
            # Note: Not using -j4 due to Fortran module dependencies
            result = subprocess.run(
                ["make", "FORTRAN=gfortran"],
                cwd=packmol_source,
                check=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Compilation failed!\nstdout: {e.stdout}\nstderr: {e.stderr}")
            raise RuntimeError(f"Failed to compile packmol: {e}") from e

        # Verify and copy the compiled binary
        source_binary = packmol_source / "packmol"
        if not source_binary.exists():
            raise RuntimeError(
                f"Compilation succeeded but binary not found at {source_binary}"
            )

        shutil.copy2(source_binary, target_binary)
        target_binary.chmod(0o755)
        print(f"✓ Successfully compiled packmol -> {target_binary}")

        # Include binary in the wheel using force_include
        relative_path = target_binary.relative_to(project_root / "src")
        build_data.setdefault("force_include", {})[str(target_binary)] = str(
            relative_path
        )

        # Mark wheel as platform-specific (not pure Python)
        # This tells hatchling to create platform-specific wheel tags
        # like cp310-macosx_14_0_arm64 instead of py3-none-any
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
