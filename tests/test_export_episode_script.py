import os
import pathlib
import subprocess
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class ExportEpisodeScriptTest(unittest.TestCase):
    def test_export_episode_script_builds_expected_command(self) -> None:
        env = os.environ.copy()
        env["DRY_RUN"] = "1"
        env["LEROBOT_EPISODE_PATH"] = "All_datas/pour_tea100/data/chunk-000/episode_000009.parquet"
        env["LEROBOT_EXPORT_OUTPUT"] = "/tmp/episode_000009.xlsx"
        env["LEROBOT_IMAGE_WIDTH"] = "120"

        result = subprocess.run(
            ["bash", "tools/05_export_episode_excel.sh"],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn(".venv_lerobot/bin/python", result.stdout)
        self.assertIn("export_lerobot_episode_to_excel.py", result.stdout)
        self.assertIn("All_datas/pour_tea100/data/chunk-000/episode_000009.parquet", result.stdout)
        self.assertIn("-o /tmp/episode_000009.xlsx", result.stdout)
        self.assertIn("--image-width 120", result.stdout)


if __name__ == "__main__":
    unittest.main()
