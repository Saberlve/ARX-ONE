import os
import pathlib
import subprocess
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class CollectScriptTest(unittest.TestCase):
    def test_collect_script_launches_lerobot_v21_collector(self) -> None:
        env = os.environ.copy()
        env["DRY_RUN"] = "1"
        env["DISPLAY"] = ":0"

        result = subprocess.run(
            ["bash", "tools/01_collect.sh"],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("collect_ledatav21.py", result.stdout)
        self.assertNotIn("python collect.py", result.stdout)
        self.assertIn("--root_path", result.stdout)
        self.assertIn("--repo_id", result.stdout)
        self.assertIn("acone_v21", result.stdout)
        self.assertNotIn("export LEROBOT_ROOT", result.stdout)
        self.assertIn(".venv_lerobot/bin/python", result.stdout)
        self.assertNotIn(".venv_ros/bin/activate", result.stdout)

    def test_lerobot_collector_does_not_hard_require_h5py_or_pyttsx3(self) -> None:
        source = (REPO_ROOT / "src/edlsrobot/datasets/collect_ledatav21.py").read_text()

        self.assertNotIn("import h5py", source)
        self.assertIn("except ModuleNotFoundError", source)
        self.assertIn("pyttsx3 = None", source)


if __name__ == "__main__":
    unittest.main()
