"""The text side of tools/eval_uz.py (no models needed).  python -m unittest discover -s tests"""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_compare():
    # eval_uz imports torch/librosa at the top; the comparison itself needs neither, so stub them if absent
    for name in ("torch", "librosa"):
        if importlib.util.find_spec(name) is None:
            sys.modules.setdefault(name, type(sys)(name))
    spec = importlib.util.spec_from_file_location("eval_uz", ROOT / "tools" / "eval_uz.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load_compare()


class Compare(unittest.TestCase):
    def test_identical_after_normalizing(self):
        r = ev.compare("Narxi 50.000 so'm!", "narxi ellik ming soʻm")
        self.assertEqual(r["cer"], 0)
        self.assertEqual(r["wer"], 0)
        self.assertEqual(r["confusions"], [])

    def test_uzbek_letter_confusions(self):
        r = ev.compare("Qo'lingizni ko'lga solmang", "kolingizni ko'lga solmang")
        self.assertIn("q→k", r["confusions"])
        self.assertIn("o'→o", r["confusions"])
        r = ev.compare("she'r va xol", "sher va hol")
        self.assertIn("'→∅", r["confusions"])
        self.assertIn("x→h", r["confusions"])

    def test_canon(self):
        self.assertEqual(ev.canon("“Salom”, dedi u — o'zbekcha!"), "salom dedi u o'zbekcha")
        self.assertEqual(ev.canon("Тошкент"), "toshkent")


if __name__ == "__main__":
    unittest.main()
