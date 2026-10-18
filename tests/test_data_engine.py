import tempfile
import unittest
from pathlib import Path

from dataset.engine import canonical_file_id, chunk_text, normalize_text, quality, read_local


class DataEngineTests(unittest.TestCase):
    def test_normalization(self):
        raw = "A\r\nB-\nC\n\n\nD\xa0E"
        self.assertEqual(normalize_text(raw), "A\nBC\n\nD E")

    def test_quality(self):
        self.assertEqual(quality("abc", 10), (False, "too-short<10"))
        self.assertEqual(quality("1234567890" * 10, 10), (False, "low-alpha-ratio"))
        self.assertEqual(quality("Research data " * 100, 100), (True, "ok"))

    def test_chunk_overlap_and_boundaries(self):
        text = "paragraph one\n\nparagraph two\n\nparagraph three " * 50
        chunks = list(chunk_text(text, 120, 20))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(x) <= 120 for x in chunks))

    def test_local_txt_and_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            txt = root / "note.txt"
            html = root / "page.html"
            txt.write_text("Machine learning " * 100, encoding="utf-8")
            html.write_text("<html><body><p>Machine learning</p><script>alert(1)</script></body></html>", encoding="utf-8")
            txt_text, txt_kind = read_local(txt)
            html_text, html_kind = read_local(html)
            self.assertEqual(txt_kind, "text")
            self.assertEqual(html_kind, "html")
            self.assertIn("Machine learning", txt_text)
            self.assertIn("Machine learning", html_text)
            self.assertNotIn("alert", html_text)

    def test_canonical_file_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "papers" / "paper.txt"
            nested.parent.mkdir()
            nested.touch()
            self.assertEqual(canonical_file_id(nested, root), "file:papers/paper.txt")


if __name__ == "__main__":
    unittest.main()
