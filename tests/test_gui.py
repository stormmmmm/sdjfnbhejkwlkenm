"""Сквозные проверки состояния реального Tk GUI без ручных диалогов."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from gui import FresnelApp, create_root
import tkinter as tk
from pictures import save_phase_png


class GuiTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = create_root()
        except tk.TclError as error:
            if "no display name" in str(error) or "couldn't connect to display" in str(error):
                self.skipTest("Tk needs a desktop display for GUI integration tests.")
            raise
        self.addCleanup(self.root.destroy)
        self.root.withdraw()
        self.app = FresnelApp(self.root)
        self.root.update_idletasks()

    def generate(self):
        self.app.seed_var.set("42")
        self.app.generate_phase()

    def test_initial_state_and_generate_calculate_invalidate(self):
        self.assertTrue(self.app.calculate_button.instate(["disabled"]))
        self.assertTrue(self.app.save_button.instate(["disabled"]))
        self.generate()
        self.assertEqual(self.app.phase.shape, (128, 128))
        self.assertTrue(self.app.calculate_button.instate(["!disabled"]))
        self.app.calculate()
        self.assertEqual(self.app.result.intensities.shape, (4, 128, 128))
        self.assertTrue(self.app.save_button.instate(["!disabled"]))
        for panel in self.app.screen_panels:
            self.assertIsNotNone(panel.original_image)
            self.assertNotIn("—", panel.metadata.get())
        self.generate()
        self.assertIsNone(self.app.result)
        self.assertTrue(self.app.save_button.instate(["disabled"]))
        self.assertTrue(all(p.original_image is None for p in self.app.screen_panels))

    def test_bad_seed_preserves_phase_and_reports_error(self):
        self.generate()
        original = self.app.phase.copy()
        for bad in ("abc", "1.5", "-1"):
            self.app.seed_var.set(bad)
            with patch("gui.messagebox.showerror") as dialog:
                self.app.generate_phase()
                dialog.assert_called_once()
            np.testing.assert_array_equal(self.app.phase, original)

    def test_load_valid_png_and_reject_rgb_without_losing_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "doe.png"
            save_phase_png(np.zeros((128, 128)), path)
            with patch("gui.filedialog.askopenfilename", return_value=str(path)):
                self.app.load_phase()
            self.assertEqual(self.app.phase.shape, (128, 128))
            self.assertEqual(self.app.seed_var.get(), "")
            original = self.app.phase.copy()
            invalid = Path(directory) / "rgb.png"
            Image.new("RGB", (128, 128)).save(invalid)
            with patch("gui.filedialog.askopenfilename", return_value=str(invalid)), \
                    patch("gui.messagebox.showerror") as dialog:
                self.app.load_phase()
                dialog.assert_called_once()
            np.testing.assert_array_equal(self.app.phase, original)

    def test_save_exports_current_gui_values_and_handles_cancel(self):
        self.generate()
        self.app.calculate()
        with tempfile.TemporaryDirectory() as directory:
            with patch("gui.filedialog.askdirectory", return_value=directory), \
                    patch("gui.messagebox.showinfo") as dialog:
                self.app.save_results()
                dialog.assert_called_once()
            runs = list(Path(directory).iterdir())
            self.assertEqual(len(runs), 1)
            for index in range(4):
                np.testing.assert_array_equal(
                    np.load(runs[0] / f"{index+1}.npy", allow_pickle=False),
                    self.app.result.intensities[index],
                )
            with patch("gui.filedialog.askdirectory", return_value=""):
                self.app.save_results()
            self.assertEqual(list(Path(directory).iterdir()), runs)


if __name__ == "__main__":
    unittest.main()
