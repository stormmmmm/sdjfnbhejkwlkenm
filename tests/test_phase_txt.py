import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from gui import FresnelApp, create_root
from pictures import load_phase_txt
from simulator import simulate, simulate_phase_txt


class PhaseTextTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'phase.txt'

    def test_preserves_phase_precision_and_supported_delimiters(self):
        phase = np.random.default_rng(7).uniform(-np.pi, np.pi, (128, 128))
        phase[0, 0], phase[-1, -1] = -np.pi, np.pi
        for delimiter in (' ', '\t', ',', ';'):
            with self.subTest(delimiter=delimiter):
                np.savetxt(self.path, phase, delimiter=delimiter, fmt='%.17g', encoding='utf-8-sig')
                with self.path.open('a', encoding='utf-8') as stream:
                    stream.write('\n# comment\n')
                np.testing.assert_array_equal(load_phase_txt(self.path), phase)

    def test_rejects_invalid_shape_values_and_malformed_text(self):
        for shape in ((127, 128), (128, 127), (1, 16384)):
            np.savetxt(self.path, np.zeros(shape))
            with self.assertRaises(ValueError):
                load_phase_txt(self.path)
        for value in (np.nan, np.inf, -np.inf, np.pi+.01, -np.pi-.01):
            phase = np.zeros((128, 128)); phase[12, 73] = value
            np.savetxt(self.path, phase)
            with self.assertRaises(ValueError):
                load_phase_txt(self.path)
        for text in ('', '# only comment', 'abc', '0 1\n2', '[0, 1]'):
            self.path.write_text(text, encoding='utf-8')
            with self.assertRaises(ValueError):
                load_phase_txt(self.path)

    def test_accepts_rounded_pi_but_rejects_larger_range_violation(self):
        phase = np.linspace(-np.pi, np.pi, 128*128).reshape(128, 128)
        np.savetxt(self.path, phase, fmt="%.9f")
        stored = np.loadtxt(self.path)
        result = load_phase_txt(self.path)
        self.assertEqual(result[0, 0], -np.pi)
        self.assertEqual(result[-1, -1], np.pi)
        np.testing.assert_array_equal(result.ravel()[1:-1], stored.ravel()[1:-1])
        for value in (np.pi + 2e-9, -np.pi - 2e-9):
            phase[0, 0] = value
            np.savetxt(self.path, phase, fmt="%.17g")
            with self.assertRaises(ValueError):
                load_phase_txt(self.path)

    def test_missing_file_has_readable_error(self):
        with self.assertRaisesRegex(ValueError, 'TXT'):
            load_phase_txt(self.path)

    def test_backend_uses_existing_physics_and_four_ffts_without_quantization(self):
        phase = np.random.default_rng(4).uniform(-np.pi, np.pi, (128, 128))
        np.savetxt(self.path, phase, fmt='%.17g')
        with patch('mathlib.np.fft.fft2', wraps=np.fft.fft2) as fft:
            result = simulate_phase_txt(self.path)
        self.assertEqual(fft.call_count, 4)
        np.testing.assert_array_equal(result.phase, phase)
        np.testing.assert_array_equal(result.intensities, simulate(phase).intensities)
        self.assertEqual(result.source, str(self.path.resolve()))
        self.assertIsNone(result.seed)

    def test_gui_loads_and_automatically_calculates_preserving_old_state_on_error(self):
        root = create_root(); self.addCleanup(root.destroy); root.withdraw()
        app = FresnelApp(root)
        phase = np.zeros((128, 128))
        np.savetxt(self.path, phase)
        with patch('gui.filedialog.askopenfilename', return_value=str(self.path)):
            app.load_txt_button.invoke()
        self.assertIsNotNone(app.result)
        np.testing.assert_array_equal(app.result.phase, phase)
        self.assertEqual(app.result.intensities.shape, (4, 128, 128))
        self.assertTrue(all(panel.original_image is not None for panel in app.screen_panels))
        previous = app.result
        self.path.write_text('broken input', encoding='utf-8')
        with patch('gui.filedialog.askopenfilename', return_value=str(self.path)), patch('gui.messagebox.showerror') as error:
            app.load_phase_text()
            error.assert_called_once()
        self.assertIs(app.result, previous)
        with patch('gui.filedialog.askopenfilename', return_value=''):
            app.load_phase_text()
        self.assertIs(app.result, previous)


if __name__ == '__main__':
    unittest.main()
