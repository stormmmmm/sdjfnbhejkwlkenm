"""Tkinter-интерфейс симулятора: расчёты выполняются только в backend.

Импорт модуля не создаёт окно. Запуск: python main.py или python gui.py.
"""

from datetime import datetime
from pathlib import Path


def _prepare_tcl():
    """Инициализировать штатную DLL Tcl до импорта tkinter в Windows.

    В этой Windows-среде импорт _tkinter без Tcl_FindExecutable оставляет
    файловую подсистему Tcl неинициализированной: существующий init.tcl
    оказывается невидимым. После импорта исправлять это уже поздно.
    Используется только единственная DLL из выбранной установки Python.
    Ни окно, ни системные файлы или настройки здесь не создаются.
    """
    import sys

    if sys.platform != "win32":
        return None
    import ctypes

    candidates = list((Path(sys.base_prefix) / "DLLs").glob("tcl*t.dll"))
    if len(candidates) != 1:
        return None
    try:
        library = ctypes.CDLL(str(candidates[0]))
        library.Tcl_FindExecutable.argtypes = [ctypes.c_char_p]
        library.Tcl_FindExecutable.restype = None
        library.Tcl_FindExecutable(sys.executable.encode("utf-8"))
        return library
    except (OSError, AttributeError):
        # Сторонняя сборка Python может использовать другой способ загрузки.
        # В этом случае обычный Tk сам сообщит о невозможности инициализации.
        return None


_TCL_RUNTIME = _prepare_tcl()

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from const import (
    IMAGE_SIZE,
    PIXEL_PITCH_M,
    RECONSTRUCTION_DISTANCES_M,
    WAVELENGTH_M,
)
from mathlib import output_pixel_pitch
from pictures import OutputPictures, load_phase_png, load_phase_txt
from simulator import export_results, generate_random_phase, simulate


BACKGROUND = "#edf1f5"
CARD = "#ffffff"
TEXT = "#182b40"
MUTED = "#52657a"
ACCENT = "#215b91"
CANVAS = "#e8edf3"


def phase_rgb(phase: np.ndarray) -> Image.Image:
    """Отобразить фиксированную шкалу [−π, π] в цвете, не меняя фазу.

    Цвета интерполируются между пятью опорными точками. Это только
    палитра интерфейса: физические вычисления не зависят от её выбора.
    """
    stops = np.array([
        [37, 48, 118], [39, 127, 158], [53, 184, 153],
        [179, 218, 83], [251, 231, 100],
    ], dtype=np.float64)
    values = np.clip((np.asarray(phase) + np.pi) / (2 * np.pi), 0, 1)
    positions = np.linspace(0, 1, len(stops))
    rgb = np.stack([
        np.interp(values, positions, stops[:, channel]) for channel in range(3)
    ], axis=-1)
    return Image.fromarray(np.rint(rgb).astype(np.uint8))


class ImagePanel(ttk.Frame):
    """Карточка с пиксельным изображением и независимыми подписями."""

    def __init__(self, parent, title: str, *, phase: bool = False):
        super().__init__(parent, style="Card.TFrame", padding=14)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.original_image = None
        self.photo = None
        self.colorbar_photo = None
        self.placeholder = "Матрица ещё не выбрана" if phase else "Ожидание расчёта"
        ttk.Label(self, text=title, style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )
        self.canvas = tk.Canvas(
            self, width=220, height=180, bg=CANVAS,
            highlightthickness=0, borderwidth=0,
        )
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._draw)
        self.metadata = tk.StringVar()
        if phase:
            self.colorbar = tk.Canvas(
                self, height=20, bg=CARD, highlightthickness=0, borderwidth=0
            )
            self.colorbar.grid(row=2, column=0, sticky="ew", pady=(12, 2))
            self.colorbar.bind("<Configure>", self._draw_colorbar)
            labels = ttk.Frame(self, style="Card.TFrame")
            labels.grid(row=3, column=0, sticky="ew")
            for column, text in enumerate(("−π", "0", "+π")):
                labels.columnconfigure(column, weight=1)
                ttk.Label(labels, text=text, style="CardMeta.TLabel").grid(
                    row=0, column=column, sticky=("w", "", "e")[column]
                )
            metadata_row = 4
        else:
            metadata_row = 2
        ttk.Label(
            self, textvariable=self.metadata, style="CardMeta.TLabel",
            justify="left", anchor="w",
        ).grid(row=metadata_row, column=0, sticky="ew", pady=(10, 0))

    def set_image(self, image: Image.Image | None):
        self.original_image = None if image is None else image.copy()
        self._draw()

    def _draw(self, event=None):
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        self.canvas.delete("all")
        if self.original_image is None:
            self.photo = None
            self.canvas.create_text(
                width / 2, height / 2, text=self.placeholder,
                fill=MUTED, font=("Segoe UI", 10), width=max(width - 20, 20),
            )
            return
        side = max(1, min(width, height) - 12)
        self.photo = ImageTk.PhotoImage(
            self.original_image.resize((side, side), Image.Resampling.NEAREST),
            master=self.canvas,
        )
        self.canvas.create_image(width / 2, height / 2, image=self.photo)

    def _draw_colorbar(self, event=None):
        width = max(self.colorbar.winfo_width(), 1)
        gradient = np.linspace(-np.pi, np.pi, 256)[None, :]
        image = phase_rgb(gradient).resize((width, 20), Image.Resampling.NEAREST)
        self.colorbar_photo = ImageTk.PhotoImage(image, master=self.colorbar)
        self.colorbar.delete("all")
        self.colorbar.create_image(0, 0, anchor="nw", image=self.colorbar_photo)


class FresnelApp:
    """Состояние интерфейса: одна фаза и четыре результата её распространения."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.phase = None
        self.result = None
        self._seed = None
        self._source = "random"
        self._busy = False
        self.seed_var = tk.StringVar()
        self.status_var = tk.StringVar(
            value="Сгенерируйте фазовую маску или загрузите PNG / TXT 128×128."
        )
        root.title("Симулятор дифракции Френеля · Фазовый ДОЭ")
        root.geometry("1240x850")
        root.minsize(1000, 720)
        root.configure(bg=BACKGROUND)
        self._style()
        self._build()
        self._update_buttons()

    def _style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BACKGROUND)
        style.configure("TLabel", background=BACKGROUND, foreground=TEXT,
                        font=("Segoe UI", 10))
        style.configure("Heading.TLabel", font=("Segoe UI", 21, "bold"))
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Card.TFrame", background=CARD)
        style.configure("CardTitle.TLabel", background=CARD,
                        font=("Segoe UI", 12, "bold"))
        style.configure("CardMeta.TLabel", background=CARD, foreground=MUTED,
                        font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 9), padding=(10, 9))
        style.configure("Accent.TButton", background=ACCENT, foreground="white")
        style.map("Accent.TButton", background=[("active", "#1a4c7b"),
                                                ("disabled", "#c7d2df")])
        style.configure("TEntry", padding=7, font=("Segoe UI", 10))

    def _build(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        header = ttk.Frame(self.root, padding=(20, 15, 20, 7))
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="Дифракция Френеля", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text=(f"Один фазовый ДОЭ → четыре плоскости · λ = {WAVELENGTH_M * 1e9:.0f} нм"
                  f" · {IMAGE_SIZE}×{IMAGE_SIZE} пикселей"
                  f" · шаг ДОЭ {PIXEL_PITCH_M * 1e6:g} мкм · амплитуда 1"),
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        controls = ttk.Frame(self.root, padding=(20, 8, 20, 12))
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure(3, weight=1)
        self.generate_button = ttk.Button(
            controls, text="Сгенерировать случайную матрицу", command=self.generate_phase
        )
        self.generate_button.grid(row=0, column=0, padx=(0, 8), sticky="w")
        self.load_button = ttk.Button(
            controls, text="Загрузить матрицу фаз PNG", command=self.load_phase
        )
        self.load_button.grid(row=0, column=1, padx=(0, 12), sticky="w")
        self.load_txt_button = ttk.Button(
            controls, text="Загрузить фазы TXT", command=self.load_phase_text
        )
        self.load_txt_button.grid(row=0, column=3, padx=(12, 0), sticky="w")
        seed_frame = ttk.Frame(controls)
        seed_frame.grid(row=0, column=2, sticky="w")
        ttk.Label(seed_frame, text="Seed:").pack(side="left", padx=(0, 5))
        self.seed_entry = ttk.Entry(seed_frame, textvariable=self.seed_var, width=11)
        self.seed_entry.pack(side="left")
        self.seed_entry.bind("<Return>", lambda event: self.generate_phase())
        self.calculate_button = ttk.Button(
            controls, text="Рассчитать дифракцию", style="Accent.TButton",
            command=self.calculate,
        )
        self.calculate_button.grid(row=1, column=0, pady=(9, 0), padx=(0, 8), sticky="ew")
        self.save_button = ttk.Button(
            controls, text="Сохранить результаты", command=self.save_results
        )
        self.save_button.grid(row=1, column=1, pady=(9, 0), padx=(0, 12), sticky="ew")
        ttk.Label(controls, text="Пустой seed — новая случайная маска", style="Muted.TLabel").grid(
            row=1, column=2, columnspan=2, pady=(9, 0), sticky="w"
        )

        plots = ttk.Frame(self.root, padding=(20, 0, 20, 0))
        plots.grid(row=2, column=0, sticky="nsew")
        for column in range(3):
            plots.columnconfigure(column, weight=1, uniform="plot")
        for row in range(2):
            plots.rowconfigure(row, weight=1, uniform="plane")
        self.phase_panel = ImagePanel(plots, "Фаза ДОЭ · рад", phase=True)
        self.phase_panel.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 12))
        self.phase_panel.metadata.set(
            "Фиксированная цветовая шкала [−π, π]\n"
            "Размер ДОЭ: 1,28 × 1,28 мм\n"
            "Столбцы → u, строки → v"
        )
        self.screen_panels = []
        for index, distance in enumerate(RECONSTRUCTION_DISTANCES_M):
            distance_text = f"{distance * 1e3:.1f}".replace(".", ",")
            panel = ImagePanel(plots, f"Экран {index + 1} · z = {distance_text} мм")
            panel.grid(
                row=index // 2, column=1 + index % 2, sticky="nsew",
                padx=(0, 12 if index % 2 == 0 else 0),
                pady=(0, 12 if index < 2 else 0),
            )
            self._set_screen_metadata(panel, distance)
            self.screen_panels.append(panel)

        footer = ttk.Frame(self.root, padding=(20, 10, 20, 12))
        footer.grid(row=3, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(
            footer,
            text="Яркость каждого экрана нормирована по своему максимуму только для показа. NPY сохраняет исходную интенсивность.",
            style="Muted.TLabel", wraplength=1150,
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            footer,
            text="Модель: одно FFT на плоскость. Дискретизация близка к пределу; возможен алиасинг (см. README).",
            style="Muted.TLabel", wraplength=1150,
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))
        self.status_label = ttk.Label(
            footer, textvariable=self.status_var, wraplength=1150,
        )
        self.status_label.grid(row=2, column=0, sticky="ew", pady=(7, 0))
        footer.bind("<Configure>", self._wrap_footer)
        self._footer = footer

    def _wrap_footer(self, event):
        for label in event.widget.winfo_children():
            if isinstance(label, ttk.Label):
                label.configure(wraplength=max(300, event.width - 40))

    def _set_screen_metadata(self, panel, distance, peak=None):
        pitch = output_pixel_pitch(distance)
        size_mm = IMAGE_SIZE * pitch * 1e3
        peak_text = "—" if peak is None else f"{peak:.6g}"
        panel.metadata.set(
            f"Шаг экрана: {pitch * 1e6:.5f} мкм\n"
            f"Область: {size_mm:.5f} × {size_mm:.5f} мм\n"
            f"Imax до нормировки: {peak_text}"
        )

    def _update_buttons(self):
        normal = "disabled" if self._busy else "normal"
        self.generate_button.configure(state=normal)
        self.load_button.configure(state=normal)
        self.load_txt_button.configure(state=normal)
        self.seed_entry.configure(state=normal)
        self.calculate_button.configure(
            state="normal" if self.phase is not None and not self._busy else "disabled"
        )
        self.save_button.configure(
            state="normal" if self.result is not None and not self._busy else "disabled"
        )

    def _set_busy(self, busy):
        self._busy = busy
        self.root.configure(cursor="watch" if busy else "")
        self._update_buttons()
        self.root.update_idletasks()

    def _choose_phase(self, phase, seed, source):
        self.phase = phase
        self._seed = seed
        self._source = source
        self.result = None
        self.phase_panel.set_image(phase_rgb(phase))
        for panel, distance in zip(self.screen_panels, RECONSTRUCTION_DISTANCES_M):
            panel.set_image(None)
            self._set_screen_metadata(panel, distance)
        self._update_buttons()

    def generate_phase(self):
        """Создать маску; неверный seed не изменяет текущие результаты."""
        if self._busy:
            return
        text = self.seed_var.get().strip()
        try:
            seed = int(text) if text else None
            if seed is not None and seed < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Некорректный seed",
                "Seed должен быть целым неотрицательным числом или пустым полем.",
                parent=self.root,
            )
            return
        try:
            phase = generate_random_phase(seed)
        except (ValueError, TypeError) as error:
            messagebox.showerror("Ошибка генерации", str(error), parent=self.root)
            return
        self._choose_phase(phase, seed, "random")
        suffix = "без seed" if seed is None else f"seed = {seed}"
        self.status_var.set(f"Случайная матрица создана ({suffix}). Нажмите «Рассчитать дифракцию».")

    def load_phase(self):
        """Строго загрузить PNG; ошибка или отмена сохраняет текущую фазу."""
        if self._busy:
            return
        filename = filedialog.askopenfilename(
            parent=self.root, title="Фазовая маска: PNG, 128×128, 8 бит, один канал",
            filetypes=[("Фазовая маска PNG", "*.png"), ("Все файлы", "*.*")],
        )
        if not filename:
            return
        try:
            phase = load_phase_png(filename)
        except (ValueError, OSError) as error:
            messagebox.showerror("Не удалось загрузить маску", str(error), parent=self.root)
            return
        self._choose_phase(phase, None, str(Path(filename).resolve()))
        self.seed_var.set("")
        self.status_var.set(f"Загружена маска {Path(filename).name}. Нажмите «Рассчитать дифракцию».")

    def load_phase_text(self):
        """Загрузить TXT-фазу без потери точности и сразу рассчитать дифракцию."""
        if self._busy:
            return
        filename = filedialog.askopenfilename(
            parent=self.root, title="Фазы TXT: 128×128 чисел в радианах [−π, π]",
            filetypes=[("Матрица фаз TXT", "*.txt"), ("Все файлы", "*.*")],
        )
        if not filename:
            return
        try:
            phase = load_phase_txt(filename)
        except (ValueError, OSError) as error:
            messagebox.showerror("Не удалось загрузить TXT", str(error), parent=self.root)
            return
        self._choose_phase(phase, None, str(Path(filename).resolve()))
        self.seed_var.set("")
        self.calculate()

    def calculate(self):
        """Одна фаза → одно входное поле → четыре поля и интенсивности."""
        if self.phase is None or self._busy:
            return
        self.status_var.set("Расчёт четырёх плоскостей…")
        self._set_busy(True)
        try:
            result = simulate(self.phase, seed=self._seed, source=self._source)
            pixels = OutputPictures(Path("app"), result.intensities).to_uint8()
            for index, (panel, distance) in enumerate(
                zip(self.screen_panels, RECONSTRUCTION_DISTANCES_M)
            ):
                panel.set_image(Image.fromarray(pixels[index]))
                self._set_screen_metadata(panel, distance, result.intensities[index].max())
            self.result = result
            self.status_var.set(
                "Готово: одна маска, четыре плоскости. Исходные интенсивности не нормированы."
            )
        except (ValueError, TypeError, OSError, ArithmeticError) as error:
            self.status_var.set("Расчёт не выполнен. Проверьте входную матрицу.")
            messagebox.showerror("Ошибка расчёта", str(error), parent=self.root)
        finally:
            self._set_busy(False)

    def save_results(self):
        """Экспортировать новый набор, не перезаписывая прежние расчёты."""
        if self.result is None or self._busy:
            return
        parent = filedialog.askdirectory(
            parent=self.root, title="Выберите папку: результаты будут в новой подпапке",
            mustexist=True,
        )
        if not parent:
            return
        name = datetime.now().strftime("fresnel_%Y%m%d_%H%M%S_%f")
        directory = Path(parent) / name
        self.status_var.set("Сохранение исходных массивов, PNG и метаданных…")
        self._set_busy(True)
        try:
            saved = export_results(self.result, directory)
            self.status_var.set(f"Результаты сохранены: {saved}")
            messagebox.showinfo(
                "Результаты сохранены",
                f"Папка: {saved}\n\n"
                "Сохранены фаза, четыре PNG, исходные интенсивности NPY и JSON.\n"
                "Дополнительно рассчитаны восстановления из сохранённого 8-битного ДОЭ.",
                parent=self.root,
            )
        except (ValueError, TypeError, OSError, ArithmeticError) as error:
            self.status_var.set("Не удалось сохранить результаты.")
            messagebox.showerror("Ошибка сохранения", str(error), parent=self.root)
        finally:
            self._set_busy(False)


def create_root():
    """Создать окно. Импорт модуля подготавливает Tcl, но не создаёт Tk."""
    return tk.Tk()


def main():
    root = create_root()
    FresnelApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
