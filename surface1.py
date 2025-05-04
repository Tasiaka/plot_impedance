import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Импорты для Matplotlib 3D и работы с геометрией
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d # Для ручного добавления 3D коллекций

# Импорты для интерполяции, поиска соседей и фильтрации
import scipy # Импортируем всю библиотеку scipy для доступа к подмодулям
from scipy.interpolate import griddata
from scipy.spatial import KDTree, Delaunay # Более конкретные импорты для ясности
from scipy.ndimage import gaussian_filter

# Импорты для PyQt6 GUI
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QPushButton,
                             QFileDialog, QMessageBox, QDoubleSpinBox, QSpinBox,
                             QGroupBox, QFormLayout)

# Импорты для встраивания Matplotlib в PyQt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class ImpedancePlotterApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Magnetic Impedance 3D Plotter (Объемный вид)")
        self.setGeometry(100, 100, 1200, 800)

        self.data = None
        self.cbar = None # Инициализируем атрибут colorbar

        self._create_widgets()
        self._create_layouts()
        self._connect_signals()
        self._setup_matplotlib()

        self._set_column_selection_enabled(False)
        self.plot_button.setEnabled(False)


    def _create_widgets(self):
        """Создание всех виджетов GUI."""
        self.file_label = QLabel("Файл: Не загружен")
        self.load_button = QPushButton("Загрузить CSV файл")

        self.column_selection_group = QGroupBox("Выбор колонок данных")
        self.column_layout = QFormLayout()
        self.x_label = QLabel("Ось X:")
        self.x_combo = QComboBox()
        self.y_label = QLabel("Ось Y:")
        self.y_combo = QComboBox()
        self.real_z_label = QLabel("Действительная часть Z:")
        self.real_z_combo = QComboBox()
        self.imag_z_label = QLabel("Мнимая часть Z:")
        self.imag_z_combo = QComboBox()
        self.column_layout.addRow(self.x_label, self.x_combo)
        self.column_layout.addRow(self.y_label, self.y_combo)
        self.column_layout.addRow(self.real_z_label, self.real_z_combo)
        self.column_layout.addRow(self.imag_z_label, self.imag_z_combo)
        self.column_selection_group.setLayout(self.column_layout)

        self.processing_params_group = QGroupBox("Параметры обработки")
        self.processing_layout = QFormLayout()

        self.grid_res_label = QLabel("Разрешение сетки:")
        self.grid_res_spinbox = QSpinBox()
        self.grid_res_spinbox.setRange(50, 500)
        self.grid_res_spinbox.setValue(150)
        self.processing_layout.addRow(self.grid_res_label, self.grid_res_spinbox)

        self.outlier_k_label = QLabel("Соседей (выбросы):")
        self.outlier_k_spinbox = QSpinBox()
        self.outlier_k_spinbox.setRange(5, 50)
        self.outlier_k_spinbox.setValue(15)
        self.processing_layout.addRow(self.outlier_k_label, self.outlier_k_spinbox)

        self.outlier_std_label = QLabel("Порог стд. откл. (выбросы):")
        self.outlier_std_spinbox = QDoubleSpinBox()
        self.outlier_std_spinbox.setRange(1.0, 10.0)
        self.outlier_std_spinbox.setSingleStep(0.1)
        self.outlier_std_spinbox.setValue(3.0)
        self.processing_layout.addRow(self.outlier_std_label, self.outlier_std_spinbox)

        self.smoothing_sigma_label = QLabel("Сигма сглаживания:")
        self.smoothing_sigma_spinbox = QDoubleSpinBox()
        self.smoothing_sigma_spinbox.setRange(0.1, 10.0)
        self.smoothing_sigma_spinbox.setSingleStep(0.1)
        self.smoothing_sigma_spinbox.setValue(2.0)
        self.processing_layout.addRow(self.smoothing_sigma_label, self.smoothing_sigma_spinbox)

        self.processing_params_group.setLayout(self.processing_layout)
        self.processing_params_group.setEnabled(False)

        self.plot_button = QPushButton("Построить 3D график")

    def _create_layouts(self):
        """Размещение виджетов в слоях."""
        control_layout = QVBoxLayout()
        control_layout.addWidget(self.file_label)
        control_layout.addWidget(self.load_button)
        control_layout.addSpacing(15)
        control_layout.addWidget(self.column_selection_group)
        control_layout.addSpacing(15)
        control_layout.addWidget(self.processing_params_group)
        control_layout.addSpacing(15)
        control_layout.addWidget(self.plot_button)
        control_layout.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addLayout(control_layout, 1)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.main_layout = main_layout

    def _connect_signals(self):
        """Подключение сигналов виджетов к слотам (методам)."""
        self.load_button.clicked.connect(self._load_csv_file)
        self.plot_button.clicked.connect(self._update_plot)

    def _setup_matplotlib(self):
        """Настройка области для Matplotlib графика."""
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        self.main_layout.addLayout(plot_layout, 3)
        self.ax = self.figure.add_subplot(111, projection='3d')

    def _set_column_selection_enabled(self, enabled):
        """Включает/выключает виджеты выбора колонок."""
        self.column_selection_group.setEnabled(enabled)
        self.processing_params_group.setEnabled(enabled)

    def _load_csv_file(self):
        """Открывает диалог выбора файла, читает CSV и заполняет комбобоксы."""
        filepath, _ = QFileDialog.getOpenFileName(self, "Выбрать CSV файл", "", "CSV Files (*.csv);;All Files (*)")
        if not filepath: return

        try:
            self.data = pd.read_csv(filepath)
            self.file_label.setText(f"Файл: {os.path.basename(filepath)}")

            self.x_combo.clear(); self.y_combo.clear(); self.real_z_combo.clear(); self.imag_z_combo.clear()
            columns = self.data.columns.tolist()
            self.x_combo.addItems(columns); self.y_combo.addItems(columns); self.real_z_combo.addItems(columns); self.imag_z_combo.addItems(columns)

            col_mapping = {
                self.x_combo: ['Frequency', 'Freq', 'Частота', 'Hz', 'frequency'],
                self.y_combo: ['Magnetic field', 'Field', 'Magnet', 'Поле', 'Oe', 'magnetic field'],
                self.real_z_combo: ['Real Impedance', 'ReZ', 'Real', 'Real Part', 'real'],
                self.imag_z_combo: ['Imaginary Impedance', 'ImZ', 'Imaginary', 'Imag Part', 'imaginary']
            }
            lower_cols = [c.lower() for c in columns]
            for combo, possible_names in col_mapping.items():
                for name in possible_names:
                    if name.lower() in lower_cols:
                        original_name = columns[lower_cols.index(name.lower())]
                        combo.setCurrentText(original_name)
                        break

            self._set_column_selection_enabled(True)
            self.plot_button.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки файла", f"Не удалось загрузить файл:\n{e}")
            self.data = None; self.file_label.setText("Файл: Ошибка загрузки")
            self._set_column_selection_enabled(False); self.plot_button.setEnabled(False)

    def _update_plot(self):
        """Считывает выбранные колонки, обрабатывает данные и строит 3D график."""
        if self.data is None:
            QMessageBox.warning(self, "Нет данных", "Пожалуйста, сначала загрузите CSV файл.")
            return

        x_col_name = self.x_combo.currentText(); y_col_name = self.y_combo.currentText()
        real_z_col_name = self.real_z_combo.currentText(); imag_z_col_name = self.imag_z_combo.currentText()

        try:
            grid_resolution = self.grid_res_spinbox.value()
            k_neighbors = self.outlier_k_spinbox.value()
            outlier_std_multiplier = self.outlier_std_spinbox.value()
            smoothing_sigma = self.smoothing_sigma_spinbox.value()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка параметров", f"Некорректные значения параметров обработки:\n{e}")
            return

        if x_col_name == y_col_name:
             QMessageBox.warning(self, "Ошибка выбора", "Колонки для X и Y должны быть разными.")
             return

        try:
            x_data = pd.to_numeric(self.data[x_col_name], errors='coerce').values
            y_data = pd.to_numeric(self.data[y_col_name], errors='coerce').values
            real_z_data = pd.to_numeric(self.data[real_z_col_name], errors='coerce').values
            imag_z_data = pd.to_numeric(self.data[imag_z_col_name], errors='coerce').values
        except KeyError as e: QMessageBox.critical(self, "Ошибка данных", f"Не удалось найти колонку: {e}"); return
        except Exception as e: QMessageBox.critical(self, "Ошибка данных", f"Ошибка при извлечении или преобразовании данных:\n{e}"); return

        z_magnitude = np.sqrt(real_z_data**2 + imag_z_data**2)

        combined_data = np.vstack((x_data, y_data, z_magnitude)).T
        finite_mask = np.all(np.isfinite(combined_data), axis=1)
        x_data_clean = x_data[finite_mask]; y_data_clean = y_data[finite_mask]; z_magnitude_clean = z_magnitude[finite_mask]

        if len(x_data_clean) < 3:
            QMessageBox.warning(self, "Недостаточно данных", "После первичной очистки осталось слишком мало точек.")
            self.ax.cla()
            # --- Надежное удаление colorbar ---
            if hasattr(self, 'cbar') and self.cbar is not None:
                try: self.cbar.remove(); print("Previous colorbar removed.")
                except (ValueError, AttributeError, KeyError) as e: print(f"Note: Error removing previous colorbar ({type(e).__name__}).")
                finally: self.cbar = None
            # --- Конец блока удаления ---
            self.canvas.draw(); return

        # --- Удаление выбросов ---
        print(f"Начато удаление выбросов: {len(x_data_clean)} точек...")
        points = np.vstack((x_data_clean, y_data_clean)).T
        try:
            tree = KDTree(points)
            try: dists, indices = tree.query(points, k=k_neighbors + 1, workers=-1)
            except TypeError: print("Параллельный поиск соседей не поддерживается, используется одно ядро."); dists, indices = tree.query(points, k=k_neighbors + 1)
        except ValueError as e: QMessageBox.critical(self, "Ошибка KDTree", f"Ошибка при построении KDTree: {e}\nВозможно, все точки лежат на одной прямой."); return
        except Exception as e: QMessageBox.critical(self, "Ошибка KDTree", f"Неожиданная ошибка при поиске соседей: {e}"); return

        is_outlier = np.zeros(len(z_magnitude_clean), dtype=bool)
        for i in range(len(z_magnitude_clean)):
            neighbor_indices = indices[i, 1:]; neighbor_z_values = z_magnitude_clean[neighbor_indices]
            finite_neighbor_z = neighbor_z_values[np.isfinite(neighbor_z_values)]
            if len(finite_neighbor_z) < 2: continue
            mean_neighbor_z = np.mean(finite_neighbor_z); std_neighbor_z = np.std(finite_neighbor_z)
            if std_neighbor_z < 1e-9:
                 if abs(z_magnitude_clean[i] - mean_neighbor_z) > 1e-9: is_outlier[i] = True
            else:
                 z_score_relative = abs(z_magnitude_clean[i] - mean_neighbor_z) / std_neighbor_z
                 if z_score_relative > outlier_std_multiplier: is_outlier[i] = True

        x_data_filtered = x_data_clean[~is_outlier]; y_data_filtered = y_data_clean[~is_outlier]; z_magnitude_filtered = z_magnitude_clean[~is_outlier]
        print(f"Удалено выбросов: {np.sum(is_outlier)}. Осталось точек: {len(x_data_filtered)}")

        if len(x_data_filtered) < 3:
             QMessageBox.warning(self, "Недостаточно данных", f"После очистки выбросов осталось слишком мало точек ({len(x_data_filtered)}).")
             self.ax.cla()
             # --- Надежное удаление colorbar ---
             if hasattr(self, 'cbar') and self.cbar is not None:
                 try: self.cbar.remove(); print("Previous colorbar removed.")
                 except (ValueError, AttributeError, KeyError) as e: print(f"Note: Error removing previous colorbar ({type(e).__name__}).")
                 finally: self.cbar = None
             # --- Конец блока удаления ---
             self.canvas.draw(); return

        # --- Интерполяция ---
        print(f"Начата интерполяция на сетку {grid_resolution}x{grid_resolution}...")
        xi_min, xi_max = x_data_filtered.min(), x_data_filtered.max(); yi_min, yi_max = y_data_filtered.min(), y_data_filtered.max()
        xi = np.linspace(xi_min, xi_max, grid_resolution); yi = np.linspace(yi_min, yi_max, grid_resolution)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x_data_filtered, y_data_filtered), z_magnitude_filtered, (xi, yi), method='cubic')

        # --- Сглаживание ---
        print(f"Начато сглаживание с sigma={smoothing_sigma}...")
        zi_smoothed = gaussian_filter(zi, sigma=smoothing_sigma, mode='nearest')

        # --- Построение 3D графика (Объемный вид) ---
        print("Начато построение 3D графика...")
        self.ax.cla() # Очищаем оси перед рисованием

        cmap_name = 'viridis'
        cmap = plt.get_cmap(cmap_name)

        finite_zi_smoothed = zi_smoothed[np.isfinite(zi_smoothed)]
        if len(finite_zi_smoothed) > 0:
            z_min_top = np.nanmin(zi_smoothed); z_max_top = np.nanmax(zi_smoothed)
            z_range = z_max_top - z_min_top
            thickness = max(z_range * 0.1, np.abs(z_min_top * 0.1) + 1e-6)
        else: z_min_top = 0; z_max_top = 1; thickness = 1

        zi_bottom = zi_smoothed - thickness
        zi_bottom[np.isnan(zi_smoothed)] = np.nan

        norm = plt.Normalize(vmin=z_min_top, vmax=z_max_top)

        # Рисуем ВЕРХНЮЮ поверхность
        surf_top = self.ax.plot_surface(xi, yi, zi_smoothed, cmap=cmap_name, antialiased=True,
                                        shade=True, alpha=1.0, norm=norm, rstride=1, cstride=1)

        # Рисуем НИЖНЮЮ поверхность
        surf_bottom = self.ax.plot_surface(xi, yi, zi_bottom, cmap=cmap_name, antialiased=True,
                                           shade=True, alpha=1.0, norm=norm, rstride=1, cstride=1)

        # --- Создание боковых стенок ---
        manual_verts = []
        edge_vertex_indices_top = {}
        rows, cols = zi_smoothed.shape
        manual_faces = [] # Инициализация здесь

        def add_side_face(p_top1, p_top2, p_bottom1, p_bottom2, is_reverse_order=False):
            verts_to_add = [p_top1, p_top2, p_bottom2, p_bottom1]
            if is_reverse_order: verts_to_add = [p_top1, p_bottom1, p_bottom2, p_top2]
            if not all(np.all(np.isfinite(p)) for p in verts_to_add): return

            start_idx = len(manual_verts)
            manual_verts.extend(verts_to_add)
            idx1, idx2, idx3, idx4 = start_idx, start_idx + 1, start_idx + 2, start_idx + 3
            face1_indices = [idx1, idx2, idx3]; face2_indices = [idx1, idx3, idx4]
            manual_faces.append(face1_indices); manual_faces.append(face2_indices)
            top_v_indices = (idx1, idx2)
            edge_vertex_indices_top[tuple(sorted(face1_indices))] = top_v_indices
            edge_vertex_indices_top[tuple(sorted(face2_indices))] = top_v_indices

        # Края
        for j in range(cols - 1): # Верхний
            if np.isfinite(zi_smoothed[0, j]) and np.isfinite(zi_smoothed[0, j+1]):
                add_side_face(p_top1=[xi[0, j], yi[0, j], zi_smoothed[0, j]],
                              p_top2=[xi[0, j+1], yi[0, j+1], zi_smoothed[0, j+1]],
                              p_bottom1=[xi[0, j], yi[0, j], zi_bottom[0, j]],
                              p_bottom2=[xi[0, j+1], yi[0, j+1], zi_bottom[0, j+1]])
        for j in range(cols - 1): # Нижний
            if np.isfinite(zi_smoothed[rows-1, j]) and np.isfinite(zi_smoothed[rows-1, j+1]):
                 add_side_face(p_top1=[xi[rows-1, j], yi[rows-1, j], zi_smoothed[rows-1, j]],
                              p_top2=[xi[rows-1, j+1], yi[rows-1, j+1], zi_smoothed[rows-1, j+1]],
                              p_bottom1=[xi[rows-1, j], yi[rows-1, j], zi_bottom[rows-1, j]],
                              p_bottom2=[xi[rows-1, j+1], yi[rows-1, j+1], zi_bottom[rows-1, j+1]],
                              is_reverse_order=True)
        for i in range(rows - 1): # Левый
            if np.isfinite(zi_smoothed[i, 0]) and np.isfinite(zi_smoothed[i+1, 0]):
                 add_side_face(p_top1=[xi[i, 0], yi[i, 0], zi_smoothed[i, 0]],
                              p_top2=[xi[i+1, 0], yi[i+1, 0], zi_smoothed[i+1, 0]],
                              p_bottom1=[xi[i, 0], yi[i, 0], zi_bottom[i, 0]],
                              p_bottom2=[xi[i+1, 0], yi[i+1, 0], zi_bottom[i+1, 0]])
        for i in range(rows - 1): # Правый
             if np.isfinite(zi_smoothed[i, cols-1]) and np.isfinite(zi_smoothed[i+1, cols-1]):
                  add_side_face(p_top1=[xi[i, cols-1], yi[i, cols-1], zi_smoothed[i, cols-1]],
                              p_top2=[xi[i+1, cols-1], yi[i+1, cols-1], zi_smoothed[i+1, cols-1]],
                              p_bottom1=[xi[i, cols-1], yi[i, cols-1], zi_bottom[i, cols-1]],
                              p_bottom2=[xi[i+1, cols-1], yi[i+1, cols-1], zi_bottom[i+1, cols-1]],
                              is_reverse_order=True)

        # Создание полигонов и цветов для боковых стенок
        side_polygons = []
        side_face_colors = []
        if manual_verts and manual_faces:
            for face_indices in manual_faces:
                triangle = [manual_verts[i] for i in face_indices]
                side_polygons.append(triangle)
                top_indices = edge_vertex_indices_top.get(tuple(sorted(face_indices)))
                if top_indices:
                    avg_z = np.mean([manual_verts[ti][2] for ti in top_indices])
                    side_face_colors.append(cmap(norm(avg_z)))
                else: side_face_colors.append('grey')

        # Добавление боковых стенок
        if side_polygons and len(side_polygons) == len(side_face_colors):
             side_collection = art3d.Poly3DCollection(side_polygons, facecolors=side_face_colors,
                                                       alpha=1.0, shade=False, antialiased=True)
             self.ax.add_collection3d(side_collection)
        elif side_polygons: print("Warning: Mismatch side polygons/colors.")

        # --- Настройка осей и заголовка ---
        z_combined_min = np.nanmin(zi_bottom) if np.any(np.isfinite(zi_bottom)) else z_min_top
        z_combined_max = np.nanmax(zi_smoothed) if np.any(np.isfinite(zi_smoothed)) else z_combined_min + 1
        z_buffer = (z_combined_max - z_combined_min) * 0.05
        self.ax.set_zlim(z_combined_min - z_buffer, z_combined_max + z_buffer)

        self.ax.set_xlabel(x_col_name); self.ax.set_ylabel(y_col_name)
        self.ax.set_zlabel(f'Модуль магнитного импеданса |Z|')
        try: filename = os.path.basename(self.file_label.text().split(": ")[1])
        except IndexError: filename = "Неизвестный файл"
        self.ax.set_title(f'3D график |Z| от {x_col_name} и {y_col_name}\n(Файл: {filename}, Sigma: {smoothing_sigma:.2f})')

        # --- Добавление/обновление цветовой шкалы ---
        # --- Надежное удаление colorbar ---
        if hasattr(self, 'cbar') and self.cbar is not None:
            try: self.cbar.remove(); print("Previous colorbar removed.")
            except (ValueError, AttributeError, KeyError) as e: print(f"Note: Error removing previous colorbar ({type(e).__name__}).")
            finally: self.cbar = None
        # --- Конец блока удаления ---

        print("Creating new colorbar.") # Отладочное сообщение
        self.cbar = self.figure.colorbar(surf_top, ax=self.ax, shrink=0.6, aspect=10, label='Модуль магнитного импеданса |Z|', norm=norm)

        self.ax.view_init(elev=25, azim=-110)
        self.canvas.draw()

# --- Основная часть приложения ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = ImpedancePlotterApp()
    main_window.show()
    sys.exit(app.exec())