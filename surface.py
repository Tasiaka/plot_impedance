import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d

import scipy 
from scipy.interpolate import griddata
from scipy.spatial import KDTree, Delaunay
from scipy.ndimage import gaussian_filter

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QPushButton,
                             QFileDialog, QMessageBox, QDoubleSpinBox, QSpinBox,
                             QGroupBox, QFormLayout)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure

class ImpedancePlotterApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Magnetic Impedance 3D Plotter")
        self.setGeometry(100, 100, 1200, 800) # Увеличиваем размер окна

        self.data = None # Для хранения загруженных данных DataFrame

        self._create_widgets()
        self._create_layouts()
        self._connect_signals()
        self._setup_matplotlib()

        # Изначально кнопки и комбобоксы выбора колонок неактивны
        self._set_column_selection_enabled(False)
        self.plot_button.setEnabled(False)


    def _create_widgets(self):
        """Создание всех виджетов GUI."""
        self.file_label = QLabel("Файл: Не загружен")
        self.load_button = QPushButton("Загрузить CSV файл")

        # Группа для выбора колонок
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


        # Группа для параметров обработки данных
        self.processing_params_group = QGroupBox("Параметры обработки")
        self.processing_layout = QFormLayout()

        self.grid_res_label = QLabel("Разрешение сетки:")
        self.grid_res_spinbox = QSpinBox()
        self.grid_res_spinbox.setRange(50, 500) # Минимальное и максимальное разрешение
        self.grid_res_spinbox.setValue(150)    # Значение по умолчанию
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
        self.processing_params_group.setEnabled(False) # Изначально неактивны


        self.plot_button = QPushButton("Построить 3D график")

    def _create_layouts(self):
        """Размещение виджетов в слоях."""
        control_layout = QVBoxLayout()
        control_layout.addWidget(self.file_label)
        control_layout.addWidget(self.load_button)
        control_layout.addSpacing(15) # Добавляем пространство

        control_layout.addWidget(self.column_selection_group)
        control_layout.addSpacing(15)

        control_layout.addWidget(self.processing_params_group)
        control_layout.addSpacing(15)

        control_layout.addWidget(self.plot_button)
        control_layout.addStretch() # Растягивает пространство снизу

        # Основной слой для размещения контролов и графика
        main_layout = QHBoxLayout()
        main_layout.addLayout(control_layout, 1) # Контролы занимают 1 часть пространства

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.main_layout = main_layout # Сохраняем ссылку на основной слой

    def _connect_signals(self):
        """Подключение сигналов виджетов к слотам (методам)."""
        self.load_button.clicked.connect(self._load_csv_file)
        self.plot_button.clicked.connect(self._update_plot)

    def _setup_matplotlib(self):
        """Настройка области для Matplotlib графика."""
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Добавляем тулбар и канвас графика в основной слой
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        self.main_layout.addLayout(plot_layout, 3) # График занимает 3 части пространства

        self.ax = self.figure.add_subplot(111, projection='3d') # Создаем 3D оси

    def _set_column_selection_enabled(self, enabled):
        """Включает/выключает виджеты выбора колонок."""
        self.column_selection_group.setEnabled(enabled)
        self.processing_params_group.setEnabled(enabled)


    def _load_csv_file(self):
        """Открывает диалог выбора файла, читает CSV и заполняет комбобоксы."""
        # Открываем диалог выбора файла
        filepath, _ = QFileDialog.getOpenFileName(self, "Выбрать CSV файл", "", "CSV Files (*.csv);;All Files (*)")

        if not filepath: # Если файл не выбран
            return

        try:
            self.data = pd.read_csv(filepath)
            self.file_label.setText(f"Файл: {os.path.basename(filepath)}")

            # Очищаем предыдущие элементы в комбобоксах
            self.x_combo.clear()
            self.y_combo.clear()
            self.real_z_combo.clear()
            self.imag_z_combo.clear()

            # Получаем список колонок
            columns = self.data.columns.tolist()

            # Заполняем комбобоксы именами колонок
            self.x_combo.addItems(columns)
            self.y_combo.addItems(columns)
            self.real_z_combo.addItems(columns)
            self.imag_z_combo.addItems(columns)

            # Попытка угадать стандартные колонки и выбрать их
            # (можно адаптировать)
            col_mapping = {
                self.x_combo: ['Frequency', 'Freq', 'Частота', 'Hz', 'frequency'],
                self.y_combo: ['Magnetic field', 'Field', 'Magnet', 'Поле', 'Oe', 'magnetic field'],
                self.real_z_combo: ['Real Impedance', 'ReZ', 'Real', 'Real Part', 'real'],
                self.imag_z_combo: ['Imaginary Impedance', 'ImZ', 'Imaginary', 'Imag Part', 'imaginary']
            }

            for combo, possible_names in col_mapping.items():
                # Приводим колонки данных к нижнему регистру для надежного поиска
                lower_cols = [c.lower() for c in columns]
                for name in possible_names:
                    if name.lower() in lower_cols:
                        # Выбираем оригинальное название колонки, а не нижний регистр
                        original_name = columns[lower_cols.index(name.lower())]
                        combo.setCurrentText(original_name)
                        break # Выбираем первое совпадение

            # Включаем выбор колонок, параметры обработки и кнопку построения
            self._set_column_selection_enabled(True)
            self.plot_button.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки файла", f"Не удалось загрузить файл:\n{e}")
            self.data = None
            self.file_label.setText("Файл: Ошибка загрузки")
            self._set_column_selection_enabled(False)
            self.plot_button.setEnabled(False)


    def _update_plot(self):
        """Считывает выбранные колонки, обрабатывает данные и строит 3D график."""
        if self.data is None:
            QMessageBox.warning(self, "Нет данных", "Пожалуйста, сначала загрузите CSV файл.")
            return

        # Считываем выбранные имена колонок
        x_col_name = self.x_combo.currentText()
        y_col_name = self.y_combo.currentText()
        real_z_col_name = self.real_z_combo.currentText()
        imag_z_col_name = self.imag_z_combo.currentText()

        # Считываем параметры обработки
        try:
            grid_resolution = self.grid_res_spinbox.value()
            k_neighbors = self.outlier_k_spinbox.value()
            outlier_std_multiplier = self.outlier_std_spinbox.value()
            smoothing_sigma = self.smoothing_sigma_spinbox.value()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка параметров", f"Некорректные значения параметров обработки:\n{e}")
            return


        # Проверяем, что выбраны разные колонки для X и Y (опционально, но полезно)
        if x_col_name == y_col_name:
             QMessageBox.warning(self, "Ошибка выбора", "Колонки для X и Y должны быть разными.")
             return

        # Извлекаем данные
        try:
            # Убедимся, что данные числовые, пробуем преобразовать
            x_data = pd.to_numeric(self.data[x_col_name], errors='coerce').values
            y_data = pd.to_numeric(self.data[y_col_name], errors='coerce').values
            real_z_data = pd.to_numeric(self.data[real_z_col_name], errors='coerce').values
            imag_z_data = pd.to_numeric(self.data[imag_z_col_name], errors='coerce').values
        except KeyError as e:
            QMessageBox.critical(self, "Ошибка данных", f"Не удалось найти колонку: {e}")
            return
        except Exception as e:
            QMessageBox.critical(self, "Ошибка данных", f"Ошибка при извлечении или преобразовании данных:\n{e}")
            return

        # Расчет модуля магнитного импеданса
        z_magnitude = np.sqrt(real_z_data**2 + imag_z_data**2)

        # --- Подготовка данных (удаление NaN, бесконечностей, которые могли появиться при to_numeric) ---
        # Объединяем все данные и удаляем строки, где есть NaN или inf
        combined_data = np.vstack((x_data, y_data, z_magnitude)).T
        finite_mask = np.all(np.isfinite(combined_data), axis=1)
        x_data_clean = x_data[finite_mask]
        y_data_clean = y_data[finite_mask]
        z_magnitude_clean = z_magnitude[finite_mask]

        if len(x_data_clean) < 3: # Для триангуляции или KDTree нужно хотя бы 3 точки
            QMessageBox.warning(self, "Недостаточно данных", "После первичной очистки данных осталось слишком мало допустимых точек для обработки.")
            # Очистка графика и выход
            self.ax.cla()
            if hasattr(self, 'cbar') and self.cbar is not None:
                try: self.cbar.remove()
                except ValueError: pass
                del self.cbar
            self.canvas.draw()
            return

        # --- Удаление выбросов (Outlier Removal) ---
        # Используем k-ближайших соседей и анализ локального стандартного отклонения
        print(f"Начато удаление выбросов: {len(x_data_clean)} точек...")

        # Строим KD-дерево для быстрого поиска соседей на плоскости XY
        points = np.vstack((x_data_clean, y_data_clean)).T
        tree = KDTree(points)

        # Маска для определения выбросов
        is_outlier = np.zeros(len(z_magnitude_clean), dtype=bool)

        # Находим k+1 ближайших соседей для каждой точки (k соседей + сама точка)
        # Использование workers=-1 задействует все доступные ядра процессора для ускорения
        try:
            # query возвращает расстояния и индексы
            dists, indices = tree.query(points, k=k_neighbors + 1, workers=-1)
        except Exception as e:
            print(f"Ошибка при поиске соседей KDTree с workers: {e}. Попытка без параллелизации.")
            # Если с workers ошибка, попробовать без параллелизации
            dists, indices = tree.query(points, k=k_neighbors + 1)

        # Проходим по каждой точке для анализа ее окрестности
        for i in range(len(z_magnitude_clean)):
            # Получаем индексы соседей (исключая саму точку - индекс 0 в результате query)
            neighbor_indices = indices[i, 1:]

            # Получаем значения Z для соседей
            neighbor_z_values = z_magnitude_clean[neighbor_indices]

            # Исключаем NaN из значений соседей (на всякий случай)
            finite_neighbor_z = neighbor_z_values[np.isfinite(neighbor_z_values)]

            if len(finite_neighbor_z) < 2: # Недостаточно конечных соседей для надежного анализа (нужно хотя бы 2 для стд откл)
                continue # is_outlier[i] остается False, точку оставляем

            # Считаем среднее и стандартное отклонение Z для соседей
            mean_neighbor_z = np.mean(finite_neighbor_z)
            std_neighbor_z = np.std(finite_neighbor_z)

            # Проверка условия выброса: значение точки сильно отличается от среднего соседей
            # Если стандартное отклонение очень близко к нулю (соседи почти одинаковы),
            # проверяем, отличается ли текущая точка от этого постоянного значения.
            if std_neighbor_z < 1e-9: # Порог для "очень близко к нулю"
                if abs(z_magnitude_clean[i] - mean_neighbor_z) > 1e-9: # Отличается от постоянного значения соседей?
                    is_outlier[i] = True
            else:
                 # Рассчитываем Z-оценку относительно соседей
                z_score_relative = abs(z_magnitude_clean[i] - mean_neighbor_z) / std_neighbor_z
                if z_score_relative > outlier_std_multiplier:
                    is_outlier[i] = True

        # Фильтруем данные, оставляя только те точки, которые не являются выбросами
        x_data_filtered = x_data_clean[~is_outlier]
        y_data_filtered = y_data_clean[~is_outlier]
        z_magnitude_filtered = z_magnitude_clean[~is_outlier]

        print(f"Удалено выбросов: {len(x_data_clean) - len(x_data_filtered)}. Осталось точек: {len(x_data_filtered)}")

        if len(x_data_filtered) < 3: # Проверяем, осталось ли достаточно точек после фильтрации
             QMessageBox.warning(self, "Недостаточно данных", f"После очистки выбросов осталось слишком мало допустимых точек ({len(x_data_filtered)}) для построения графика.")
             # Очистка графика и выход
             self.ax.cla()
             if hasattr(self, 'cbar') and self.cbar is not None:
                try: self.cbar.remove()
                except ValueError: pass
                del self.cbar
            self.canvas.draw()
            return


        # --- Интерполяция данных на регулярную сетку (используем очищенные данные) ---
        print(f"Начата интерполяция на сетку {grid_resolution}x{grid_resolution}...")
        # Создаем сетку для интерполяции
        xi_min, xi_max = x_data_filtered.min(), x_data_filtered.max()
        yi_min, yi_max = y_data_filtered.min(), y_data_filtered.max()

        # Создаем координаты сетки
        xi = np.linspace(xi_min, xi_max, grid_resolution)
        yi = np.linspace(yi_min, yi_max, grid_resolution)
        xi, yi = np.meshgrid(xi, yi) # Преобразуем в 2D массивы для meshgrid

        # Интерполируем значения Z на созданную сетку
        # 'cubic' - метод интерполяции для гладкой поверхности
        # use 'nearest' outside convex hull to avoid NaNs in areas with data
        zi = griddata((x_data_filtered, y_data_filtered), z_magnitude_filtered, (xi, yi), method='cubic') #, fill_value=np.nan)
        # Заполняем NaN значения вне выпуклой оболочки данных нан-ами (griddata делает это по умолчанию, если не указан fill_value)


        # --- Сглаживание интерполированной сетки (Smoothing) ---
        print(f"Начато сглаживание с sigma={smoothing_sigma}...")

        # Применяем Гауссовский фильтр к сетке zi
        # mode='nearest' обрабатывает границы сетки, используя значения ближайших точек
        # gaussian_filter из scipy.ndimage умеет работать с NaN в zi
        # Для лучшей обработки NaN, иногда полезно временно заполнить их интерполяцией или оставить как есть.
        # Тестирование показывает, что gaussian_filter с mode='nearest' достаточно хорошо справляется с NaN.
        zi_smoothed = gaussian_filter(zi, sigma=smoothing_sigma, mode='nearest')


        # --- Построение 3D графика поверхности (основная поверхность) ---
        print("Начато построение 3D графика...")
        # Очищаем предыдущие оси
        self.ax.cla()

        # Рисуем основную сглаженную поверхность
       
        surf = self.ax.plot_surface(xi, yi, zi_smoothed, cmap='RdYlGn_r', antialiased=True, rstride=1, cstride=1)
        # rstride=1, cstride=1 - используем все точки сетки для отрисовки (самая высокая детализация)


        # --- Ручное создание "объема" (юбка и дно) ---

        # Определяем базовый уровень Z для "дна" и "юбки"
        # Можно использовать минимальное значение Z из данных, или 0, или другое значение
        # Найдем минимальное конечное значение Z на сглаженной сетке
        finite_zi_smoothed = zi_smoothed[np.isfinite(zi_smoothed)]
        if len(finite_zi_smoothed) > 0:
            z_base = finite_zi_smoothed.min() # Уровень дна = минимальное конечное значение Z на поверхности
            # Опционально: установить дно на 0, если минимальное значение положительное
            # if z_base > 0: z_base = 0
        else:
            z_base = 0 # Если вся сетка NaN, пусть дно будет на 0 (или выбрать другое разумное значение)

        # Создаем вершины и грани для юбки и дна
        manual_verts = []
        manual_faces = []
        manual_facecolors = [] # Для раскраски юбки и дна тем же градиентом

        # Получаем цветовой градиент (colormap) и нормализацию
        cmap = plt.get_cmap('RdYlGn_r')
        # Нормализация для цветов всей фигуры (от мин Z дна до макс Z поверхности)
        overall_z_min = min(z_base, finite_zi_smoothed.min()) if len(finite_zi_smoothed) > 0 else z_base
        overall_z_max = finite_zi_smoothed.max() if len(finite_zi_smoothed) > 0 else z_base + 1 # Избегаем деления на ноль если Z константа
        if overall_z_max == overall_z_min: overall_z_max += 1 # Если Z - константа, немного расширим диапазон
        norm = plt.Normalize(overall_z_min, overall_z_max)


        # Обрабатываем края сетки (4 стороны: верхняя, нижняя, левая, правая)
        rows, cols = zi_smoothed.shape

        # Функции для добавления грани (прямоугольника из 2х треугольников)
        def add_quad_face(v1_idx, v2_idx, v3_idx, v4_idx, color_value):
            current_verts_count = len(manual_verts) # Вершины уже добавлены в manual_verts
            manual_faces.extend([[v1_idx, v2_idx, v3_idx],
                                [v1_idx, v3_idx, v4_idx]])
            manual_facecolors.extend([cmap(norm(color_value))] * 2)

        # --- Юбка по верхнему краю (i=0) ---
        for j in range(cols - 1):
            # Только если обе точки края конечны
            if np.isfinite(zi_smoothed[0, j]) and np.isfinite(zi_smoothed[0, j+1]):
                # Вершины на верхнем крае поверхности
                v_top1 = [xi[0, j], yi[0, j], zi_smoothed[0, j]]
                v_top2 = [xi[0, j+1], yi[0, j+1], zi_smoothed[0, j+1]]
                # Соответствующие вершины на базовом уровне Z
                v_bottom1 = [xi[0, j], yi[0, j], z_base]
                v_bottom2 = [xi[0, j+1], yi[0, j+1], z_base]

                # Индексы этих вершин в общем списке manual_verts после их добавления
                current_idx = len(manual_verts)
                manual_verts.extend([v_top1, v_top2, v_bottom2, v_bottom1]) # Добавляем в порядке v1, v2, v3, v4 для грани

                # Цвет грани юбки (берем цвет от средней Z по верхнему краю грани)
                avg_z = (zi_smoothed[0, j] + zi_smoothed[0, j+1]) / 2.0
                add_quad_face(current_idx, current_idx + 1, current_idx + 2, current_idx + 3, avg_z)


        # --- Юбка по нижнему краю (i=rows-1) ---
        for j in range(cols - 1):
            if np.isfinite(zi_smoothed[rows-1, j]) and np.isfinite(zi_smoothed[rows-1, j+1]):
                v_top1 = [xi[rows-1, j], yi[rows-1, j], zi_smoothed[rows-1, j]]
                v_top2 = [xi[rows-1, j+1], yi[rows-1, j+1], zi_smoothed[rows-1, j+1]]
                v_bottom1 = [xi[rows-1, j], yi[rows-1, j], z_base]
                v_bottom2 = [xi[rows-1, j+1], yi[rows-1, j+1], z_base]

                current_idx = len(manual_verts)
                manual_verts.extend([v_top1, v_top2, v_bottom2, v_bottom1])
                 # Добавляем грани в обратном порядке для правильной нормали, если смотреть снаружи
                manual_faces.extend([[current_idx, current_idx+3, current_idx+2],
                                    [current_idx, current_idx+2, current_idx+1]])

                avg_z = (zi_smoothed[rows-1, j] + zi_smoothed[rows-1, j+1]) / 2.0
                manual_facecolors.extend([cmap(norm(avg_z))] * 2)


        # --- Юбка по левому краю (j=0) ---
        for i in range(rows - 1):
            if np.isfinite(zi_smoothed[i, 0]) and np.isfinite(zi_smoothed[i+1, 0]):
                v_top1 = [xi[i, 0], yi[i, 0], zi_smoothed[i, 0]]
                v_top2 = [xi[i+1, 0], yi[i+1, 0], zi_smoothed[i+1, 0]]
                v_bottom1 = [xi[i, 0], yi[i, 0], z_base]
                v_bottom2 = [xi[i+1, 0], yi[i+1, 0], z_base]

                current_idx = len(manual_verts)
                manual_verts.extend([v_top1, v_top2, v_bottom2, v_bottom1])
                manual_faces.extend([[current_idx, current_idx+1, current_idx+2],
                                    [current_idx, current_idx+2, current_idx+3]])

                avg_z = (zi_smoothed[i, 0] + zi_smoothed[i+1, 0]) / 2.0
                manual_facecolors.extend([cmap(norm(avg_z))] * 2)


        # --- Юбка по правому краю (j=cols-1) ---
        for i in range(rows - 1):
            if np.isfinite(zi_smoothed[i, cols-1]) and np.isfinite(zi_smoothed[i+1, cols-1]):
                v_top1 = [xi[i, cols-1], yi[i, cols-1], zi_smoothed[i, cols-1]]
                v_top2 = [xi[i+1, cols-1], yi[i+1, cols-1], zi_smoothed[i+1, cols-1]]
                v_bottom1 = [xi[i, cols-1], yi[i, cols-1], z_base]
                v_bottom2 = [xi[i+1, cols-1], yi[i+1, cols-1], z_base]

                current_idx = len(manual_verts)
                manual_verts.extend([v_top1, v_top2, v_bottom2, v_bottom1])
                 # Грани в обратном порядке
                manual_faces.extend([[current_idx, current_idx+3, current_idx+2],
                                    [current_idx, current_idx+2, current_idx+1]])

                avg_z = (zi_smoothed[i, cols-1] + zi_smoothed[i+1, cols-1]) / 2.0
                manual_facecolors.extend([cmap(norm(avg_z))] * 2)


        # --- Создание "Дна" ---
        # Используем точки на уровне z_base, соответствующие точкам сетки xi, yi
        # Исключаем NaN из zi_smoothed, чтобы получить только те точки сетки, где есть данные
        finite_mask_zi = np.isfinite(zi_smoothed)
        bottom_verts_grid = np.vstack((xi[finite_mask_zi].ravel(),
                                       yi[finite_mask_zi].ravel(),
                                       np.full(np.sum(finite_mask_zi), z_base))).T

        if len(bottom_verts_grid) >= 3:
             # Триангулируем XY координаты точек дна
            bottom_tri = Delaunay(bottom_verts_grid[:, :2])
            bottom_faces_indices = bottom_tri.simplices

             # Нужно сопоставить индексы триангуляции с индексами в manual_verts
             # Это сложно, если просто добавлять в один список verts
             # Проще создать отдельную Poly3DCollection для дна
             # ИЛИ Удостовериться, что порядок вершин дна в manual_verts соответствует tri.points

             # Давайте создадим отдельную коллекцию для дна для упрощения
            bottom_collection_verts = bottom_verts_grid
            bottom_collection_faces = bottom_faces_indices

             # Цвет для дна (используем цвет, соответствующий z_base)
            bottom_color = cmap(norm(z_base)) # Цвет на нижней границе
        # --- Добавление юбки и дна к графику ---
        # Добавляем коллекцию юбки
        if manual_verts and manual_faces:
            skirt_polygons = []
            for face_indices in manual_faces:
                triangle = [manual_verts[i] for i in face_indices]
                skirt_polygons.append(triangle)

            if len(manual_facecolors) == len(skirt_polygons):
                skirt_collection = art3d.Poly3DCollection(skirt_polygons,
                                                            facecolors=manual_facecolors,
                                                            shade=False)
                self.ax.add_collection3d(skirt_collection)
            else:
                print(f"Warning: Mismatch between number of skirt faces ({len(skirt_polygons)}) and face colors ({len(manual_facecolors)}). Skirt not added.")


        # Добавляем коллекцию дна
        if 'bottom_collection_verts' in locals() and bottom_collection_verts.size > 0 and 'bottom_collection_faces' in locals():
            bottom_polygons = []
            for face_indices in bottom_collection_faces:
                triangle = [bottom_collection_verts[i] for i in face_indices]
                bottom_polygons.append(triangle)

            bottom_collection = art3d.Poly3DCollection(bottom_polygons,
                                                       facecolors=bottom_color, 
                                                       shade=False)
            self.ax.add_collection3d(bottom_collection)
        elif 'bottom_collection_verts' in locals() and bottom_collection_verts.size > 0:
             print("Warning: Bottom collection vertices exist, but faces (from Delaunay triangulation) are missing or empty.")


        # --- Убедимся, что оси Z включают уровень дна (Остальная часть кода без изменений) ---
        current_zlim = self.ax.get_zlim()
        if np.isfinite(z_base):
            self.ax.set_zlim(min(current_zlim[0], z_base), max(current_zlim[1], z_base + 1e-9))
        else:
            print(f"Warning: z_base ({z_base}) is not finite. Z limits not adjusted for base.")


        # --- Настройка осей и заголовка ---
        self.ax.set_xlabel(x_col_name)
        self.ax.set_ylabel(y_col_name)
        self.ax.set_zlabel(f'Модуль магнитного импеданса |Z|')
        try:
            filename = os.path.basename(self.file_label.text().split(": ")[1])
        except IndexError:
            filename = "Неизвестный файл"
        self.ax.set_title(f'3D график |Z| от {x_col_name} и {y_col_name}\n'
                          f'(Файл: {filename})')


        # --- Добавление/обновление цветовой шкалы (привязана к основной поверхности) ---
        if hasattr(self, 'cbar') and self.cbar is not None:
            try:
                self.cbar.remove()
            except ValueError:
                pass
            del self.cbar

        # Добавляем новый Colorbar, привязанный к основной поверхности (surf)
        self.cbar = self.figure.colorbar(surf, ax=self.ax, shrink=0.5, aspect=5, label='Модуль магнитного импеданса |Z|', norm=norm) # Используем общую нормализацию


        # --- Обновление канваса ---
        self.canvas.draw()

# --- Основная часть приложения ---
if __name__ == '__main__':
    # Создаем приложение PyQt
    app = QApplication(sys.argv)

    # Создаем и показываем главное окно
    main_window = ImpedancePlotterApp()
    main_window.show()

    # Запускаем цикл обработки событий PyQt
    sys.exit(app.exec())