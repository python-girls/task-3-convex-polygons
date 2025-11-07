import math
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return abs(self.x - other.x) < 1e-10 and abs(self.y - other.y) < 1e-10

    def __hash__(self):
        return hash((round(self.x, 10), round(self.y, 10)))

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return f"Point({self.x}, {self.y})"


class ConvexPolygon:
    def __init__(self, vertices: List[Point]):
        if len(vertices) < 3:
            raise ValueError("Многоугольник должен иметь как минимум 3 вершины")

        self.vertices = vertices

        # проверка выпуклости
        if not self._is_convex():
            raise ValueError("Многоугольник не является выпуклым")

        # упорядочиваем вершины против часовой стрелки
        self._order_vertices_ccw()

    def _is_convex(self) -> bool:
        n = len(self.vertices)
        if n < 3:
            return False

        sign = 0
        for i in range(n):
            # векторное произведение для трех последовательных точек
            dx1 = self.vertices[(i + 1) % n].x - self.vertices[i].x
            dy1 = self.vertices[(i + 1) % n].y - self.vertices[i].y
            dx2 = self.vertices[(i + 2) % n].x - self.vertices[(i + 1) % n].x
            dy2 = self.vertices[(i + 2) % n].y - self.vertices[(i + 1) % n].y

            cross = dx1 * dy2 - dy1 * dx2

            if abs(cross) > 1e-10:
                if sign == 0:
                    sign = 1 if cross > 0 else -1
                elif (cross > 0 and sign == -1) or (cross < 0 and sign == 1):
                    return False

        return True

    def _order_vertices_ccw(self):
        if len(self.vertices) < 3:
            return

        # находим центр многоугольника
        center_x = sum(p.x for p in self.vertices) / len(self.vertices)
        center_y = sum(p.y for p in self.vertices) / len(self.vertices)
        center = Point(center_x, center_y)

        # сортируем вершины по углу относительно центра
        def angle_from_center(point: Point):
            return math.atan2(point.y - center_y, point.x - center_x)

        self.vertices.sort(key=angle_from_center)

    def perimeter(self) -> float:
        perimeter = 0.0
        n = len(self.vertices)

        for i in range(n):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % n]
            perimeter += math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)

        return perimeter

    def area(self) -> float:
        area = 0.0
        n = len(self.vertices)

        for i in range(n):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % n]
            area += (p1.x * p2.y - p2.x * p1.y)

        return abs(area) / 2.0

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Более эффективный метод для выпуклых многоугольников"""
        test_point = Point(point[0], point[1])
        n = len(self.vertices)

        # Для выпуклого многоугольника проверяем знаки векторных произведений
        sign = None
        for i in range(n):
            a = self.vertices[i]
            b = self.vertices[(i + 1) % n]

            cross = (b.x - a.x) * (test_point.y - a.y) - (b.y - a.y) * (test_point.x - a.x)

            if abs(cross) < 1e-10:
                continue  # Точка на ребре

            if sign is None:
                sign = cross > 0
            elif (cross > 0) != sign:
                return False

        return True

    def contains_polygon(self, other: 'ConvexPolygon') -> bool:
        # проверяем все вершины другого многоугольника
        for vertex in other.vertices:
            if not self.contains_point((vertex.x, vertex.y)):
                return False
        return True

    def _cross_product(self, a: Point, b: Point, c: Point) -> float:
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

    def _lines_intersect(self, a: Point, b: Point, c: Point, d: Point) -> bool:
        """Проверяет пересекаются ли два отрезка"""

        def ccw(A, B, C):
            return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

    def _do_polygons_intersect(self, other: 'ConvexPolygon') -> bool:
        """Проверяет, пересекаются ли многоугольники"""
        # Проверяем если одна вершина одного многоугольника находится внутри другого
        for vertex in self.vertices:
            if other.contains_point((vertex.x, vertex.y)):
                return True
        for vertex in other.vertices:
            if self.contains_point((vertex.x, vertex.y)):
                return True

        # Проверяем пересечения ребер
        n1, n2 = len(self.vertices), len(other.vertices)
        for i in range(n1):
            for j in range(n2):
                a, b = self.vertices[i], self.vertices[(i + 1) % n1]
                c, d = other.vertices[j], other.vertices[(j + 1) % n2]

                if self._lines_intersect(a, b, c, d):
                    return True

        return False

    def intersects(self, other: 'ConvexPolygon') -> bool:
        """Проверяет пересекаются ли два выпуклых многоугольника"""
        return self._do_polygons_intersect(other)

    def _is_inside_clip_edge(self, point: Point, edge_start: Point, edge_end: Point) -> bool:
        # для выпуклого многоугольника проверяем по векторному произведению
        cross = self._cross_product(edge_start, edge_end, point)
        return cross >= -1e-10

    def _line_intersection(self, a: Point, b: Point, c: Point, d: Point) -> Optional[Point]:
        denom = (a.x - b.x) * (c.y - d.y) - (a.y - b.y) * (c.x - d.x)

        if abs(denom) < 1e-10:
            return None  # Параллельные линии

        t = ((a.x - c.x) * (c.y - d.y) - (a.y - c.y) * (c.x - d.x)) / denom
        u = -((a.x - b.x) * (a.y - c.y) - (a.y - b.y) * (a.x - c.x)) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            x = a.x + t * (b.x - a.x)
            y = a.y + t * (b.y - a.y)
            return Point(x, y)

        return None

    def intersection(self, other: 'ConvexPolygon') -> Optional['ConvexPolygon']:
        """Находит пересечение двух выпуклых многоугольников используя алгоритм Сазерленда-Ходжмана"""
        # Создаем копию первого многоугольника как начальный клиппируемый многоугольник
        output_polygon = self.vertices.copy()

        # Проходим по всем ребрам второго многоугольника как отсекающим плоскостям
        n = len(other.vertices)
        for i in range(n):
            if not output_polygon:
                return None

            # Текущее отсекающее ребро
            clip_start = other.vertices[i]
            clip_end = other.vertices[(i + 1) % n]

            input_list = output_polygon
            output_polygon = []

            # Вектор нормали к отсекающему ребру (направлен внутрь второго многоугольника)
            edge_vec = Point(clip_end.x - clip_start.x, clip_end.y - clip_start.y)
            normal = Point(-edge_vec.y, edge_vec.x)  # Перпендикуляр против часовой стрелки

            prev_point = input_list[-1]
            prev_inside = self._is_inside(prev_point, clip_start, normal)

            for current_point in input_list:
                current_inside = self._is_inside(current_point, clip_start, normal)

                # Если текущая точка внутри - добавляем ее
                if current_inside:
                    # Если предыдущая точка была снаружи - добавляем пересечение
                    if not prev_inside:
                        intersection = self._compute_intersection(prev_point, current_point, clip_start, clip_end)
                        if intersection:
                            output_polygon.append(intersection)
                    output_polygon.append(current_point)
                else:
                    # Если предыдущая точка была внутри - добавляем пересечение
                    if prev_inside:
                        intersection = self._compute_intersection(prev_point, current_point, clip_start, clip_end)
                        if intersection:
                            output_polygon.append(intersection)

                prev_point = current_point
                prev_inside = current_inside

        if len(output_polygon) < 3:
            return None

        try:
            return ConvexPolygon(output_polygon)
        except ValueError:
            return None

    def _is_inside(self, point: Point, clip_point: Point, normal: Point) -> bool:
        """Проверяет, находится ли точка внутри отсекающего ребра"""
        # Вектор от точки отсечения к проверяемой точке
        to_point = Point(point.x - clip_point.x, point.y - clip_point.y)

        # Скалярное произведение с нормалью
        dot_product = to_point.x * normal.x + to_point.y * normal.y

        return dot_product >= -1e-10

    def _compute_intersection(self, p1: Point, p2: Point, clip_start: Point, clip_end: Point) -> Optional[Point]:
        """Вычисляет точку пересечения двух отрезков"""
        # Параметрическое представление первого отрезка: p1 + t*(p2 - p1)
        # Параметрическое представление второго отрезка: clip_start + u*(clip_end - clip_start)

        denom = (p1.x - p2.x) * (clip_start.y - clip_end.y) - (p1.y - p2.y) * (clip_start.x - clip_end.x)

        if abs(denom) < 1e-10:
            return None  # Отрезки параллельны

        t = ((p1.x - clip_start.x) * (clip_start.y - clip_end.y) - (p1.y - clip_start.y) * (
                    clip_start.x - clip_end.x)) / denom
        u = -((p1.x - p2.x) * (p1.y - clip_start.y) - (p1.y - p2.y) * (p1.x - clip_start.x)) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            x = p1.x + t * (p2.x - p1.x)
            y = p1.y + t * (p2.y - p1.y)
            return Point(x, y)

        return None

    def _do_polygons_intersect(self, other: 'ConvexPolygon') -> bool:
        """Проверяет, пересекаются ли многоугольники (более надежная версия)"""
        # Проверяем если одна вершина одного многоугольника находится внутри другого
        for vertex in self.vertices:
            if other.contains_point((vertex.x, vertex.y)):
                return True
        for vertex in other.vertices:
            if self.contains_point((vertex.x, vertex.y)):
                return True

        # Проверяем пересечения ребер используя Separating Axis Theorem (SAT) для выпуклых многоугольников
        polygons = [self, other]

        for polygon in polygons:
            other_polygon = other if polygon == self else self

            n = len(polygon.vertices)
            for i in range(n):
                # Получаем нормаль к текущему ребру
                p1 = polygon.vertices[i]
                p2 = polygon.vertices[(i + 1) % n]

                edge = Point(p2.x - p1.x, p2.y - p1.y)
                normal = Point(-edge.y, edge.x)  # Перпендикуляр

                # Проецируем первый многоугольник на нормаль
                min1, max1 = float('inf'), float('-inf')
                for vertex in polygon.vertices:
                    projection = (vertex.x - p1.x) * normal.x + (vertex.y - p1.y) * normal.y
                    min1 = min(min1, projection)
                    max1 = max(max1, projection)

                # Проецируем второй многоугольник на нормаль
                min2, max2 = float('inf'), float('-inf')
                for vertex in other_polygon.vertices:
                    projection = (vertex.x - p1.x) * normal.x + (vertex.y - p1.y) * normal.y
                    min2 = min(min2, projection)
                    max2 = max(max2, projection)

                # Если проекции не пересекаются - многоугольники не пересекаются
                if max1 < min2 or max2 < min1:
                    return False

        return True

    def triangulate(self) -> List[List[Tuple[float, float]]]:
        if len(self.vertices) < 3:
            return []

        triangles = []
        n = len(self.vertices)

        # для выпуклого многоугольника используем простой метод веера
        for i in range(1, n - 1):
            triangle = [
                (self.vertices[0].x, self.vertices[0].y),
                (self.vertices[i].x, self.vertices[i].y),
                (self.vertices[i + 1].x, self.vertices[i + 1].y)
            ]
            triangles.append(triangle)

        return triangles

    def plot(self, color='blue', label=None, show_vertices=True):
        x = [p.x for p in self.vertices] + [self.vertices[0].x]
        y = [p.y for p in self.vertices] + [self.vertices[0].y]

        plt.plot(x, y, 'o-', color=color, label=label, markersize=5 if show_vertices else 0)
        plt.fill(x, y, color=color, alpha=0.3)

    def get_vertices(self) -> List[Tuple[float, float]]:
        return [(p.x, p.y) for p in self.vertices]

    def __str__(self):
        vertices_str = ", ".join(str(v) for v in self.vertices)
        return f"ConvexPolygon([{vertices_str}])"

    def __repr__(self):
        return self.__str__()


def visualize_polygons(polygons, title="Визуализация многоугольников", current_idx=None):
    """Функция для визуализации списка многоугольников"""
    if not polygons:
        return

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    for i, polygon in enumerate(polygons):
        color = colors[i % len(colors)]
        if current_idx == i:
            label = f'Многоугольник {i + 1} (ТЕКУЩИЙ, S={polygon.area():.1f})'
            linewidth = 3
        else:
            label = f'Многоугольник {i + 1} (S={polygon.area():.1f})'
            linewidth = 2

        x = [p.x for p in polygon.vertices] + [polygon.vertices[0].x]
        y = [p.y for p in polygon.vertices] + [polygon.vertices[0].y]

        plt.plot(x, y, 'o-', color=color, label=label, markersize=5, linewidth=linewidth)
        plt.fill(x, y, color=color, alpha=0.2)

    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


def input_polygon(prompt: str) -> ConvexPolygon:
    print(f"\n{prompt}")
    print("Введите координаты вершин многоугольника (x y). Для завершения введите 'end'")

    vertices = []
    while True:
        try:
            user_input = input(f"Вершина {len(vertices) + 1}: ").strip()
            if user_input.lower() == 'end':
                break

            coords = user_input.split()
            if len(coords) != 2:
                print("Ошибка: введите две координаты через пробел")
                continue

            x, y = float(coords[0]), float(coords[1])
            vertices.append(Point(x, y))

            if len(vertices) >= 3:
                print(f"Текущие вершины: {[str(v) for v in vertices]}")
                choice = input("Продолжить ввод? (y/n): ").strip().lower()
                if choice == 'n':
                    break

        except ValueError:
            print("Ошибка: введите числа")

    try:
        return ConvexPolygon(vertices)
    except ValueError as e:
        print(f"Ошибка создания многоугольника: {e}")
        return None


def input_point(prompt: str) -> Tuple[float, float]:
    print(f"\n{prompt}")
    while True:
        try:
            user_input = input("Введите координаты точки (x y): ").strip()
            coords = user_input.split()
            if len(coords) != 2:
                print("Ошибка: введите две координаты через пробел")
                continue
            return float(coords[0]), float(coords[1])
        except ValueError:
            print("Ошибка: введите числа")


def main():
    polygons = []
    current_polygon = None
    current_idx = None
    auto_visualize = True  # Флаг автоматической визуализации

    while True:
        print("\n" + "=" * 50)
        print("          РАБОТА С ВЫПУКЛЫМИ МНОГОУГОЛЬНИКАМИ")
        print("=" * 50)
        print("1. Создать новый многоугольник")
        print("2. Выбрать текущий многоугольник")
        print("3. Вычислить периметр")
        print("4. Вычислить площадь")
        print("5. Проверить точку внутри многоугольника")
        print("6. Проверить вложенность многоугольников")
        print("7. Найти пересечение многоугольников")
        print("8. Выполнить триангуляцию")
        print("9. Визуализировать многоугольники")
        print("10. Показать все многоугольники")
        print("11. Переключить автоматическую визуализацию")
        print("0. Выход")

        if current_polygon:
            print(f"\nТекущий многоугольник: {current_polygon}")

        choice = input("\nВыберите действие: ").strip()

        if choice == '0':
            print("Выход из программы.")
            break

        elif choice == '1':
            polygon = input_polygon("СОЗДАНИЕ НОВОГО МНОГОУГОЛЬНИКА")
            if polygon:
                polygons.append(polygon)
                current_polygon = polygon
                current_idx = len(polygons) - 1
                print(f"Многоугольник создан: {polygon}")

                # Автоматическая визуализация после создания
                if auto_visualize and len(polygons) > 0:
                    visualize_polygons(polygons, "Текущие многоугольники", current_idx)

        elif choice == '2':
            if not polygons:
                print("Нет созданных многоугольников")
                continue

            print("\nСПИСОК МНОГОУГОЛЬНИКОВ:")
            for i, poly in enumerate(polygons):
                status = " ← ТЕКУЩИЙ" if i == current_idx else ""
                print(f"{i + 1}. {poly}{status}")

            try:
                idx = int(input("Выберите номер многоугольника: ")) - 1
                if 0 <= idx < len(polygons):
                    current_polygon = polygons[idx]
                    current_idx = idx
                    print(f"Текущий многоугольник: {current_polygon}")

                    # Автоматическая визуализация после выбора
                    if auto_visualize:
                        visualize_polygons(polygons, f"Выбран многоугольник {idx + 1}", current_idx)
                else:
                    print("Неверный номер")
            except ValueError:
                print("Ошибка: введите число")

        elif choice == '3':
            if not current_polygon:
                print("Сначала выберите или создайте многоугольник")
                continue
            perimeter = current_polygon.perimeter()
            print(f"\nПЕРИМЕТР МНОГОУГОЛЬНИКА: {perimeter:.2f}")

        elif choice == '4':
            if not current_polygon:
                print("Сначала выберите или создайте многоугольник")
                continue
            area = current_polygon.area()
            print(f"\nПЛОЩАДЬ МНОГОУГОЛЬНИКА: {area:.2f}")

        elif choice == '5':
            if not current_polygon:
                print("Сначала выберите или создайте многоугольник")
                continue
            point = input_point("ПРОВЕРКА ТОЧКИ ВНУТРИ МНОГОУГОЛЬНИКА")
            is_inside = current_polygon.contains_point(point)
            print(f"Точка {point} {'находится' if is_inside else 'не находится'} внутри многоугольника")

        elif choice == '6':
            if len(polygons) < 2:
                print("Нужно как минимум 2 многоугольника")
                continue

            print("\nПРОВЕРКА ВЛОЖЕННОСТИ МНОГОУГОЛЬНИКОВ")
            print("Список многоугольников:")
            for i, poly in enumerate(polygons):
                status = " ← ТЕКУЩИЙ" if i == current_idx else ""
                print(f"{i + 1}. {poly}{status}")

            try:
                idx1 = int(input("Выберите первый многоугольник: ")) - 1
                idx2 = int(input("Выберите второй многоугольник: ")) - 1

                if 0 <= idx1 < len(polygons) and 0 <= idx2 < len(polygons):
                    poly1 = polygons[idx1]
                    poly2 = polygons[idx2]

                    contains1 = poly1.contains_polygon(poly2)
                    contains2 = poly2.contains_polygon(poly1)

                    print(f"\n{poly1} содержит {poly2}: {contains1}")
                    print(f"{poly2} содержит {poly1}: {contains2}")

                    # Визуализация для наглядности
                    if auto_visualize:
                        plt.figure(figsize=(10, 8))

                        # Рисуем первый многоугольник
                        x1 = [p.x for p in poly1.vertices] + [poly1.vertices[0].x]
                        y1 = [p.y for p in poly1.vertices] + [poly1.vertices[0].y]
                        plt.plot(x1, y1, 'o-', color='blue', label=f'Многоугольник 1', linewidth=2)
                        plt.fill(x1, y1, color='blue', alpha=0.2)

                        # Рисуем второй многоугольник
                        x2 = [p.x for p in poly2.vertices] + [poly2.vertices[0].x]
                        y2 = [p.y for p in poly2.vertices] + [poly2.vertices[0].y]
                        plt.plot(x2, y2, 'o-', color='red', label=f'Многоугольник 2', linewidth=2)
                        plt.fill(x2, y2, color='red', alpha=0.2)

                        plt.title(f'Проверка вложенности многоугольников\n'
                                  f'Многоугольник 1 содержит 2: {contains1}\n'
                                  f'Многоугольник 2 содержит 1: {contains2}')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.axis('equal')
                        plt.show()
                else:
                    print("Неверные номера")
            except ValueError:
                print("Ошибка: введите числа")

        elif choice == '7':
            if len(polygons) < 2:
                print("Нужно как минимум 2 многоугольника")
                continue

            print("\nПЕРЕСЕЧЕНИЕ МНОГОУГОЛЬНИКОВ")
            print("Список многоугольников:")
            for i, poly in enumerate(polygons):
                status = " ← ТЕКУЩИЙ" if i == current_idx else ""
                print(f"{i + 1}. {poly}{status}")

            try:
                idx1 = int(input("Выберите первый многоугольник: ")) - 1
                idx2 = int(input("Выберите второй многоугольник: ")) - 1

                if 0 <= idx1 < len(polygons) and 0 <= idx2 < len(polygons):
                    poly1 = polygons[idx1]
                    poly2 = polygons[idx2]

                    print(f"\nПроверка пересечения...")
                    intersects = poly1.intersects(poly2)
                    print(f"Многоугольники пересекаются: {intersects}")

                    if intersects:
                        intersection = poly1.intersection(poly2)
                        if intersection:
                            print(f"Пересечение найдено: {intersection}")
                            print(f"Площадь пересечения: {intersection.area():.2f}")

                            # Визуализация пересечения
                            if auto_visualize:
                                plt.figure(figsize=(10, 8))

                                # Рисуем первый многоугольник
                                x1 = [p.x for p in poly1.vertices] + [poly1.vertices[0].x]
                                y1 = [p.y for p in poly1.vertices] + [poly1.vertices[0].y]
                                plt.plot(x1, y1, 'o-', color='blue', label=f'Многоугольник 1', linewidth=2)
                                plt.fill(x1, y1, color='blue', alpha=0.2)

                                # Рисуем второй многоугольник
                                x2 = [p.x for p in poly2.vertices] + [poly2.vertices[0].x]
                                y2 = [p.y for p in poly2.vertices] + [poly2.vertices[0].y]
                                plt.plot(x2, y2, 'o-', color='red', label=f'Многоугольник 2', linewidth=2)
                                plt.fill(x2, y2, color='red', alpha=0.2)

                                # Рисуем пересечение
                                x_int = [p.x for p in intersection.vertices] + [intersection.vertices[0].x]
                                y_int = [p.y for p in intersection.vertices] + [intersection.vertices[0].y]
                                plt.plot(x_int, y_int, 'o-', color='green', label=f'Пересечение', linewidth=3)
                                plt.fill(x_int, y_int, color='green', alpha=0.5)

                                plt.title(
                                    f'Пересечение многоугольников\nПлощадь пересечения: {intersection.area():.2f}')
                                plt.legend()
                                plt.grid(True, alpha=0.3)
                                plt.axis('equal')
                                plt.show()
                        else:
                            print("Не удалось построить многоугольник пересечения")
                    else:
                        print("Многоугольники не пересекаются")
                else:
                    print("Неверные номера")
            except ValueError:
                print("Ошибка: введите числа")

        elif choice == '8':
            if not current_polygon:
                print("Сначала выберите или создайте многоугольник")
                continue

            triangles = current_polygon.triangulate()
            print(f"\nТРИАНГУЛЯЦИЯ МНОГОУГОЛЬНИКА")
            print(f"Получено треугольников: {len(triangles)}")

            # Проверяем сумму площадей треугольников
            total_area = 0
            for i, triangle in enumerate(triangles):
                triangle_poly = ConvexPolygon([Point(x, y) for x, y in triangle])
                triangle_area = triangle_poly.area()
                total_area += triangle_area
                print(f"Треугольник {i + 1}: {triangle} (площадь: {triangle_area:.2f})")

            print(f"Сумма площадей треугольников: {total_area:.2f}")
            print(f"Площадь исходного многоугольника: {current_polygon.area():.2f}")
            print(f"Разница: {abs(total_area - current_polygon.area()):.6f}")

            # Визуализация триангуляции
            if auto_visualize and triangles:
                plt.figure(figsize=(10, 8))

                # Рисуем исходный многоугольник
                x_orig = [p.x for p in current_polygon.vertices] + [current_polygon.vertices[0].x]
                y_orig = [p.y for p in current_polygon.vertices] + [current_polygon.vertices[0].y]
                plt.plot(x_orig, y_orig, 'o-', color='black', label='Исходный многоугольник', linewidth=2)

                # Рисуем треугольники
                colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
                for i, triangle in enumerate(triangles):
                    color = colors[i % len(colors)]
                    triangle_points = [Point(x, y) for x, y in triangle]
                    x_tri = [p.x for p in triangle_points] + [triangle_points[0].x]
                    y_tri = [p.y for p in triangle_points] + [triangle_points[0].y]
                    plt.fill(x_tri, y_tri, color=color, alpha=0.3, label=f'Треугольник {i + 1}')
                    plt.plot(x_tri, y_tri, 'o-', color=color, linewidth=1)

                plt.title(f'Триангуляция многоугольника\nТреугольников: {len(triangles)}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.axis('equal')
                plt.show()

        elif choice == '9':
            if not polygons:
                print("Нет многоугольников для визуализации")
                continue

            visualize_polygons(polygons, "Визуализация всех многоугольников", current_idx)

        elif choice == '10':
            if not polygons:
                print("Нет созданных многоугольников")
                continue

            print("\nВСЕ МНОГОУГОЛЬНИКИ:")
            for i, polygon in enumerate(polygons):
                status = " ← ТЕКУЩИЙ" if i == current_idx else ""
                print(f"{i + 1}. {polygon}{status}")
                print(f"   Периметр: {polygon.perimeter():.2f}")
                print(f"   Площадь: {polygon.area():.2f}")
                print()

        elif choice == '11':
            auto_visualize = not auto_visualize
            print(f"Автоматическая визуализация: {'включена' if auto_visualize else 'выключена'}")

        else:
            print("Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    main()