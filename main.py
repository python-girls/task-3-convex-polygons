import math
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
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

    def _cross_product(self, a: Point, b: Point, c: Point) -> float:
        """Векторное произведение (b-a) x (c-a)"""
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

    def _is_convex(self) -> bool:
        n = len(self.vertices)
        if n < 3:
            return False

        sign = 0
        for i in range(n):
            cross = self._cross_product(
                self.vertices[i],
                self.vertices[(i + 1) % n],
                self.vertices[(i + 2) % n]
            )

            if abs(cross) > 1e-10:
                if sign == 0:
                    sign = 1 if cross > 0 else -1
                elif (cross > 0 and sign == -1) or (cross < 0 and sign == 1):
                    return False

        return True

    def _order_vertices_ccw(self):
        """Упорядочивание вершин против часовой стрелки"""
        if len(self.vertices) < 3:
            return

        # Находим самую левую-нижнюю точку как начальную
        start_idx = 0
        for i in range(1, len(self.vertices)):
            if (self.vertices[i].x < self.vertices[start_idx].x or
                    (math.isclose(self.vertices[i].x, self.vertices[start_idx].x) and
                     self.vertices[i].y < self.vertices[start_idx].y)):
                start_idx = i

        # Переупорядочиваем список вершин
        self.vertices = self.vertices[start_idx:] + self.vertices[:start_idx]

        # Сортируем оставшиеся вершины по полярному углу относительно начальной
        if len(self.vertices) > 1:
            reference = self.vertices[0]
            self.vertices[1:] = sorted(
                self.vertices[1:],
                key=lambda p: math.atan2(p.y - reference.y, p.x - reference.x)
            )

            # Проверяем направление (должно быть против часовой стрелки)
            if len(self.vertices) >= 3:
                cross = self._cross_product(self.vertices[0], self.vertices[1], self.vertices[2])
                if cross < 0:
                    # Если по часовой - разворачиваем
                    self.vertices = [self.vertices[0]] + self.vertices[1:][::-1]

    def perimeter(self) -> float:
        perimeter = 0.0
        n = len(self.vertices)

        for i in range(n):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % n]
            perimeter += math.hypot(p2.x - p1.x, p2.y - p1.y)

        return perimeter

    def area(self) -> float:
        area = 0.0
        n = len(self.vertices)

        for i in range(n):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % n]
            area += p1.x * p2.y - p2.x * p1.y

        return abs(area) / 2.0

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Проверка точки в выпуклом многоугольнике методом углов"""
        test_point = Point(point[0], point[1])
        n = len(self.vertices)

        # Если точка совпадает с вершиной
        for vertex in self.vertices:
            if vertex == test_point:
                return True

        total_angle = 0.0
        for i in range(n):
            a = self.vertices[i]
            b = self.vertices[(i + 1) % n]

            # Проверка нахождения точки на ребре
            cross = self._cross_product(a, b, test_point)
            if abs(cross) < 1e-10:
                # Проверяем, лежит ли точка на отрезке
                min_x, max_x = min(a.x, b.x), max(a.x, b.x)
                min_y, max_y = min(a.y, b.y), max(a.y, b.y)
                if (min_x - 1e-10 <= test_point.x <= max_x + 1e-10 and
                        min_y - 1e-10 <= test_point.y <= max_y + 1e-10):
                    return True

            # Векторы от точки к вершинам
            v1 = Point(a.x - test_point.x, a.y - test_point.y)
            v2 = Point(b.x - test_point.x, b.y - test_point.y)

            # Вычисляем угол между векторами
            dot = v1.x * v2.x + v1.y * v2.y
            norm1 = math.hypot(v1.x, v1.y)
            norm2 = math.hypot(v2.x, v2.y)

            if norm1 < 1e-10 or norm2 < 1e-10:
                continue

            cos_angle = dot / (norm1 * norm2)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle = math.acos(cos_angle)

            # Определяем знак угла
            cross = v1.x * v2.y - v1.y * v2.x
            if cross < 0:
                angle = -angle

            total_angle += angle

        # Точка внутри, если сумма углов ≈ ±2π
        return abs(abs(total_angle) - 2 * math.pi) < 1e-10

    def contains_polygon(self, other: 'ConvexPolygon') -> bool:
        """Проверяет, содержит ли текущий многоугольник другой"""
        # Проверяем все вершины другого многоугольника
        for vertex in other.vertices:
            if not self.contains_point((vertex.x, vertex.y)):
                return False
        return True

    def _lines_intersect(self, a: Point, b: Point, c: Point, d: Point) -> bool:
        """Проверяет пересекаются ли два отрезка"""

        def orientation(p, q, r):
            val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
            if abs(val) < 1e-10:
                return 0  # коллинеарны
            return 1 if val > 0 else 2  # по часовой или против

        def on_segment(p, q, r):
            return (min(p.x, r.x) - 1e-10 <= q.x <= max(p.x, r.x) + 1e-10 and
                    min(p.y, r.y) - 1e-10 <= q.y <= max(p.y, r.y) + 1e-10)

        o1 = orientation(a, b, c)
        o2 = orientation(a, b, d)
        o3 = orientation(c, d, a)
        o4 = orientation(c, d, b)

        # Общий случай
        if o1 != o2 and o3 != o4:
            return True

        # Специальные случаи (коллинеарность)
        if o1 == 0 and on_segment(a, c, b):
            return True
        if o2 == 0 and on_segment(a, d, b):
            return True
        if o3 == 0 and on_segment(c, a, d):
            return True
        if o4 == 0 and on_segment(c, b, d):
            return True

        return False

    def _do_polygons_intersect(self, other: 'ConvexPolygon') -> bool:
        """Проверяет, пересекаются ли многоугольники с использованием SAT"""
        polygons = [self, other]

        for polygon in polygons:
            other_polygon = other if polygon == self else self

            n = len(polygon.vertices)
            for i in range(n):
                # Получаем нормаль к текущему ребру
                p1 = polygon.vertices[i]
                p2 = polygon.vertices[(i + 1) % n]

                edge = Point(p2.x - p1.x, p2.y - p1.y)
                normal = Point(-edge.y, edge.x)  # Перпендикуляр против часовой стрелки

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

    def intersects(self, other: 'ConvexPolygon') -> bool:
        """Проверяет пересекаются ли два выпуклых многоугольника"""
        # Сначала быстрая проверка ограничивающих прямоугольников
        bbox1 = self.get_bounding_box()
        bbox2 = other.get_bounding_box()

        # Проверка пересечения ограничивающих прямоугольников
        if (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
                bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3]):
            return False

        # Если ограничивающие прямоугольники пересекаются, используем SAT
        return self._do_polygons_intersect(other)

    def intersection(self, other: 'ConvexPolygon') -> Optional['ConvexPolygon']:
        """Находит пересечение двух выпуклых многоугольников"""
        # Простой алгоритм для выпуклых многоугольников:
        # 1. Находим все точки пересечения ребер
        # 2. Добавляем вершины, лежащие внутри другого многоугольника
        # 3. Строим выпуклую оболочку

        intersection_points = []

        # Вершины первого многоугольника внутри второго
        for vertex in self.vertices:
            if other.contains_point((vertex.x, vertex.y)):
                intersection_points.append(vertex)

        # Вершины второго многоугольника внутри первого
        for vertex in other.vertices:
            if self.contains_point((vertex.x, vertex.y)):
                intersection_points.append(vertex)

        # Точки пересечения ребер
        n1, n2 = len(self.vertices), len(other.vertices)
        for i in range(n1):
            a1 = self.vertices[i]
            a2 = self.vertices[(i + 1) % n1]

            for j in range(n2):
                b1 = other.vertices[j]
                b2 = other.vertices[(j + 1) % n2]

                intersection = self._compute_intersection(a1, a2, b1, b2)
                if intersection:
                    intersection_points.append(intersection)

        # Удаляем дубликаты
        unique_points = []
        for point in intersection_points:
            if not any(p == point for p in unique_points):
                unique_points.append(point)

        if len(unique_points) < 3:
            return None

        # Строим выпуклую оболочку
        try:
            return self._convex_hull(unique_points)
        except ValueError:
            return None

    def _compute_intersection(self, p1: Point, p2: Point, q1: Point, q2: Point) -> Optional[Point]:
        """Вычисляет точку пересечения двух отрезков"""
        # Параметрическое представление: p1 + t*(p2-p1) = q1 + u*(q2-q1)
        denom = (p2.x - p1.x) * (q2.y - q1.y) - (p2.y - p1.y) * (q2.x - q1.x)

        if abs(denom) < 1e-10:
            return None  # Параллельны или коллинеарны

        t = ((q1.x - p1.x) * (q2.y - q1.y) - (q1.y - p1.y) * (q2.x - q1.x)) / denom
        u = ((q1.x - p1.x) * (p2.y - p1.y) - (q1.y - p1.y) * (p2.x - p1.x)) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            x = p1.x + t * (p2.x - p1.x)
            y = p1.y + t * (p2.y - p1.y)
            return Point(x, y)

        return None

    def _convex_hull(self, points: List[Point]) -> 'ConvexPolygon':
        """Строит выпуклую оболочку для набора точек (алгоритм Грэхема)"""
        if len(points) < 3:
            raise ValueError("Нужно хотя бы 3 точки для выпуклой оболочки")

        # Находим самую левую-нижнюю точку
        start_point = min(points, key=lambda p: (p.x, p.y))

        # Сортируем по полярному углу относительно start_point
        def polar_angle(p):
            angle = math.atan2(p.y - start_point.y, p.x - start_point.x)
            return angle

        sorted_points = sorted([p for p in points if p != start_point], key=polar_angle)

        # Строим выпуклую оболочку
        hull = [start_point]

        for point in sorted_points:
            while len(hull) >= 2:
                a, b = hull[-2], hull[-1]
                cross = self._cross_product(a, b, point)
                if cross <= 0:  # Удаляем точку, если поворот по часовой или коллинеарны
                    hull.pop()
                else:
                    break
            hull.append(point)

        return ConvexPolygon(hull)

    def triangulate(self) -> List[List[Tuple[float, float]]]:
        """Триангуляция выпуклого многоугольника методом веера"""
        if len(self.vertices) < 3:
            return []

        triangles = []
        n = len(self.vertices)

        for i in range(1, n - 1):
            triangle = [
                (self.vertices[0].x, self.vertices[0].y),
                (self.vertices[i].x, self.vertices[i].y),
                (self.vertices[i + 1].x, self.vertices[i + 1].y)
            ]
            triangles.append(triangle)

        return triangles

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Возвращает ограничивающий прямоугольник (min_x, min_y, max_x, max_y)"""
        if not self.vertices:
            return 0.0, 0.0, 0.0, 0.0

        min_x = min(v.x for v in self.vertices)
        max_x = max(v.x for v in self.vertices)
        min_y = min(v.y for v in self.vertices)
        max_y = max(v.y for v in self.vertices)

        return min_x, min_y, max_x, max_y

    def plot(self, color='blue', label=None, show_vertices=True):
        """Визуализация многоугольника"""
        if not self.vertices:
            return

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

