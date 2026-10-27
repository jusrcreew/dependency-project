from __future__ import annotations

import re
import sys
from collections import deque
from typing import Dict, List, Optional, Set, Tuple


class DependencyGraph:
    def __init__(self) -> None:
        self._deps: Dict[str, Dict[str, float]] = {}
        self._dependents: Dict[str, List[str]] = {}
        self.version = 0

    def add_component(self, name: str) -> None:
        if name not in self._deps:
            self._deps[name] = {}
            self._dependents[name] = []
            self.version += 1

    def add_dependency(self, from_: str, to: str, weight: float = 0.0) -> None:
        self.add_component(from_)
        self.add_component(to)
        if to not in self._deps[from_]:
            self._dependents[to].append(from_)
        self._deps[from_][to] = weight
        self.version += 1

    @property
    def components(self) -> List[str]:
        return list(self._deps)

    def get_dependencies(self, name: str) -> List[str]:
        return list(self._deps[name])

    def get_weight(self, from_: str, to: str) -> float:
        return self._deps[from_][to]

    def has_component(self, name: str) -> bool:
        return name in self._deps

    def edges(self) -> List[Tuple[str, str, float]]:
        return [(a, b, w) for a, ds in self._deps.items() for b, w in ds.items()]

    def get_topological_order(self) -> Optional[List[str]]:
        remaining = {v: len(ds) for v, ds in self._deps.items()}
        queue = deque(v for v, n in remaining.items() if n == 0)
        order: List[str] = []
        while queue:
            v = queue.popleft()
            order.append(v)
            for dependent in self._dependents[v]:
                remaining[dependent] -= 1
                if remaining[dependent] == 0:
                    queue.append(dependent)
        return order if len(order) == len(self._deps) else None

    def is_acyclic(self) -> bool:
        return self.get_topological_order() is not None

    @classmethod
    def from_text(cls, text: str) -> "DependencyGraph":
        graph = cls()
        line_re = re.compile(r"^(\S+)\s+зависит\s+от\s*(.*)$", re.IGNORECASE)
        item_re = re.compile(r"^(.+?)\s*(?:\(\s*(\d+(?:\.\d+)?)\s*\))?$")
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip().rstrip(".")
            if not line:
                continue
            m = line_re.match(line)
            if not m:
                graph.add_component(line)
                continue
            name, rest = m.group(1), m.group(2)
            graph.add_component(name)
            for item in re.split(r"[,;]|\s+и\s+", rest):
                item = item.strip()
                if not item or item.lower() == "ничего":
                    continue
                im = item_re.match(item)
                graph.add_dependency(name, im.group(1), float(im.group(2) or 0))
        return graph

    @classmethod
    def from_file(cls, path: str) -> "DependencyGraph":
        with open(path, encoding="utf-8") as f:
            return cls.from_text(f.read())


class DependencyAnalyzer:
    def __init__(self, graph: DependencyGraph) -> None:
        self.graph = graph
        self._cache: Dict[str, Set[str]] = {}
        self._cache_version = -1
        self._acyclic = True

    def _check(self, name: str) -> None:
        if not self.graph.has_component(name):
            raise KeyError(f"Компонент '{name}' не найден в графе")

    def _refresh_cache(self) -> None:
        if self._cache_version != self.graph.version:
            self._cache.clear()
            self._acyclic = self.graph.is_acyclic()
            self._cache_version = self.graph.version

    def find_dependencies_bfs(self, start: str, unique: bool = False) -> List[List[str]]:
        self._check(start)
        levels: List[List[str]] = []
        expanded = {start}
        seen = {start}
        frontier = [start]
        while frontier:
            level: List[str] = []
            in_level: Set[str] = set()
            for node in frontier:
                for dep in self.graph.get_dependencies(node):
                    if dep in in_level or (unique and dep in seen):
                        continue
                    in_level.add(dep)
                    level.append(dep)
            if not level:
                break
            levels.append(level)
            seen.update(level)
            frontier = [n for n in level if n not in expanded]
            expanded.update(frontier)
        return levels

    def find_dependencies_dfs(self, start: str, use_cache: bool = True) -> Set[str]:
        self._check(start)
        self._refresh_cache()
        if use_cache and self._acyclic:
            return set(self._dfs_cached(start))
        return self._dfs_plain(start)

    def _dfs_plain(self, start: str) -> Set[str]:
        result: Set[str] = set()

        def visit(node: str) -> None:
            for dep in self.graph.get_dependencies(node):
                if dep not in result:
                    result.add(dep)
                    visit(dep)

        visit(start)
        return result

    def _dfs_cached(self, node: str) -> Set[str]:
        if node in self._cache:
            return self._cache[node]
        result: Set[str] = set()
        for dep in self.graph.get_dependencies(node):
            result.add(dep)
            result |= self._dfs_cached(dep)
        self._cache[node] = result
        return result

    def find_critical_path(self) -> Tuple[List[str], float]:
        order = self.graph.get_topological_order()
        if order is None:
            raise ValueError("Обнаружен цикл: критический путь не определён")
        dist = {v: 0.0 for v in order}
        parent: Dict[str, Optional[str]] = {v: None for v in order}
        for v in order:
            for dep in self.graph.get_dependencies(v):
                cand = dist[dep] + self.graph.get_weight(v, dep)
                if cand > dist[v]:
                    dist[v], parent[v] = cand, dep
        end = max(order, key=lambda v: dist[v])
        path = [end]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        return path, dist[end]


def to_dot(graph: DependencyGraph, critical_path: Optional[List[str]] = None) -> str:
    on_path = set(zip(critical_path, critical_path[1:])) if critical_path else set()
    lines = ["digraph dependencies {", "  rankdir=LR;", "  node [shape=box, style=rounded];"]
    for v in graph.components:
        lines.append(f'  "{v}";')
    for a, b, w in graph.edges():
        attrs = []
        if w:
            attrs.append(f'label="{w:g}"')
        if (a, b) in on_path:
            attrs.append('color=red, penwidth=2')
        suffix = f" [{', '.join(attrs)}]" if attrs else ""
        lines.append(f'  "{a}" -> "{b}"{suffix};')
    lines.append("}")
    return "\n".join(lines)


def visualize(graph: DependencyGraph, filename: str = "dependencies",
              critical_path: Optional[List[str]] = None) -> None:
    dot = to_dot(graph, critical_path)
    with open(f"{filename}.dot", "w", encoding="utf-8") as f:
        f.write(dot)
    try:
        import graphviz
        graphviz.Source(dot).render(filename, format="png", cleanup=False)
        print(f"Граф сохранён: {filename}.png")
    except Exception as exc:
        print(f"PNG не создан ({exc.__class__.__name__}); DOT-файл: {filename}.dot")


EXAMPLE = """\
A зависит от B, C
B зависит от D
C зависит от D, E
E зависит от B
"""


def main(argv: List[str]) -> None:
    graph = DependencyGraph.from_file(argv[1]) if len(argv) > 1 else DependencyGraph.from_text(EXAMPLE)
    start = argv[2] if len(argv) > 2 else "A"
    analyzer = DependencyAnalyzer(graph)

    order = graph.get_topological_order()
    if order is None:
        print("Обнаружен цикл: проект нельзя собрать")
    else:
        print("Топологическая сортировка:", order)

    if graph.has_component(start):
        print(f"BFS для {start!r}:", analyzer.find_dependencies_bfs(start))
        print(f"DFS для {start!r}:", sorted(analyzer.find_dependencies_dfs(start)))

    path = None
    if order is not None:
        path, total = analyzer.find_critical_path()
        print(f"Критический путь: {' -> '.join(path)} (вес {total:g})")
    visualize(graph, "dependencies", path)


if __name__ == "__main__":
    main(sys.argv)