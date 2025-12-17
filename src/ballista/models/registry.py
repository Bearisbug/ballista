from typing import Dict, Type


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, Type] = {}

    def register(self, name: str):
        def deco(cls):
            if name in self._items:
                raise KeyError(f"{name} already registered in {self.name}")
            self._items[name] = cls
            return cls
        return deco

    def get(self, name: str) -> Type:
        if name not in self._items:
            raise KeyError(f"{name} is not registered in {self.name}. Available={sorted(self._items.keys())}")
        return self._items[name]


MODELS = Registry("models")
LOSSES = Registry("losses")
