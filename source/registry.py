"""Named registries for pluggable model components."""


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, type] = {}

    def register(self, name: str):
        def decorator(cls):
            if name in self._items:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            cls.key = name
            self._items[name] = cls
            return cls
        return decorator

    def get(self, name: str) -> type:
        if name not in self._items:
            from .config import ConfigError
            raise ConfigError(f"unknown {self.kind} '{name}'; valid: {sorted(self._items)}")
        return self._items[name]

    def create(self, name: str, **params):
        return self.get(name)(**params)

    def names(self) -> list[str]:
        return list(self._items)
