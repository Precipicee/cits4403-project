"""Named registries for pluggable model components."""


class Registry:
    """Maps names to interchangeable implementations of one model component, so a configuration
    can choose a rule by name, e.g. ``"absolute"`` severance or the ``"power"`` kernel."""

    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, type] = {}

    def register(self, name: str):
        """Class decorator that registers the class under ``name``; names must be unique."""
        def decorator(cls):
            if name in self._items:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            cls.key = name
            self._items[name] = cls
            return cls
        return decorator

    def get(self, name: str) -> type:
        """The class registered as ``name``; raises ``ConfigError`` listing valid names if none."""
        if name not in self._items:
            from .config import ConfigError
            raise ConfigError(f"unknown {self.kind} '{name}'; valid: {sorted(self._items)}")
        return self._items[name]

    def create(self, name: str, **params):
        """An instance of the class registered as ``name``."""
        return self.get(name)(**params)

    def names(self) -> list[str]:
        """Registered names, in registration order."""
        return list(self._items)
