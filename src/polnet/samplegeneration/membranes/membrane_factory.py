class MembraneFactory:
    _registry = {}

    @classmethod
    def register(cls, name):
        """Decorator to register a membrane generator class with a given name."""

        def inner(subclass):
            cls._registry[name] = subclass
            return subclass

        return inner

    @classmethod
    def create(cls, mb_type, params):
        """Create an instance of a membrane generator based on the type and parameters."""
        if mb_type not in cls._registry:
            raise ValueError(f"Membrane type '{mb_type}' is not registered.")
        return cls._registry[mb_type](params)
