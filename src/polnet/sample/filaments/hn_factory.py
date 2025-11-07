class HnFactory:
    """Factory class to create helicoidal structure generator instances based on type.
    """

    """Registry of helicoidal structure generator classes.
    """
    __registry = {}

    @classmethod
    def register(cls, name):
        """Decorator to register a membrane generator class with a given name.
        
        Args:
            name (str): The name to register the membrane generator class under.
        
        Returns:
            function: The decorator function.
        """

        def inner(subclass):
            cls.__registry[name] = subclass
            return subclass
        return inner

    @classmethod
    def create(cls, hn_type, params):
        """Create an instance of a helicoidal structure generator based on the type and parameters.

        Args:
            hn_type (str): The type of helicoidal structure generator to create.
            params (dict): The parameters to initialize the helicoidal structure generator.

        Returns:
            HnGen: An instance of the requested helicoidal structure generator.
        """
        if hn_type not in cls.__registry:
            raise ValueError(f"Helicoidal structure type '{hn_type}' is not registered. Available types: {list(cls.__registry.keys())}")
        return cls.__registry[hn_type].from_params(params)
