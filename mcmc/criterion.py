from .event import Event


class AcceptanceCriterion:
    def __init__(self) -> None:
        ...

    def __call__(self, event: Event):
        ...
