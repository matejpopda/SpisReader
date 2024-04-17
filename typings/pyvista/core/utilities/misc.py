from typing import TypeVar, Type

T = TypeVar("T")


def abstract_class(cls_: Type[T]) -> Type[T]: ...
