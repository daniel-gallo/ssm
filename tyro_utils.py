from functools import reduce
from typing import Annotated, Dict, Type

import tyro
from typing_extensions import get_args, get_origin
from tyro.constructors import (
    ConstructorRegistry,
    PrimitiveConstructorSpec,
    PrimitiveTypeInfo,
    UnsupportedTypeAnnotationError,
)

from hps import Hyperparams

registry = ConstructorRegistry()


@registry.primitive_rule
def custom_tuple_rule(
    type_info: PrimitiveTypeInfo,
) -> PrimitiveConstructorSpec | UnsupportedTypeAnnotationError | None:
    # Check if the type is a tuple (fixed or variable-length)
    if get_origin(type_info.type) is not tuple:
        return None

    type_args = get_args(type_info.type)
    is_variable_length = len(type_args) == 2 and type_args[1] is Ellipsis

    # Handle variable-length tuples (e.g., tuple[int, ...])
    if is_variable_length:
        contained_type = type_args[0]
        inner_spec = ConstructorRegistry.get_primitive_spec(
            PrimitiveTypeInfo.make(contained_type, type_info.markers)
        )
        if isinstance(inner_spec, UnsupportedTypeAnnotationError):
            return inner_spec

        def instance_from_str(args: list[str]) -> tuple:
            # Split comma-separated args and flatten
            parts = []
            for arg in args:
                parts.extend(arg.split(","))
            return tuple(inner_spec.instance_from_str([part]) for part in parts)

        def str_from_instance(instance: tuple) -> list[str]:
            return [
                ",".join(inner_spec.str_from_instance(e)[0] for e in instance)
            ]

        metavar = f"{inner_spec.metavar},..."

    # Handle fixed-length tuples (e.g., tuple[int, str])
    else:
        inner_specs = []
        for t in type_args:
            spec = ConstructorRegistry.get_primitive_spec(
                PrimitiveTypeInfo.make(t, type_info.markers)
            )
            if isinstance(spec, UnsupportedTypeAnnotationError):
                return spec
            inner_specs.append(spec)

        def instance_from_str(args: list[str]) -> tuple:
            # Split comma-separated args and flatten
            parts = []
            for arg in args:
                parts.extend(arg.split(","))
            if len(parts) != len(inner_specs):
                raise ValueError(
                    f"Expected {len(inner_specs)} elements, got {len(parts)}"
                )
            return tuple(
                spec.instance_from_str([part])
                for spec, part in zip(inner_specs, parts)
            )

        def str_from_instance(instance: tuple) -> list[str]:
            return [
                ",".join(
                    spec.str_from_instance(elem)[0]
                    for spec, elem in zip(inner_specs, instance)
                )
            ]

        metavar = ",".join(spec.metavar for spec in inner_specs)

    return PrimitiveConstructorSpec(
        nargs="*",  # Allow both space-separated and comma-separated inputs
        metavar=metavar,
        instance_from_str=instance_from_str,
        str_from_instance=str_from_instance,
        is_instance=lambda x: isinstance(x, tuple)
        and (
            (is_variable_length and all(inner_spec.is_instance(e) for e in x))
            or (not is_variable_length and len(x) == len(inner_specs))
        ),
    )


def get_hyperparams(command_to_hp: Dict[str, Type[Hyperparams]]):
    """
    This exists for two reasons:
        * Disable
           --flag / --no-flag
          and enable
           --flag True / --flag False

        * Make it possible to pass tuples in two ways:
            * --flag 1,2,3
            * --flag 1 2 3 (default way)

          This works for variable-length tuples, e.g. Tuple[int, ...]
          and fixed-length tuples, e.g. Tuple[str, int].
    """

    def annotated_subcommand(name, cls):
        return Annotated[
            cls, tyro.conf.subcommand(name), tyro.conf.FlagConversionOff
        ]

    subcommands = [
        annotated_subcommand(command, hp)
        for command, hp in command_to_hp.items()
    ]
    with registry:
        return tyro.cli(reduce(lambda a, b: a | b, subcommands))
