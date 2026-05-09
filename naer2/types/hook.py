import ast
import inspect
from inspect import Parameter, signature
from logging import debug
from textwrap import dedent
from typing import Callable, Self, Set, Type

from ..transpiler.analyzers.assigned_members import AssignedMemberFinder


class Hook:
    def __init__(self, name: str, function: Callable):
        if not isinstance(name, str):
            raise ValueError("Hook name must be a string")
        self.name: str = name
        if not isinstance(function, Callable):
            raise ValueError("Hook function must be a Callable")
        self.function: Callable = function
        parsed_ast = ast.parse(dedent(inspect.getsource(function)))
        if not isinstance(parsed_ast, ast.Module):
            raise ValueError("Can't parse " + function + ", its source is invalid")
        if len(parsed_ast.body) != 1:
            raise ValueError("Can't parse " + function + ", its source is invalid")
        if not isinstance(parsed_ast.body[0], ast.FunctionDef):
            raise ValueError("Can't parse " + function + ", its source is invalid")
        self.ast: ast.FunctionDef = parsed_ast.body[0]
        self.assigned_members: Set[str] = AssignedMemberFinder(self.ast).run()

    def verify(self, allowed_types: Set[Type]) -> Self:
        sig = signature(self.function)
        params = list(sig.parameters.values())

        debug(self.name)
        debug(str(allowed_types))

        # Ensure 'self' is the first parameter if it is a parameter
        self_adjust = 0
        for index, param in enumerate(params):
            annotation_name = getattr(param.annotation, "__name__", str(param.annotation))
            debug(
                "Found parameter: "
                + str(param.name)
                + " , type: "
                + annotation_name
                + " , kind: "
                + str(param.kind)
                + " , default: "
                + str(param.default)
            )
            if param.name == "self":
                if index != 0:
                    raise TypeError(
                        f"{self.name} must have 'self' as the first parameter."
                    )
                else:
                    self_adjust = 1

        # Map annotated parameter types to their names (excludes unannotated params)
        type_to_name = {
            param.annotation: param.name
            for param in params[self_adjust:]
            if param.annotation != Parameter.empty
        }

        # All annotations must be from the allowed set
        if set(type_to_name.keys()) - allowed_types:
            raise TypeError(
                f"{self.name} arguments can only use types inside the allowed set."
            )

        # Required parameters (no default) must be annotated.
        # Optional parameters (with a default) may be unannotated — the framework
        # will not dispatch to them by type, but they can still receive values
        # positionally (e.g. the optional target_region in pattern callbacks).
        for param in params[self_adjust:]:
            if param.annotation == Parameter.empty and param.default == Parameter.empty:
                raise TypeError(
                    f"Required parameter '{param.name}' of '{self.name}' must be "
                    f"annotated with an allowed type."
                )

        # Annotated parameters must use unique types (the framework dispatches by type)
        annotated = [p for p in params[self_adjust:] if p.annotation != Parameter.empty]
        if len(type_to_name) != len(annotated):
            raise TypeError(
                f"Annotated parameters of '{self.name}' must each use a unique type "
                f"(found duplicate annotations)."
            )

        # We now have a valid hook with the function and a map to get the parameter names
        self.parameter_type_to_name: dict[Type, str] = type_to_name
        return self
