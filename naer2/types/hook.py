import ast
import inspect
from inspect import Parameter, signature
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

        print(allowed_types)

        # Ensure 'self' is the first parameter if it is a parameter
        self_adjust = 0
        for index, param in enumerate(params):
            print(param)
            if param.name == "self":
                if index != 0:
                    raise TypeError(
                        f"{self.name} must have 'self' as the first parameter."
                    )
                else:
                    self_adjust = 1

        # Map type annotation types to their parameter names
        type_to_name = {
            param.annotation: param.name
            for param in params[self_adjust:]
            if param.annotation != Parameter.empty
        }

        # Ensure all annotations are allowed and unique
        if set(type_to_name.keys()) - allowed_types:
            raise TypeError(
                f"{self.name} arguments can only use types inside the allowed set."
            )

        # Ensure all parameters are annotated
        if len(type_to_name) != len(params) - self_adjust:
            raise TypeError(
                f"All parameters (except 'self') of {self.name} must be uniquely annotated with allowed types."
            )
        # We now have a valid hook with the function and a map to get the parameter names
        self.parameter_type_to_name: dict[Type, str] = type_to_name
        return self
