import ast
import inspect
from textwrap import dedent


class ClassTranspiler:
    def __init__(self, cls):
        if not isinstance(cls, type):
            raise ValueError(
                "Can't parse " + cls + " with " + self.__class__.__name__ + " decorator"
            )
        potential_ast = ast.parse(dedent(inspect.getsource(cls)))
        if not isinstance(potential_ast, ast.Module):
            raise ValueError(
                "Can't parse " + cls + " with " + self.__class__.__name__ + " decorator"
            )
        if len(potential_ast.body) != 1:
            raise ValueError(
                "Can't parse " + cls + " with " + self.__class__.__name__ + " decorator"
            )
        if not isinstance(potential_ast.body[0], ast.ClassDef):
            raise ValueError(
                "Can't parse " + cls + " with " + self.__class__.__name__ + " decorator"
            )
        if len(potential_ast.body[0].decorator_list) != 1:
            raise ValueError(
                "The "
                + self.__class__.__name__
                + " decorator is not compatible with other decorators"
            )
        self.cls = cls
        self.ast = potential_ast.body[0]
        self.naer_alias = self.ast.decorator_list[0].value.id
        self.funcs = {}
        self.parse_methods()

    def debug(self):
        print(ast.dump(self.ast, indent=4))
        print(ast.unparse(self.ast))
        print(self.funcs)
        print(self.static_attrs)
        print(self.dynamic_attrs)

    def compile(self, instance):
        self.parse_members(instance)
        self.replace_values()
        self.debug()

    def parse_methods(self):
        for node in ast.walk(self.ast):
            if isinstance(node, ast.FunctionDef):
                if node.name == "__init__":
                    self.funcs["__init__"] = node
                elif node.decorator_list:
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Attribute):
                            if decorator.value.id == self.naer_alias:
                                self.funcs[decorator.attr] = node

    def parse_members(self, instance):
        # parse which members are dynamic/static from source ast
        # this also finds constants and initializes the instance values dict
        finder = AssignedMemberFinder()
        for name, func in self.funcs.items():
            if name == "__init__":
                finder.find(func, all_static=True)
            else:
                finder.find(func)
        self.static_attrs = finder.static_values
        self.dynamic_attrs = finder.dynamic_attrs
        # now complete the instance values dicts with actual instance values
        for member in finder.static_attrs.difference(finder.dynamic_attrs):
            if hasattr(instance, member):
                self.static_attrs[member] = getattr(instance, member)
        for member in finder.dynamic_attrs:
            if hasattr(instance, member):
                self.dynamic_attrs[member] = getattr(instance, member)
            else:
                self.dynamic_attrs[member] = 0

    def replace_values(self):
        transformer = StaticMemberTransformer(self.static_attrs)
        for name, func in self.funcs.items():
            if name != "__init__":
                self.funcs[name] = transformer.visit(func)


class AssignedMemberFinder(ast.NodeVisitor):
    def __init__(self):
        self.static_attrs = set()
        self.dynamic_attrs = set()
        self.static_values = {}
        self.all_static = False

    def find(self, node, all_static=False):
        self.all_static = all_static
        return super().visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                if isinstance(target.value, ast.Name):
                    if target.value.id == "self":
                        if isinstance(node.value, ast.Constant):
                            self.static_attrs.add(target.attr)
                            self.static_values[target.attr] = node.value.value
                        elif self.all_static:
                            self.static_attrs.add(target.attr)
                        else:
                            self.dynamic_attrs.add(target.attr)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Attribute):
            if isinstance(node.target.value, ast.Name):
                if node.target.value.id == "self":
                    if isinstance(node.value, ast.Constant):
                        self.static_attrs.add(node.target.attr)
                        self.static_values[node.target.attr] = node.value.value
                    elif self.all_static:
                        self.static_attrs.add(node.target.attr)
                    else:
                        self.dynamic_attrs.add(node.target.attr)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Attribute):
            if isinstance(node.target.value, ast.Name):
                if node.target.value.id == "self":
                    if self.all_static:
                        self.static_attrs.add(node.target.attr)
                    else:
                        self.dynamic_attrs.add(node.target.attr)


class StaticMemberTransformer(ast.NodeTransformer):
    def __init__(self, static_members: dict):
        super().__init__()
        self.static_members = static_members

    def visit(self, node):
        node = super().visit(node)
        # first pass replaces all occurences of static attributes with their values
        # but this can create problems when a static attribute is assigned to a constant
        # then a constant = constant assignment is created
        # a second pass with this transformer removes such instances
        return super().visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            if node.value.id == "self":
                if node.attr in self.static_members.keys():
                    return ast.Constant(value=self.static_members[node.attr])
        return self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Constant):
                return None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Constant):
            return None
        return self.generic_visit(node)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Constant):
            return None
        return self.generic_visit(node)
