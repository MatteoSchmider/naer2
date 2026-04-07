import ast
import inspect
from textwrap import dedent
import sys

node_types = {}


class Node:
    def __init__(self, cls):
        self.original = cls
        self.ast = ast.parse(dedent(inspect.getsource(cls)))
        self.members = []
        self.funcs = {}
        self.codes = {}
        self.register_methods()
        if self.funcs["__init__"] is None:
            raise ValueError("Node must have an __init__ method")
        # print(ast.dump(self.init, indent=4))
        # print(ast.dump(self.send, indent=4))
        # print(ast.dump(self.receive, indent=4))
        # print(ast.dump(self.update, indent=4))

    def register_methods(self):
        for node in ast.walk(self.ast):
            if isinstance(node, ast.FunctionDef):
                if node.name == "__init__":
                    self.funcs["__init__"], self.members = modify_attribute_writes(
                        node, self.members
                    )
                    self.codes["__init__"] = (
                        ast.unparse(self.funcs["__init__"])
                        + "\n"
                        + "__init__(self, *args, **kwds)"
                    )
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Attribute):
                        decorator_str = f"{decorator.value.id}.{decorator.attr}"
                        print(decorator_str)
                        if decorator_str == "naer.send":
                            node.decorator_list.remove(decorator)
                            self.funcs["send"], self.members = modify_attribute_writes(
                                node, self.members
                            )
                            self.codes["send"] = (
                                ast.unparse(self.funcs["send"])
                                + "\n"
                                + "send(self, *args, **kwds)"
                            )
                        elif decorator_str == "naer.receive":
                            node.decorator_list.remove(decorator)
                            self.funcs["receive"], self.members = (
                                modify_attribute_writes(node, self.members)
                            )
                            self.codes["receive"] = (
                                ast.unparse(self.funcs["receive"])
                                + "\n"
                                + "receive(self, *args, **kwds)"
                            )
                        elif decorator_str == "naer.update":
                            node.decorator_list.remove(decorator)
                            node.name = "update"
                            self.funcs["update"], self.members = (
                                modify_attribute_writes(node, self.members)
                            )
                            self.codes["update"] = (
                                ast.unparse(self.funcs["update"])
                                + "\n"
                                + "update(self, *args, **kwds)"
                            )

    def run_modified(self, node, func, args, kwds):
        if func == "__init__" and node.__dict__:
            raise ValueError("Node already initialized")
        if func not in self.funcs.keys():
            raise AttributeError("Attribute does not exist")
        locals = {"self": node, "args": args, "kwds": kwds}
        code = self.codes[func]
        # print("Executing the following code as " + func + ":")
        # print(code)
        # print("with locals: ", locals)
        exec(code, globals(), locals)


def modify_attribute_writes(func: ast.FunctionDef, members: list = []):
    current_body = func.body
    for node in ast.walk(func):
        if hasattr(node, "body"):
            current_body = node.body
        if isinstance(node, ast.Assign):
            index = current_body.index(node)
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    if target.value.id == "self":
                        members.append(target.attr)
                        new_node = ast.Expr(
                            value=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id="object", ctx=ast.Load()),
                                    attr="__setattr__",
                                    ctx=ast.Load(),
                                ),
                                args=[
                                    ast.Name(id="self", ctx=ast.Load()),
                                    ast.Constant(value=target.attr),
                                    node.value,
                                ],
                                keywords=[],
                            )
                        )
                        current_body.insert(index, new_node)
            current_body.remove(node)
        if isinstance(node, ast.AnnAssign):
            index = current_body.index(node)
            target = node.target
            if isinstance(target, ast.Attribute):
                if target.value.id == "self":
                    members.append(target.attr)
                    new_node = ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="object", ctx=ast.Load()),
                                attr="__setattr__",
                                ctx=ast.Load(),
                            ),
                            args=[
                                ast.Name(id="self", ctx=ast.Load()),
                                ast.Constant(value=target.attr),
                                node.value,
                            ],
                            keywords=[],
                        )
                    )
                    current_body.insert(index, new_node)
            current_body.remove(node)
        if isinstance(node, ast.AugAssign):
            index = current_body.index(node)
            target = node.target
            if isinstance(target, ast.Attribute):
                if target.value.id == "self":
                    members.append(target.attr)
                    new_node = ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="object", ctx=ast.Load()),
                                attr="__setattr__",
                                ctx=ast.Load(),
                            ),
                            args=[
                                ast.Name(id="self", ctx=ast.Load()),
                                ast.Constant(value=target.attr),
                                ast.BinOp(target, node.op, node.value),
                            ],
                            keywords=[],
                        )
                    )
                    current_body.insert(index, new_node)
            current_body.remove(node)
    return func, list(set(members))


def setattr(node, name, value):
    raise AttributeError("Nodes may not be modified outside of naer code")


def node(cls):
    node_types[cls] = Node(cls)
    cls.__init__ = lambda node, *args, **kwds: node_types[cls].run_modified(
        node, "__init__", args, kwds
    )
    cls.send = lambda node, *args, **kwds: node_types[cls].run_modified(
        node, "send", args, kwds
    )
    cls.receive = lambda node, *args, **kwds: node_types[cls].run_modified(
        node, "receive", args, kwds
    )
    cls.update = lambda node, *args, **kwds: node_types[cls].run_modified(
        node, "update", args, kwds
    )
    cls.__setattr__ = setattr
    return cls


def send(func, *args, **kwds):
    return func


def receive(func, *args, **kwds):
    return func


def update(func, *args, **kwds):
    print(func.__name__)
    print(func.__module__)
    print(func.__globals__)
    print(func.__qualname__)
    @update_marker
    def wrapper(*args, **kwds):
        return func(args, kwds)

    return wrapper


def update_marker(func):
    return func


def send_spike():
    pass
