import ast


class StmtTransformer(ast.NodeTransformer):
    """Base for transformers that operate on a flat list of statements."""

    def transform(self, stmts: list) -> list:
        mod = ast.Module(body=stmts, type_ignores=[])
        result = self.visit(mod)
        ast.fix_missing_locations(result)
        return result.body
