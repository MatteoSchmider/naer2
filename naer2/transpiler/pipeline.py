import ast
import inspect
from textwrap import dedent

from .analyzers.assigned_members import DynamicMembersAnalyzer
from .context import TranspilerContext
from .template import population_kernel


class PipelineManager:
    def __init__(self, stages, template_ast):
        self.stages = stages
        self.context = TranspilerContext(template_ast)

    def run(self, ast):

        for stage in self.stages:
            ast = stage(self.context)
            self.context.validate()

        return self.context


class PopulationTranspiler(PipelineManager):
    def __init__(self):
        super().__init__(
            stages=[
                DynamicMembersAnalyzer(),
            ],
            template_ast=ast.parse(dedent(inspect.getsource(population_kernel))),
        )
