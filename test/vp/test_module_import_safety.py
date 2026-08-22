"""Fast STATIC pre-check for import-time NameErrors.

NOT the authoritative gate. test_package_imports.py imports every module for
real, in a subprocess, and that is the ground truth. This file exists because
the package's dependencies are not installed on the source-edit box, so a real
import cannot run there -- this catches the common cases seconds after an edit
instead of minutes later on a GPU host.

Treat a disagreement between the two as a bug in THIS file. It approximates
Python's scoping rules, and that approximation has been wrong in six distinct
ways (class-body annotations, mutually exclusive branch scopes, aug-assign
targets, walrus ordering, try/else paths, lambda parameters). A real import
cannot be wrong about semantics because it does not model them.

Original rationale follows.

Every vpipe module must be importable without a NameError.

This exists because a refactor dedented a function body out to module scope:
the function ended after binding its first local, and the loops that followed
became MODULE-LEVEL statements referencing that local. The module still parsed
-- it is valid Python -- but importing it raised NameError, which meant no
server could start at all, for any model.

ast.parse() passed. A scope-blind "free names" check also passed, because it
collected bindings from anywhere in the tree, so a name bound inside some
function counted as bound at module level. Both were necessary and neither was
sufficient.

This check is scope-aware: it looks ONLY at module-level statements and ONLY at
module-level bindings, descending into nested function and class bodies for
neither. A name loaded by a statement that runs at import time, and never bound
at import time, is an import-time NameError waiting to happen.
"""
import ast
import builtins
import pathlib
import sys

PKG = pathlib.Path(__file__).resolve().parents[2] / "python" / "sglang" / "srt" / "vpipe"

SAFE = set(dir(builtins)) | {
    "__file__", "__name__", "__doc__", "__package__", "__spec__",
    "__loader__", "__builtins__", "__all__", "annotations",
}


def _targets(node):
    for x in ast.walk(node):
        if isinstance(x, ast.Name):
            yield x.id


def module_level_bindings(tree):
    bound = set()
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for alias in stmt.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(stmt.name)
        elif isinstance(stmt, ast.If):
            # DEFINITE assignment: a name counts as bound after an if only if
            # BOTH arms bind it. Unioning the branches (what ast.walk does) let
            # a name bound only in the else-arm satisfy later statements.
            then_b = module_level_bindings(ast.Module(body=stmt.body, type_ignores=[]))
            else_b = (module_level_bindings(ast.Module(body=stmt.orelse, type_ignores=[]))
                      if stmt.orelse else set())
            bound |= (then_b & else_b) if stmt.orelse else set()
        elif isinstance(stmt, ast.Try):
            # try/except binds only what EVERY completing path binds. The common
            # `try: import x / except ImportError: x = None` binds x on both, so
            # it stays clean; a name bound only in the try body does not.
            body_b = module_level_bindings(ast.Module(body=stmt.body, type_ignores=[]))
            handler_bs = [
                module_level_bindings(ast.Module(body=h.body, type_ignores=[]))
                for h in stmt.handlers
            ]
            # The else-clause runs only when NO exception occurred, so its
            # bindings belong to the no-exception path -- they must still be
            # intersected with every handler path. Unioning them marked
            # `try: f() / except: pass / else: x = 1` as binding x, though the
            # handled path reaches the next statement without it.
            else_b = (module_level_bindings(ast.Module(body=stmt.orelse, type_ignores=[]))
                      if stmt.orelse else set())
            common = body_b | else_b
            for hb in handler_bs:
                common &= hb
            if stmt.finalbody:
                # finally always runs, so what it binds IS definite
                common |= module_level_bindings(
                    ast.Module(body=stmt.finalbody, type_ignores=[]))
            bound |= common
        elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            # a loop body may execute zero times, so it binds nothing definitely
            pass
        else:
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        bound.update(_targets(t))
                elif isinstance(sub, ast.AnnAssign):
                    # `x: int` with NO value binds nothing at runtime -- it only
                    # records an annotation, so a later load of x still raises.
                    if sub.value is not None:
                        bound.update(_targets(sub.target))
                elif isinstance(sub, ast.AugAssign):
                    # `x += 1` LOADS x before storing it; it cannot introduce a
                    # binding, and the load is caught by the visitor.
                    pass
                elif isinstance(sub, ast.withitem) and sub.optional_vars is not None:
                    bound.update(_targets(sub.optional_vars))
                elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for alias in sub.names:
                        bound.add(alias.asname or alias.name.split(".")[0])
    return bound


def _ordered_loads(stmt, outer_bound):
    """Loads in `stmt` that nothing has bound yet, respecting lexical scopes.

    Comprehension and lambda parameters apply ONLY inside their own bodies; a
    walrus target becomes available only after the expression that binds it.
    """
    bad = []

    def visit(node, scope):
        if isinstance(node, ast.Lambda):
            a = node.args
            inner = set(scope)
            inner.update(x.arg for x in
                         list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs))
            if a.vararg: inner.add(a.vararg.arg)
            if a.kwarg: inner.add(a.kwarg.arg)
            # defaults evaluate in the ENCLOSING scope, body in the inner one
            for d in list(a.defaults) + [d for d in a.kw_defaults if d is not None]:
                visit(d, scope)
            visit(node.body, inner)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            inner = set(scope)
            for i, gen in enumerate(node.generators):
                # the first iterable is evaluated in the ENCLOSING scope
                visit(gen.iter, scope if i == 0 else inner)
                inner.update(_targets(gen.target))
                for cond in gen.ifs:
                    visit(cond, inner)
            for part in ([node.key, node.value] if isinstance(node, ast.DictComp)
                         else [node.elt]):
                visit(part, inner)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # decorators and defaults evaluate NOW; the body runs on call
            a = node.args
            for part in (list(node.decorator_list) + list(a.defaults)
                         + [d for d in a.kw_defaults if d is not None]):
                visit(part, scope)
            scope.add(node.name)
            return
        if isinstance(node, ast.ClassDef):
            # bases, keywords and decorators evaluate NOW, in this scope; the
            # body executes immediately too, in its own namespace, in order
            for part in (list(node.decorator_list) + list(node.bases)
                         + [k.value for k in node.keywords]):
                visit(part, scope)
            inner = set(scope)
            for sub in node.body:
                visit(sub, inner)
                if isinstance(sub, ast.Assign):
                    for t in sub.targets: inner.update(_targets(t))
                elif isinstance(sub, ast.AnnAssign):
                    # `class C: x: int` binds nothing at class-execution time,
                    # so a later `y = x` in the same body still raises.
                    if sub.value is not None:
                        inner.update(_targets(sub.target))
                elif isinstance(sub, ast.AugAssign):
                    inner.update(_targets(sub.target))
            scope.add(node.name)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in scope:
                bad.append((node.id, node.lineno))
            return
        if isinstance(node, ast.NamedExpr):
            visit(node.value, scope)
            scope.update(_targets(node.target))
            return
        if isinstance(node, ast.AugAssign):
            # `x += 1` READS x before storing it, but the target node carries
            # Store context, so the generic Name branch never sees the load.
            visit(node.value, scope)
            for name in _targets(node.target):
                if name not in scope:
                    bad.append((name, node.lineno))
            scope.update(_targets(node.target))
            return
        if isinstance(node, ast.If):
            # MUTUALLY EXCLUSIVE arms. Sharing one mutable scope leaked the
            # then-arm's bindings into the else-arm, so `if F: def x(): pass
            # else: y = x` looked clean. Visit each arm with its own copy and
            # merge by intersection (definite assignment).
            visit(node.test, scope)
            then_scope, else_scope = set(scope), set(scope)
            for sub in node.body: visit(sub, then_scope)
            for sub in node.orelse: visit(sub, else_scope)
            scope |= (then_scope & else_scope) if node.orelse else set()
            return
        if isinstance(node, ast.Try):
            body_scope = set(scope)
            for sub in node.body: visit(sub, body_scope)
            for sub in node.orelse: visit(sub, body_scope)
            handler_scopes = []
            for h in node.handlers:
                hs = set(scope)
                if h.name: hs.add(h.name)
                for sub in h.body: visit(sub, hs)
                handler_scopes.append(hs)
            common = body_scope
            for hs in handler_scopes:
                common &= hs
            scope |= common
            for sub in node.finalbody: visit(sub, scope)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    scope = set(outer_bound)
    if isinstance(stmt, ast.ExceptHandler) and stmt.name:
        scope.add(stmt.name)
    visit(stmt, scope)
    return bad

def unbound_module_level_names(path):
    """Names loaded at import before anything binds them.

    Collecting every module binding up front made `value = later; later = 1`
    look safe, though executing the module raises NameError on the first
    statement -- the same ordering bug that was fixed inside class bodies. Walk
    module statements in order: check each statement's import-time loads, THEN
    record what it binds.
    """
    tree = ast.parse(path.read_text(), str(path))
    bound = set(SAFE)
    bad = []
    for stmt in tree.body:
        sub = ast.Module(body=[stmt], type_ignores=[])
        # Comprehension and lambda locals are LEXICAL: they are in scope only
        # inside their own bodies. Collecting them statement-wide suppressed
        # real errors -- `value = (missing, lambda missing: missing)` looked
        # clean because the lambda parameter masked the tuple's load, though
        # Python raises on the first element. Walk with a scope stack instead.
        for name, lineno in _ordered_loads(stmt, bound):
            bad.append((name, lineno))
        bound |= module_level_bindings(sub)
    return sorted(set(bad))


def scan():
    problems = {}
    for p in sorted(PKG.glob("*.py")):
        bad = unbound_module_level_names(p)
        if bad:
            problems[p.name] = bad
    return problems


def test_no_import_time_nameerror():
    # The package must actually have been found. Without this the test passes
    # when PKG resolves to the wrong place: zero modules scanned, no problems
    # found, green. A check whose subject can be empty is not a check.
    modules = list(PKG.glob("*.py"))
    assert len(modules) >= 20, (
        f"expected the vpipe package at {PKG}, found {len(modules)} modules"
    )
    problems = scan()
    assert not problems, "module-level names that would NameError on import: " + repr(problems)


if __name__ == "__main__":
    modules = list(PKG.glob("*.py"))
    if len(modules) < 20:
        print(f"  FATAL: expected the vpipe package at {PKG}, found {len(modules)} modules")
        sys.exit(2)
    found = scan()
    for name, bad in found.items():
        for n, ln in bad:
            print(f"  {name}:{ln}  unbound at module scope: {n}")
    print(f"MODULE IMPORT SAFETY: {'FAIL' if found else 'PASS'} ({len(list(PKG.glob('*.py')))} modules)")
    sys.exit(1 if found else 0)
