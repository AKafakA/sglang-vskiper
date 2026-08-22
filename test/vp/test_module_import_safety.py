"""Every vpipe module must be importable without a NameError.

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
        else:
            # assignments, for-targets, with-items, except-names that really do
            # execute at module scope
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        bound.update(_targets(t))
                elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
                    bound.update(_targets(sub.target))
                elif isinstance(sub, (ast.For, ast.AsyncFor, ast.comprehension)):
                    bound.update(_targets(sub.target))
                elif isinstance(sub, ast.withitem) and sub.optional_vars is not None:
                    bound.update(_targets(sub.optional_vars))
                elif isinstance(sub, ast.ExceptHandler) and sub.name:
                    bound.add(sub.name)
                elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for alias in sub.names:
                        bound.add(alias.asname or alias.name.split(".")[0])
    return bound


def module_level_loads(tree):
    """Names LOADED by statements that execute at import, skipping nested scopes."""
    loads = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # a function BODY runs on call, not on import; its decorators and
            # default expressions run now
            parts = list(stmt.decorator_list) + list(stmt.args.defaults) + [
                d for d in stmt.args.kw_defaults if d is not None
            ]
            for part in parts:
                for n in ast.walk(part):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                        loads.append((n.id, n.lineno))
            continue
        if isinstance(stmt, ast.ClassDef):
            # A CLASS BODY DOES EXECUTE AT IMPORT, and so do its bases, keywords
            # and decorators. Treating ClassDef like FunctionDef meant
            # `class C(MissingBase)` -- or any missing name in a class body --
            # passed this checker and still failed to import, leaving the
            # catastrophic failure class only half covered.
            for part in (list(stmt.decorator_list) + list(stmt.bases)
                         + [k.value for k in stmt.keywords]):
                for n in ast.walk(part):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                        loads.append((n.id, n.lineno))
            # the body executes in its own namespace, but names it LOADS still
            # resolve outward to module scope
            class_local = set()
            for sub in stmt.body:
                if isinstance(sub, ast.Assign):
                    for t in sub.targets: class_local.update(_targets(t))
                elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
                    class_local.update(_targets(sub.target))
                elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    class_local.add(sub.name)
            for sub in stmt.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue                      # method bodies run on call
                for n in ast.walk(sub):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) \
                       and n.id not in class_local:
                        loads.append((n.id, n.lineno))
            continue
        stack = [stmt]
        while stack:
            node = stack.pop()
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    loads.append((child.id, child.lineno))
                stack.append(child)
    return loads


def unbound_module_level_names(path):
    tree = ast.parse(path.read_text(), str(path))
    bound = module_level_bindings(tree) | SAFE
    return sorted({(n, ln) for n, ln in module_level_loads(tree) if n not in bound})


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
