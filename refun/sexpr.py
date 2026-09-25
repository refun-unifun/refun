"""Tree-sitter S-expression views of decompiled C.

The AST view is the third of ReFuN's four encoder inputs: the stripped
decompiled function parsed with the tree-sitter C grammar and linearised.
Three linearisations are used, matching three corpus columns:

    S-Expression_of_decompiled_code_stripped   sexpr_with_text()   model input
    S-Expression_decompiled_code_*_clean       sexpr_clean()       structure only
    Root Node                                  sexpr_fields()      dedup only

`sexpr_clean` collapses every identifier, type and literal to IDENT/TYPE/LIT,
so it answers "does structure alone help?" without surface-token memorisation.

Ghidra output is not always valid C. tree-sitter is error-tolerant and yields
ERROR nodes rather than failing, which is what we want; `parse_status` reports
parse health so a broken decompiler run is visible rather than silent.

Fidelity: with tree_sitter 0.26 / tree_sitter_c 0.24, `sexpr_with_text`
reproduces the published x64_O0 column byte-for-byte on 98.5% of records
(n=200). The residue is error-recovery drift between grammar versions on input
that already fails to parse, and only affects such input. `sexpr_fields` is
more version-sensitive (49%) but is never a model input.
"""
from typing import Optional

_PARSER = None

# Named leaf nodes that carry a program identifier or a literal value. These are
# exactly the nodes `sexpr_clean` collapses -- everything that could let the
# model match on surface text rather than structure.
_IDENT_NODES = frozenset({"identifier", "field_identifier", "statement_identifier"})
_TYPE_NODES = frozenset({
    "type_identifier", "primitive_type", "sized_type_specifier",
})
_LITERAL_NODES = frozenset({
    "number_literal", "string_literal", "char_literal", "concatenated_string",
    "true", "false", "null",
})


def _parser():
    """Build the C parser on first use. Imported lazily so training on the
    published corpus does not require tree-sitter at all."""
    global _PARSER
    if _PARSER is None:
        try:
            import tree_sitter_c
            from tree_sitter import Language, Parser
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "S-expression generation needs tree-sitter:\n"
                "    pip install tree_sitter tree_sitter_c\n"
                "It is only required when *building* a corpus; training on the "
                "published corpus uses the stored S-expression column."
            ) from e
        _PARSER = Parser(Language(tree_sitter_c.language()))
    return _PARSER


def _quote(text: str, strict: bool = False) -> str:
    """Source span as a quoted token.

    The published corpus escapes only newlines, leaving embedded quotes and
    backslashes raw. That is reproduced by default so the stored columns
    regenerate byte-for-byte; `strict=True` produces a properly escaped span
    for a new corpus. The model never re-parses the span, so this affects
    downstream tooling only.
    """
    if strict:
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    return '"' + text.replace("\n", "\\n") + '"'


def sexpr_with_text(code: str, max_nodes: int = 20000, strict_escape: bool = False) -> str:
    """The model-facing view: `(type "source" child ...)` for internal nodes,
    bare type for named leaves, literal text for anonymous tokens.

    This is the format stored in `S-Expression_of_decompiled_code_stripped`.
    """
    if not code or not code.strip():
        return ""
    src = code.encode("utf-8", "replace")
    root = _parser().parse(src).root_node
    budget = [max_nodes]

    def walk(node) -> str:
        if budget[0] <= 0:
            return ""
        budget[0] -= 1
        if node.child_count == 0:
            # Named leaves (identifier, primitive_type, ...) contribute their
            # type; anonymous leaves ( '{', ';', 'return' ) contribute the token.
            if node.is_named:
                return node.type
            return src[node.start_byte:node.end_byte].decode("utf-8", "replace")
        text = src[node.start_byte:node.end_byte].decode("utf-8", "replace")
        parts = [node.type, _quote(text, strict_escape)]
        for i in range(node.child_count):
            piece = walk(node.child(i))
            if piece:
                parts.append(piece)
            if budget[0] <= 0:
                break
        return "(" + " ".join(parts) + ")"

    return walk(root)


def sexpr_clean(code: str, max_nodes: int = 20000) -> str:
    """Structure-only view. Identifiers, type names and literals are replaced by
    `IDENT` / `TYPE` / `LIT`, so no surface token survives.

    This is the format stored in the `*_clean` columns and is the right input
    for the "structure without surface text" ablation.
    """
    if not code or not code.strip():
        return ""
    src = code.encode("utf-8", "replace")
    root = _parser().parse(src).root_node
    budget = [max_nodes]

    def walk(node) -> str:
        if budget[0] <= 0:
            return ""
        budget[0] -= 1
        if node.child_count == 0:
            if not node.is_named:
                return src[node.start_byte:node.end_byte].decode("utf-8", "replace")
            t = node.type
            if t in _IDENT_NODES:
                return "IDENT"
            if t in _TYPE_NODES:
                return "TYPE"
            if t in _LITERAL_NODES:
                return "LIT"
            return t
        parts = [node.type]
        for i in range(node.child_count):
            piece = walk(node.child(i))
            if piece:
                parts.append(piece)
            if budget[0] <= 0:
                break
        return "(" + " ".join(parts) + ")"

    return walk(root)


def sexpr_fields(code: str) -> str:
    """tree-sitter's own canonical S-expression: named nodes only, with field
    labels (`type:`, `declarator:`, `body:`). Stored as the `Root Node` column."""
    if not code or not code.strip():
        return ""
    src = code.encode("utf-8", "replace")
    return str(_parser().parse(src).root_node)


def parse_status(code: str) -> dict:
    """Parse health for one function; a spike in `error_nodes` across a config
    means a bad decompiler run rather than a quietly degraded input view."""
    if not code or not code.strip():
        return {"ok": False, "reason": "empty", "error_nodes": 0, "nodes": 0}
    root = _parser().parse(code.encode("utf-8", "replace")).root_node
    errors = nodes = 0
    stack = [root]
    while stack:
        n = stack.pop()
        nodes += 1
        if n.type == "ERROR" or n.is_missing:
            errors += 1
        for i in range(n.child_count):
            stack.append(n.child(i))
    return {
        "ok": not root.has_error,
        "reason": "" if not root.has_error else "parse_errors",
        "error_nodes": errors,
        "nodes": nodes,
    }


def add_sexpr_columns(example: dict, code_field: str = "decompiled_code_stripped",
                      prefix: str = "") -> dict:
    """`datasets.map` helper deriving all three views for one record."""
    code = example.get(code_field) or ""
    return {
        f"{prefix}S-Expression_of_decompiled_code_stripped": sexpr_with_text(code),
        f"{prefix}S-Expression_decompiled_code_stripped_clean": sexpr_clean(code),
        f"{prefix}Root Node": sexpr_fields(code),
    }


if __name__ == "__main__":
    import sys
    sample = sys.stdin.read() if not sys.stdin.isatty() else (
        "\nvoid FUN_00157910(undefined8 *param_1)\n\n{\n  *param_1 = &PTR_FUN_00426068;\n"
        "  if ((undefined8 *)param_1[1] != param_1 + 3) {\n    FUN_002dac70();\n  }\n"
        "  return;\n}\n"
    )
    print("--- with text ---");  print(sexpr_with_text(sample))
    print("\n--- clean ---");    print(sexpr_clean(sample))
    print("\n--- fields ---");   print(sexpr_fields(sample))
    print("\n--- status ---");   print(parse_status(sample))
