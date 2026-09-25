"""Tree-sitter S-expression views of decompiled C.

The AST view is the third of ReFuN's four encoder inputs. It is produced by
parsing the *stripped* decompiled function with the tree-sitter C grammar and
linearising the parse tree. Three linearisations are used in the paper; all
three are reproduced here so the published corpus columns can be regenerated
from `decompiled_code_stripped` alone.

    column in the corpus                          function here
    --------------------------------------------- --------------------
    S-Expression_of_decompiled_code_stripped      sexpr_with_text()
    S-Expression_decompiled_code_*_clean          sexpr_clean()
    Root Node                                     sexpr_fields()

`sexpr_with_text` is the one the model actually consumes (`--view_sexpr`).
It interleaves node types with the source span of each internal node, so the
encoder sees structure and surface form together.

`sexpr_clean` is the identifier-free variant: every identifier, type name and
literal collapses to `IDENT` / `TYPE` / `LIT`. It answers "does structure alone
help?" and is what an ablation that wants to rule out surface-token memorisation
should use.

Ghidra's decompiler output is not always valid C (it emits `undefined8`,
`code *`, `__thiscall`, and occasional `WARNING:` comments). tree-sitter is
error-tolerant and produces ERROR nodes rather than failing, which is the
behaviour we want -- a partially-parsed function is still usable signal. Parse
failure is reported via `parse_status`, never silently swallowed.

REPRODUCTION FIDELITY
---------------------
Regenerating the published x64_O0 train split with tree_sitter 0.26 /
tree_sitter_c 0.24 reproduces `S-Expression_of_decompiled_code_stripped`
byte-for-byte on 98.5% of records (n=200). The residual 1.5% are error-recovery
differences: on malformed decompiler output the grammar version used to build
the corpus inserted a MISSING token (typically `;`) where the current version
does not. This is a tree-sitter version difference, not a change in the
linearisation, and it only affects functions that already fail to parse
cleanly. `sexpr_fields` ("Root Node") is more version-sensitive -- 49% exact --
because tree-sitter's own canonical printer changed how it renders MISSING and
ERROR nodes; that column is used only for deduplication, never as model input.
Pin the grammar in `requirements.txt` if byte-identical regeneration matters.
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
    """Lazily build the C parser. Imported lazily so that importing `refun.train`
    does not hard-require tree-sitter for users who only train on the published
    corpus (where the S-expression column already exists)."""
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
    """Source span as a double-quoted token.

    The published corpus escapes *only* newlines: an embedded `"` or `\\` in the
    decompiled source is written through unescaped, which makes the quoted span
    ambiguous to a strict S-expression reader. That is reproduced faithfully by
    default (`strict=False`) so this function regenerates the stored columns
    byte-for-byte. Pass `strict=True` for a properly escaped, re-parseable span
    when building a *new* corpus. The model never re-parses the span, so the
    choice does not affect training -- only downstream tooling.
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
    """Parse health for one function. The corpus builder records this so that a
    config with a broken decompiler run is visible as a spike in `error_nodes`
    rather than as a quietly degraded input view."""
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
    """`datasets.map` helper: derive all three views for one record."""
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
