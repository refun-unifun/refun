# -*- coding: utf-8 -*-
# export_all_features.py
# Ghidra 11.x headless script (Jython 2.7 compatible)
# Extracts comprehensive features from a single binary.
# Environment variables consumed:
#   GHIDRA_OUTPUT_DIR  - directory to write all_features.json
#   IS_STRIPPED        - "true" / "false" (default "false")

import json
import os
import sys
import traceback

from ghidra.app.decompiler import DecompInterface, DecompileOptions
from ghidra.program.model.pcode import PcodeOp, Varnode
from ghidra.program.model.listing import CodeUnit
from ghidra.program.model.symbol import SymbolType
from ghidra.program.model.block import BasicBlockModel
from ghidra.util.task import ConsoleTaskMonitor

monitor = ConsoleTaskMonitor()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def safe(fn, default="N/A"):
    try:
        return fn()
    except Exception as e:
        return "ERROR: " + str(e)


def java_iter_to_list(java_iterator):
    result = []
    while java_iterator.hasNext():
        result.append(java_iterator.next())
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PROGRAM-LEVEL EXTRACTORS
# ─────────────────────────────────────────────────────────────────────────────

def get_memory_map():
    try:
        blocks = []
        for block in currentProgram.getMemory().getBlocks():
            blocks.append({
                "name":       str(block.getName()),
                "start":      str(block.getStart()),
                "end":        str(block.getEnd()),
                "size":       block.getSize(),
                "readable":   bool(block.isRead()),
                "writable":   bool(block.isWrite()),
                "executable": bool(block.isExecute()),
                "initialized":bool(block.isInitialized()),
                "type":       str(block.getType()),
            })
        return blocks
    except Exception as e:
        return [{"error": str(e)}]


def get_global_variables():
    try:
        result = []
        sym_table = currentProgram.getSymbolTable()
        listing   = currentProgram.getListing()
        for sym in java_iter_to_list(sym_table.getAllSymbols(True)):
            if sym.getSymbolType() == SymbolType.LABEL and sym.isGlobal():
                dt_name = None
                try:
                    data = listing.getDataAt(sym.getAddress())
                    if data:
                        dt_name = str(data.getDataType().getName())
                except Exception:
                    pass
                result.append({
                    "name":      str(sym.getName()),
                    "address":   str(sym.getAddress()),
                    "type":      dt_name or "unknown",
                    "namespace": str(sym.getParentNamespace().getName()),
                })
        return result
    except Exception as e:
        return [{"error": str(e)}]


def get_strings_table():
    try:
        result = []
        data_iter = currentProgram.getListing().getDefinedData(True)
        while data_iter.hasNext():
            data = data_iter.next()
            if data.hasStringValue():
                try:
                    result.append({
                        "address":   str(data.getAddress()),
                        "value":     str(data.getValue()),
                        "length":    data.getLength(),
                        "data_type": str(data.getDataType().getName()),
                    })
                except Exception:
                    pass
        return result
    except Exception as e:
        return [{"error": str(e)}]


def get_import_export_table():
    try:
        imports = []
        exports = []

        # Imports via ExternalManager
        ext_mgr = currentProgram.getExternalManager()
        for lib_name in java_iter_to_list(ext_mgr.getExternalLibraryNames()):
            for ext_loc in java_iter_to_list(ext_mgr.getExternalLocations(lib_name)):
                addr = None
                try:
                    if ext_loc.getAddress():
                        addr = str(ext_loc.getAddress())
                except Exception:
                    pass
                imports.append({
                    "name":    str(ext_loc.getLabel()),
                    "library": str(lib_name),
                    "address": addr,
                })

        # Exports — symbols that are external entry points
        sym_table = currentProgram.getSymbolTable()
        ep_iter   = sym_table.getExternalEntryPointIterator()
        while ep_iter.hasNext():
            addr   = ep_iter.next()
            symbol = sym_table.getPrimarySymbol(addr)
            if symbol:
                exports.append({
                    "name":    str(symbol.getName()),
                    "address": str(addr),
                })

        return {"imports": imports, "exports": exports}
    except Exception as e:
        return {"error": str(e)}


def get_data_types_structures():
    try:
        from ghidra.program.model.data import Structure, Enum, Union
        result = []
        dtm = currentProgram.getDataTypeManager()
        for dt in java_iter_to_list(dtm.getAllDataTypes()):
            try:
                if not isinstance(dt, (Structure, Enum, Union)):
                    continue
                dt_info = {
                    "name":     str(dt.getName()),
                    "category": str(dt.getCategoryPath()),
                    "size":     max(0, dt.getLength()),
                    "type":     type(dt).__name__,
                    "fields":   [],
                }
                try:
                    for i in range(dt.getNumComponents()):
                        comp = dt.getComponent(i)
                        if comp:
                            dt_info["fields"].append({
                                "name":   str(comp.getFieldName() or ""),
                                "offset": comp.getOffset(),
                                "type":   str(comp.getDataType().getName()),
                                "size":   comp.getLength(),
                            })
                except Exception:
                    pass
                result.append(dt_info)
            except Exception:
                pass
        return result
    except Exception as e:
        return [{"error": str(e)}]


def get_program_comments():
    try:
        result = []
        listing = currentProgram.getListing()
        mem     = currentProgram.getMemory()
        cmt_iter = listing.getCommentAddressIterator(mem, True)
        TYPES = [
            (CodeUnit.PRE_COMMENT,        "pre"),
            (CodeUnit.POST_COMMENT,       "post"),
            (CodeUnit.EOL_COMMENT,        "eol"),
            (CodeUnit.PLATE_COMMENT,      "plate"),
            (CodeUnit.REPEATABLE_COMMENT, "repeatable"),
        ]
        while cmt_iter.hasNext():
            addr = cmt_iter.next()
            for ct, name in TYPES:
                cmt = listing.getComment(ct, addr)
                if cmt:
                    result.append({"address": str(addr), "type": name, "text": str(cmt)})
        return result
    except Exception as e:
        return [{"error": str(e)}]


def get_call_graph_program():
    try:
        nodes = []
        edges = []
        fm = currentProgram.getFunctionManager()
        it = fm.getFunctions(True)
        while it.hasNext():
            func = it.next()
            nodes.append({
                "name":    str(func.getName()),
                "address": str(func.getEntryPoint()),
            })
            for called in java_iter_to_list(func.getCalledFunctions(monitor)):
                edges.append({
                    "caller": str(func.getEntryPoint()),
                    "callee": str(called.getEntryPoint()),
                })
        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION-LEVEL EXTRACTORS
# ─────────────────────────────────────────────────────────────────────────────

def get_assembly(func):
    try:
        lines = []
        listing  = currentProgram.getListing()
        instr_it = listing.getInstructions(func.getBody(), True)
        while instr_it.hasNext():
            lines.append(str(instr_it.next()))
        return "\n".join(lines)
    except Exception as e:
        return "ERROR: " + str(e)


def get_instruction_metadata(func):
    try:
        result   = []
        listing  = currentProgram.getListing()
        instr_it = listing.getInstructions(func.getBody(), True)
        while instr_it.hasNext():
            instr = instr_it.next()
            try:
                raw_bytes = instr.getBytes()
                byte_str  = " ".join("{:02x}".format(b & 0xff) for b in raw_bytes)
                ft_addr   = instr.getFallThrough()
                result.append({
                    "address":    str(instr.getAddress()),
                    "mnemonic":   str(instr.getMnemonicString()),
                    "bytes":      byte_str,
                    "operands":   str(instr),
                    "fall_through": str(ft_addr) if ft_addr else None,
                    "flow_type":  str(instr.getFlowType()),
                    "length":     instr.getLength(),
                })
            except Exception:
                result.append({"address": str(instr.getAddress()), "error": "parse_failed"})
        return result
    except Exception as e:
        return [{"error": str(e)}]


def get_raw_pcode_listing(func):
    try:
        lines    = []
        listing  = currentProgram.getListing()
        instr_it = listing.getInstructions(func.getBody(), True)
        while instr_it.hasNext():
            instr = instr_it.next()
            ops   = instr.getPcode()
            if ops:
                for op in ops:
                    lines.append(str(op))
        return "\n".join(lines)
    except Exception as e:
        return "ERROR: " + str(e)


def get_cfg(func):
    try:
        model      = BasicBlockModel(currentProgram)
        cfg_nodes  = []
        cfg_edges  = []
        blocks_it  = model.getCodeBlocksContaining(func.getBody(), monitor)
        block_list = java_iter_to_list(blocks_it)
        for block in block_list:
            cfg_nodes.append({
                "id":    str(block.getFirstStartAddress()),
                "name":  str(block.getName()),
                "start": str(block.getFirstStartAddress()),
                "end":   str(block.getMaxAddress()),
            })
            dests = block.getDestinations(monitor)
            while dests.hasNext():
                dest_ref = dests.next()
                cfg_edges.append({
                    "from":      str(block.getFirstStartAddress()),
                    "to":        str(dest_ref.getDestinationAddress()),
                    "flow_type": str(dest_ref.getFlowType()),
                })
        return {"nodes": cfg_nodes, "edges": cfg_edges}
    except Exception as e:
        return {"error": str(e)}


def get_dataflow_graph(high_func):
    try:
        edges    = []
        op_iter  = high_func.getPcodeOps()
        while op_iter.hasNext():
            op     = op_iter.next()
            output = op.getOutput()
            if output:
                use_iter = output.getDescendants()
                while use_iter.hasNext():
                    use_op = use_iter.next()
                    edges.append({
                        "def_op":   op.getMnemonic(),
                        "def_addr": str(op.getSeqnum().getTarget()),
                        "use_op":   use_op.getMnemonic(),
                        "use_addr": str(use_op.getSeqnum().getTarget()),
                        "varnode":  str(output),
                    })
        return edges
    except Exception as e:
        return [{"error": str(e)}]


def get_pcode_edges(high_func):
    try:
        edges   = []
        op_iter = high_func.getPcodeOps()
        while op_iter.hasNext():
            op = op_iter.next()
            for inp in op.getInputs():
                def_op = inp.getDef()
                if def_op:
                    edges.append([
                        def_op.getMnemonic() + "_" + str(def_op.hashCode()),
                        op.getMnemonic()     + "_" + str(op.hashCode()),
                    ])
        return edges
    except Exception as e:
        return [["error", str(e)]]


def get_pcode_graph(high_func):
    try:
        nodes   = {}
        edges   = []
        op_iter = high_func.getPcodeOps()
        while op_iter.hasNext():
            op      = op_iter.next()
            op_id   = str(op.getSeqnum())
            output  = op.getOutput()
            inputs  = [str(inp) for inp in op.getInputs()]
            nodes[op_id] = {
                "id":       op_id,
                "mnemonic": op.getMnemonic(),
                "output":   str(output) if output else None,
                "inputs":   inputs,
                "address":  str(op.getSeqnum().getTarget()),
            }
            if output:
                use_iter = output.getDescendants()
                while use_iter.hasNext():
                    use_op = use_iter.next()
                    edges.append({
                        "from": op_id,
                        "to":   str(use_op.getSeqnum()),
                        "type": "data",
                    })
        return {"nodes": list(nodes.values()), "edges": edges}
    except Exception as e:
        return {"error": str(e)}


def get_graph_columns(func):
    """Return per-basic-block pcode (column-per-block representation)."""
    try:
        columns  = []
        model    = BasicBlockModel(currentProgram)
        listing  = currentProgram.getListing()
        blocks_it = model.getCodeBlocksContaining(func.getBody(), monitor)
        while blocks_it.hasNext():
            block      = blocks_it.next()
            pcode_list = []
            instr_it   = listing.getInstructions(block, True)
            while instr_it.hasNext():
                instr = instr_it.next()
                ops   = instr.getPcode()
                if ops:
                    for op in ops:
                        pcode_list.append(str(op))
            columns.append({
                "block_start": str(block.getFirstStartAddress()),
                "block_end":   str(block.getMaxAddress()),
                "pcode":       pcode_list,
            })
        return columns
    except Exception as e:
        return [{"error": str(e)}]


def get_xrefs(func):
    try:
        ref_mgr   = currentProgram.getReferenceManager()
        entry     = func.getEntryPoint()
        body      = func.getBody()

        xrefs_to  = []
        refs_to   = ref_mgr.getReferencesTo(entry)
        while refs_to.hasNext():
            ref = refs_to.next()
            xrefs_to.append({
                "from": str(ref.getFromAddress()),
                "type": str(ref.getReferenceType()),
            })

        xrefs_from = []
        for addr_range in body:
            refs_from = ref_mgr.getReferenceIterator(addr_range.getMinAddress())
            while refs_from.hasNext():
                ref = refs_from.next()
                if body.contains(ref.getFromAddress()):
                    xrefs_from.append({
                        "to":   str(ref.getToAddress()),
                        "type": str(ref.getReferenceType()),
                    })

        return {"xrefs_to": xrefs_to, "xrefs_from": xrefs_from}
    except Exception as e:
        return {"error": str(e)}


def get_function_comments(func):
    try:
        listing   = currentProgram.getListing()
        INLINE_CT = [
            (CodeUnit.PRE_COMMENT,  "pre"),
            (CodeUnit.POST_COMMENT, "post"),
            (CodeUnit.EOL_COMMENT,  "eol"),
        ]
        inline = []
        cmt_it = listing.getCommentAddressIterator(func.getBody(), True)
        while cmt_it.hasNext():
            addr = cmt_it.next()
            for ct, name in INLINE_CT:
                cmt = listing.getComment(ct, addr)
                if cmt:
                    inline.append({"address": str(addr), "type": name, "text": str(cmt)})
        return {
            "plate_comment":      str(func.getComment())      if func.getComment()      else None,
            "repeatable_comment": str(func.getRepeatableComment()) if func.getRepeatableComment() else None,
            "inline_comments":    inline,
        }
    except Exception as e:
        return {"error": str(e)}


def get_decompiled_assembly(func, high_func):
    """Assembly listing annotated with high-level variable names from decompiler."""
    try:
        lines    = []
        listing  = currentProgram.getListing()
        instr_it = listing.getInstructions(func.getBody(), True)
        while instr_it.hasNext():
            instr = instr_it.next()
            line  = str(instr)
            # Annotate output varnode with high-level variable name
            try:
                for op in instr.getPcode():
                    out = op.getOutput()
                    if out:
                        hv = high_func.getHighVariable(out)
                        if hv and hv.getName():
                            line += "  ; " + str(hv.getName())
                            break
            except Exception:
                pass
            lines.append(line)
        return "\n".join(lines)
    except Exception as e:
        return "ERROR: " + str(e)


def clang_to_sexpression(token, depth=0):
    """Recursively convert a Ghidra ClangToken tree to S-expression string."""
    if depth > 60:
        return '"..."'
    try:
        num_ch = token.numChildren()
        if num_ch == 0:
            txt = str(token).replace("\\", "\\\\").replace('"', '\\"')
            return '"{}"'.format(txt)
        cls_name = type(token).__name__
        children = [clang_to_sexpression(token.Child(i), depth + 1) for i in range(num_ch)]
        return "({} {})".format(cls_name, " ".join(children))
    except Exception as e:
        return '(error "{}")'.format(str(e)[:80].replace('"', '\\"'))


# ─────────────────────────────────────────────────────────────────────────────
# PER-FUNCTION PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

def process_function(func, ifc, is_stripped):
    entry_hex  = "FUN_{:08x}".format(func.getEntryPoint().getOffset())
    func_name  = str(func.getName())
    masked_name = entry_hex if is_stripped else func_name

    info = {
        # Identity
        "original_function_name": func_name,
        "stripped_function_name": entry_hex,
        "address":               str(func.getEntryPoint()),
        "function_signature":    str(func.getSignature()),
        "calling_convention":    str(func.getCallingConventionName()),
        "return_type":           str(func.getReturnType()),
        "parameters":            [],
        "is_thunk":              bool(func.isThunk()),
        "is_external":           bool(func.isExternal()),
        "size":                  func.getBody().getNumAddresses(),
        "local_var_count":       0,

        # Code representations
        "original_code":           "N/A",   # decompiled with real names
        "decompiled_code":         "N/A",
        "assembly":                "N/A",
        "decompiled_assembly":     "N/A",
        "raw_pcode_listing":       "N/A",
        "renamed_masked_code":     "N/A",

        # Graphs
        "control_flow_graph":  {},
        "dataflow_graph":      [],
        "pcode_edges":         [],
        "pcode_graph":         {},
        "graph_columns":       [],

        # Context
        "xrefs":                  {},
        "comments_annotations":   {},
        "instruction_metadata":   [],
        "decompiler_warnings":    "",
        "call_graph_local":       {"calls": [], "called_by": []},

        # S-expressions
        "s_expression_original": "N/A",
        "s_expression_stripped": "N/A",
    }

    # ── Parameters ────────────────────────────────────────────────────────────
    try:
        info["parameters"] = [
            {"name": str(p.getName()), "type": str(p.getDataType())}
            for p in func.getParameters()
        ]
    except Exception:
        pass

    # ── Local variable count ──────────────────────────────────────────────────
    try:
        info["local_var_count"] = len(list(func.getLocalVariables()))
    except Exception:
        pass

    # ── Assembly ──────────────────────────────────────────────────────────────
    info["assembly"]          = get_assembly(func)
    info["instruction_metadata"] = get_instruction_metadata(func)
    info["raw_pcode_listing"] = get_raw_pcode_listing(func)

    # ── CFG ───────────────────────────────────────────────────────────────────
    info["control_flow_graph"] = get_cfg(func)

    # ── Xrefs ─────────────────────────────────────────────────────────────────
    info["xrefs"] = get_xrefs(func)

    # ── Comments ──────────────────────────────────────────────────────────────
    info["comments_annotations"] = get_function_comments(func)

    # ── Graph columns (raw pcode per BB) ──────────────────────────────────────
    info["graph_columns"] = get_graph_columns(func)

    # ── Local call graph ──────────────────────────────────────────────────────
    try:
        info["call_graph_local"] = {
            "calls":     [{"name": str(f.getName()), "address": str(f.getEntryPoint())}
                          for f in java_iter_to_list(func.getCalledFunctions(monitor))],
            "called_by": [{"name": str(f.getName()), "address": str(f.getEntryPoint())}
                          for f in java_iter_to_list(func.getCallingFunctions(monitor))],
        }
    except Exception:
        pass

    # ── Decompilation ─────────────────────────────────────────────────────────
    try:
        res = ifc.decompileFunction(func, 60, monitor)
    except Exception as e:
        info["decompiler_warnings"] = "DECOMPILE_EXCEPTION: " + str(e)
        return info

    if not res:
        info["decompiler_warnings"] = "NO_RESULT"
        return info

    # Warnings / errors from decompiler
    try:
        err_msg = res.getErrorMessage()
        info["decompiler_warnings"] = str(err_msg) if err_msg else ""
    except Exception:
        pass

    dec_func = res.getDecompiledFunction()
    if dec_func:
        c_code = str(dec_func.getC())
        info["decompiled_code"] = c_code
        info["original_code"]   = c_code                     # original names preserved
        # Mask names for stripped version
        masked = c_code.replace(func_name, masked_name)
        info["renamed_masked_code"] = masked

    high_func = res.getHighFunction()
    if high_func:
        info["decompiled_assembly"] = get_decompiled_assembly(func, high_func)
        info["dataflow_graph"]      = get_dataflow_graph(high_func)
        info["pcode_edges"]         = get_pcode_edges(high_func)
        info["pcode_graph"]         = get_pcode_graph(high_func)

    # ── S-expressions from ClangAST ───────────────────────────────────────────
    try:
        markup = res.getCCodeMarkup()
        if markup:
            sexp_orig = clang_to_sexpression(markup)
            sexp_strip = sexp_orig.replace(
                '"{}"'.format(func_name),
                '"{}"'.format(masked_name),
            )
            info["s_expression_original"] = sexp_orig
            info["s_expression_stripped"] = sexp_strip
    except Exception as e:
        info["s_expression_original"] = "SEXP_ERROR: " + str(e)

    return info


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXPORT ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def export_all():
    out_dir     = os.getenv("GHIDRA_OUTPUT_DIR", os.getcwd())
    is_stripped = os.getenv("IS_STRIPPED", "false").lower() == "true"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    prog = currentProgram
    lang = prog.getLanguage()
    cs   = prog.getCompilerSpec()

    # ── Program-level info ────────────────────────────────────────────────────
    program_info = {
        "name":             str(prog.getName()),
        "language_id":      str(lang.getLanguageID()),
        "compiler_spec":    str(cs.getCompilerSpecID()),
        "image_base":       str(prog.getImageBase()),
        "address_size":     lang.getAddressFactory().getDefaultAddressSpace().getSize(),
        "endian":           "big" if lang.isBigEndian() else "little",
        "executable_format":str(prog.getExecutableFormat()),
        "md5":              str(prog.getExecutableMD5()),
        "creation_date":    str(prog.getCreationDate()),
        "memory_map":              get_memory_map(),
        "global_variables":        get_global_variables(),
        "strings_table":           get_strings_table(),
        "import_export_table":     get_import_export_table(),
        "data_types_structures":   get_data_types_structures(),
        "comments":                get_program_comments(),
        "call_graph":              get_call_graph_program(),
    }

    # ── Relocations / patched bytes ───────────────────────────────────────────
    try:
        relocs = []
        rel_tbl = prog.getRelocationTable()
        rel_it  = rel_tbl.getRelocations()
        while rel_it.hasNext():
            r = rel_it.next()
            relocs.append({
                "address": str(r.getAddress()),
                "type":    r.getType(),
                "values":  list(r.getValues()) if r.getValues() else [],
            })
        program_info["patched_bytes"] = relocs
    except Exception as e:
        program_info["patched_bytes"] = [{"error": str(e)}]

    # ── Function-level extraction ─────────────────────────────────────────────
    options = DecompileOptions()
    ifc     = DecompInterface()
    ifc.setOptions(options)
    ifc.openProgram(prog)

    functions = []
    fm = prog.getFunctionManager()
    it = fm.getFunctions(True)
    while it.hasNext():
        func = it.next()
        try:
            functions.append(process_function(func, ifc, is_stripped))
        except Exception as e:
            functions.append({
                "original_function_name": str(func.getName()),
                "address":                str(func.getEntryPoint()),
                "error":                  str(e),
                "traceback":              traceback.format_exc(),
            })

    ifc.dispose()

    result = {
        "program_info": program_info,
        "functions":    functions,
    }

    out_path = os.path.join(out_dir, "all_features.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("Exported {} functions to {}".format(len(functions), out_path))


export_all()
