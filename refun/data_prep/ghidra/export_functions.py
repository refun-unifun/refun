
import json, os, math
from ghidra.app.decompiler import DecompInterface, DecompileOptions
from ghidra.service.graph import AttributedGraph
from ghidra.program.model.pcode import PcodeOp
from ghidra.util.task import ConsoleTaskMonitor

def getInstructionListing(func):
    instructions = []
    listing = currentProgram.getListing()
    instr_iter = listing.getInstructions(func.getBody(), True)
    while instr_iter.hasNext():
        instructions.append(str(instr_iter.next()))
    return "\n".join(instructions)

def get_pcode_edges(func):
    try:
        from ghidra.program.model.pcode import PcodeOp, Varnode, SequenceNumber
    except Exception as e:
        return "Error importing pcode modules: " + str(e)
    options = DecompileOptions()
    ifc = DecompInterface()
    if not ifc.openProgram(currentProgram):
        return "Unable to open program for decompilation"
    res = ifc.decompileFunction(func, 30, monitor)
    highFunc = res.getHighFunction()
    tempFunc = highFunc.getFunctionPrototype()
    numParams = tempFunc.getNumParams()
    parameterList = []
    for i in range(numParams):
        try:
            param = tempFunc.getParam(i)
            if param:
                it = param.getHighVariable().getInstances().iterator()
                while it.hasNext():
                    parameterList.append(it.next())
        except:
            continue
    allEdges = set()
    op_iter = highFunc.getPcodeOps()
    while op_iter.hasNext():
        onePcode = op_iter.next()
        opcode = onePcode.getOpcode()
        if opcode == PcodeOp.INDIRECT:
            try:
                addrOp = onePcode.getSeqnum().getTarget()
                inputs = onePcode.getInputs()
                if len(inputs) > 1:
                    offset = int(inputs[1].getOffset())
                    seqN = SequenceNumber(addrOp, offset)
                    actualPcode = highFunc.getPcodeOp(seqN)
                    allEdges.add((actualPcode, onePcode))
            except:
                pass
        elif opcode == PcodeOp.CALL:
            inputs = onePcode.getInputs()
            if len(inputs) > 1:
                for j in range(1, len(inputs)):
                    inp = inputs[j]
                    allEdges.add((inp.getDef() if inp.getDef() else inp, onePcode))
        else:
            for inp in onePcode.getInputs():
                allEdges.add((inp.getDef() if inp.getDef() else inp, onePcode))

    def gen_label(node):
        try:
            from ghidra.program.model.pcode import Varnode, PcodeOp
            if isinstance(node, Varnode):
                return "const-" + str(node.getOffset()) if node.isConstant() else "tmp_" + str(node.hashCode())
            else:
                return node.getMnemonic() + "_" + str(node.hashCode())
        except:
            return "error"

    edges_list = []
    for edge in allEdges:
        try:
            edges_list.append((gen_label(edge[0]), gen_label(edge[1])))
        except:
            continue
    return str(edges_list)

def export_functions():
    fm = currentProgram.getFunctionManager()
    functions = []
    ifc = DecompInterface()
    ifc.openProgram(currentProgram)
    it = fm.getFunctions(True)
    while it.hasNext():
        func = it.next()
        func_info = {
            'original_function_name': func.getName(),
            'stripped_function_name': func.getName(),
            'address': str(func.getEntryPoint()),
            'assembly': getInstructionListing(func)
        }
        res = ifc.decompileFunction(func, 60, monitor)
        if res and res.getDecompiledFunction():
            func_info['decompiled'] = res.getDecompiledFunction().getC()
        else:
            func_info['decompiled'] = ""

        # Dataflow graph & images are not generated, so set them to "N/A"
        func_info['dataflowgraph'] = "N/A"
        func_info['graphImage'] = "N/A"

        try:
            func_info['pcode_edges'] = get_pcode_edges(func)
        except:
            func_info['pcode_edges'] = "Error pcode"
        functions.append(func_info)
    outDir = os.getenv("GHIDRA_OUTPUT_DIR", os.getcwd())
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    with open(os.path.join(outDir, "functions.json"), "w") as f:
        json.dump(functions, f, indent=4)
    return functions

export_functions()
