"""
Intermediate Representation (IR) Generator - COMPLETE ENHANCED VERSION
Full support for semantic analyzer features:
- ✅ const/static/volatile qualifiers
- ✅ References
- ✅ Inheritance
- ✅ Access control
- ✅ Preprocessor metadata
"""

from dataclasses import dataclass
from typing import List, Optional, Any, Dict
from enum import Enum, auto
import sys
sys.path.append('..')
from parser.parser import ASTNode, ASTNodeType

# ----------------------------------------------------------------------
#  IR Operation Codes (ENHANCED with new opcodes)
# ----------------------------------------------------------------------
class IROpCode(Enum):
    """IR Operation Codes"""
    
    # --- Data & Variables ---
    DECLARE = auto()        # DECLARE var_name, type
    CONST_DECLARE = auto()  # CONST_DECLARE var_name, type (immutable)
    STATIC_DECLARE = auto() # STATIC_DECLARE var_name, type (class-level)
    VOLATILE_DECLARE = auto() # VOLATILE_DECLARE var_name, type (hardware/optimization)
    REF_DECLARE = auto()    # REF_DECLARE ref_name, target_var (reference binding)
    ASSIGN = auto()         # ASSIGN target, value
    LOAD = auto()           # LOAD result, var_name
    LOAD_CONST = auto()     # LOAD_CONST result, value 
    
    # --- Arithmetic & Logic ---
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    NEG = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    
    # --- Comparisons ---
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    
    # --- Control Flow ---
    LABEL = auto()
    JUMP = auto()
    JUMP_IF_FALSE = auto()
    BREAK = auto()
    CONTINUE = auto()
    TERNARY = auto()

    # --- Functions ---
    FUNC_BEGIN = auto()
    FUNC_END = auto()
    PARAM = auto()
    ARG = auto()
    CALL = auto()
    RETURN = auto()
    
    # --- Arrays ---
    ARRAY_DECL = auto()
    ARRAY_LOAD = auto()
    ARRAY_STORE = auto()
    
    # --- Pointers ---
    LOAD_ADDRESS = auto()
    DEREF_LOAD = auto()
    DEREF_STORE = auto()
    
    # --- Classes & Inheritance (ENHANCED) ---
    CLASS_BEGIN = auto()         # CLASS_BEGIN name
    CLASS_END = auto()           # CLASS_END name
    CLASS_INHERIT = auto()       # CLASS_INHERIT derived, base
    MEMBER_DECLARE = auto()      # MEMBER_DECLARE name, type
    MEMBER_ACCESS_LEVEL = auto() # MEMBER_ACCESS_LEVEL member, level
    GET_MEMBER = auto()          # GET_MEMBER result, obj, member
    SET_MEMBER = auto()          # SET_MEMBER obj, member, value
    GET_MEMBER_PTR = auto()      # GET_MEMBER_PTR result, ptr, member
    SET_MEMBER_PTR = auto()      # SET_MEMBER_PTR ptr, member, value
    LOAD_THIS = auto()           # LOAD_THIS result
    
    # --- Preprocessor Metadata ---
    INCLUDE = auto()             # INCLUDE filename
    DEFINE = auto()              # DEFINE macro_name, value
    
    # --- I/O & Util ---
    PRINT = auto()
    INPUT = auto()
    COMMENT = auto()
    NOP = auto()

# ----------------------------------------------------------------------
#  IR Instruction
# ----------------------------------------------------------------------
@dataclass
class IRInstruction:
    """Single IR instruction"""
    opcode: IROpCode
    result: Optional[str] = None
    arg1: Optional[Any] = None
    arg2: Optional[Any] = None
    arg3: Optional[Any] = None
    comment: str = ""
    
    def to_string(self) -> str:
        """Convert to readable string"""
        if self.opcode == IROpCode.COMMENT:
            return f"# {self.arg1}"
        
        if self.opcode == IROpCode.LABEL:
            return f"{self.arg1}:"
        
        parts = [f"    {self.opcode.name:<20}"]
        
        if self.result:
            parts.append(f"{self.result}")
        
        if self.arg1 is not None:
            if self.result:
                parts.append("=")
            parts.append(f"{self.arg1}")
        
        if self.arg2 is not None:
            parts.append(f"{self.arg2}")
        
        if self.arg3 is not None:
            parts.append(f"{self.arg3}")
        
        result = " ".join(str(p) for p in parts)
        
        if self.comment:
            result += f"    # {self.comment}"
        
        return result

# ----------------------------------------------------------------------
#  IR Generator (COMPLETE ENHANCED)
# ----------------------------------------------------------------------
class IRGenerator:
    """Generates IR from a semantically-annotated AST with symbol table"""
    
    def __init__(self, ast: ASTNode, symbol_table=None, includes=None, defines=None):
        self.ast = ast
        self.symbol_table = symbol_table  # Symbol table from semantic analyzer
        self.includes = includes or set()
        self.defines = defines or {}
        self.instructions: List[IRInstruction] = []
        self.temp_counter = 0
        self.label_counter = 0
        self.current_function = None
        self.current_class = None
        self.loop_stack: List[Dict[str, str]] = []
        self.switch_stack: List[str] = []
    
    # --- Counter Helpers ---
    def new_temp(self) -> str:
        self.temp_counter += 1
        return f"t{self.temp_counter}"
    
    def new_label(self, prefix="L") -> str:
        self.label_counter += 1
        return f"{prefix}{self.label_counter}"
    
    def emit(self, opcode: IROpCode, result=None, arg1=None, arg2=None, arg3=None, comment=""):
        instruction = IRInstruction(opcode, result, arg1, arg2, arg3, comment)
        self.instructions.append(instruction)
        return instruction

    # --- Loop/Switch Context ---
    def enter_loop(self, start_label: str, end_label: str):
        self.loop_stack.append({'start': start_label, 'end': end_label})

    def exit_loop(self):
        if self.loop_stack:
            self.loop_stack.pop()

    def get_loop_start(self) -> Optional[str]:
        return self.loop_stack[-1]['start'] if self.loop_stack else None

    def get_loop_end(self) -> Optional[str]:
        return self.loop_stack[-1]['end'] if self.loop_stack else None
        
    def enter_switch(self, end_label: str):
        self.switch_stack.append(end_label)
        
    def exit_switch(self):
        if self.switch_stack:
            self.switch_stack.pop()
            
    def get_switch_end(self) -> Optional[str]:
        return self.switch_stack[-1] if self.switch_stack else None

    # --- Main Generation ---
    def generate(self) -> List[IRInstruction]:
        """Generate IR from AST with preprocessor metadata"""
        self.instructions = []
        self.emit(IROpCode.COMMENT, arg1="=" * 60)
        self.emit(IROpCode.COMMENT, arg1="Generated IR Code (Enhanced)")
        self.emit(IROpCode.COMMENT, arg1="=" * 60)
        
        # ✅ FIXED: Emit preprocessor metadata
        if self.includes:
            self.emit(IROpCode.COMMENT, arg1="--- Preprocessor Includes ---")
            for include in sorted(self.includes):
                self.emit(IROpCode.INCLUDE, arg1=include, comment=f"#include {include}")
        
        if self.defines:
            self.emit(IROpCode.COMMENT, arg1="--- Preprocessor Defines ---")
            for name, value in self.defines.items():
                self.emit(IROpCode.DEFINE, arg1=name, arg2=value, comment=f"#define {name} {value}")
        
        self.emit(IROpCode.COMMENT, arg1="--- Program Code ---")
        
        if self.ast:
            self.generate_node(self.ast)
        
        return self.instructions

    # --- L-Value Store Helper ---
    def generate_store(self, lvalue_node: ASTNode, value_temp: str):
        """Generate correct STORE based on l-value type"""
        if lvalue_node.node_type == ASTNodeType.IDENTIFIER:
            # Check if it's a reference - references update the bound variable
            symbol = self.symbol_table.lookup(lvalue_node.value) if self.symbol_table else None
            if symbol and symbol.is_reference:
                self.emit(IROpCode.ASSIGN, result=lvalue_node.value, arg1=value_temp,
                         comment=f"ref {lvalue_node.value} = {value_temp}")
            else:
                self.emit(IROpCode.ASSIGN, result=lvalue_node.value, arg1=value_temp)
            
        elif lvalue_node.node_type == ASTNodeType.ARRAY_ACCESS:
            array_temp = self.generate_node(lvalue_node.children[0])
            index_temp = self.generate_node(lvalue_node.children[1])
            self.emit(IROpCode.ARRAY_STORE, arg1=array_temp, arg2=index_temp, arg3=value_temp)
            
        elif lvalue_node.node_type == ASTNodeType.DEREFERENCE:
            ptr_temp = self.generate_node(lvalue_node.children[0])
            self.emit(IROpCode.DEREF_STORE, arg1=ptr_temp, arg2=value_temp)
                      
        elif lvalue_node.node_type == ASTNodeType.MEMBER_ACCESS:
            obj_temp = self.generate_node(lvalue_node.children[0])
            member_name = lvalue_node.value
            self.emit(IROpCode.SET_MEMBER, arg1=obj_temp, arg2=member_name, arg3=value_temp)
                      
        elif lvalue_node.node_type == ASTNodeType.MEMBER_ACCESS_POINTER:
            ptr_temp = self.generate_node(lvalue_node.children[0])
            member_name = lvalue_node.value
            self.emit(IROpCode.SET_MEMBER_PTR, arg1=ptr_temp, arg2=member_name, arg3=value_temp)
        else:
            self.emit(IROpCode.COMMENT, arg1=f"ERROR: Cannot store to {lvalue_node.node_type.name}")

    # --- Node Dispatcher ---
    def generate_node(self, node: ASTNode) -> Optional[str]:
        """Generate IR for a node"""
        if not node:
            return None
        
        # Add semantic type info as comment
        if hasattr(node, 'data_type') and node.data_type:
            type_info = f"Type: {node.data_type}"
            if hasattr(node, 'codegen'):
                if node.codegen.get('is_lvalue'):
                    type_info += " (lvalue)"
                if node.codegen.get('is_const'):
                    type_info += " (const)"
            self.emit(IROpCode.COMMENT, arg1=f"--- {node.node_type.name} --- {type_info}")

        nt = node.node_type
        
        # Program / Block
        if nt == ASTNodeType.PROGRAM:
            return self.generate_program(node)
        if nt == ASTNodeType.BLOCK:
            return self.generate_block(node)
            
        # Class Definition
        if nt == ASTNodeType.CLASS_DEFINITION:
            return self.generate_class_definition(node)

        # Declarations
        if nt == ASTNodeType.FUNCTION:
            return self.generate_function(node)
        if nt == ASTNodeType.VARIABLE_DECLARATION:
            return self.generate_variable_declaration(node)
        if nt == ASTNodeType.ARRAY_DECLARATION:
            return self.generate_array_declaration(node)
        if nt == ASTNodeType.MULTIPLE_DECLARATION:
            for child in node.children:
                self.generate_node(child)
            return None

        # Statements
        if nt == ASTNodeType.EXPRESSION_STATEMENT:
            if node.children:
                return self.generate_node(node.children[0])
            return None
        if nt == ASTNodeType.ASSIGNMENT:
            return self.generate_assignment(node)
        if nt == ASTNodeType.COMPOUND_ASSIGNMENT:
            return self.generate_compound_assignment(node)
        if nt in {ASTNodeType.INCREMENT, ASTNodeType.DECREMENT}:
            return self.generate_increment_decrement(node)
        if nt == ASTNodeType.RETURN_STATEMENT:
            return self.generate_return_statement(node)

        # Control Flow
        if nt == ASTNodeType.IF_STATEMENT:
            return self.generate_if_statement(node)
        if nt == ASTNodeType.WHILE_LOOP:
            return self.generate_while_loop(node)
        if nt == ASTNodeType.DO_WHILE_LOOP:
            return self.generate_do_while_loop(node)
        if nt == ASTNodeType.FOR_LOOP:
            return self.generate_for_loop(node)
        if nt == ASTNodeType.SWITCH_STATEMENT:
            return self.generate_switch_statement(node)
        if nt == ASTNodeType.CASE_STATEMENT:
            return self.generate_case_statement(node)
        if nt == ASTNodeType.DEFAULT_STATEMENT:
            return self.generate_default_statement(node)
        if nt == ASTNodeType.BREAK_STATEMENT:
            return self.generate_break(node)
        if nt == ASTNodeType.CONTINUE_STATEMENT:
            return self.generate_continue(node)

        # Expressions
        if nt == ASTNodeType.BINARY_OP:
            return self.generate_binary_op(node)
        if nt == ASTNodeType.UNARY_OP:
            return self.generate_unary_op(node)
        if nt == ASTNodeType.TERNARY_OP:
            return self.generate_ternary_op(node)
        if nt == ASTNodeType.FUNCTION_CALL:
            return self.generate_function_call(node)

        # Atomics
        if nt == ASTNodeType.LITERAL:
            return self.generate_literal(node)
        if nt == ASTNodeType.IDENTIFIER:
            return self.generate_identifier_load(node)
        if nt == ASTNodeType.ARRAY_ACCESS:
            return self.generate_array_load(node)
            
        # Pointer & Member
        if nt == ASTNodeType.ADDRESS_OF:
            return self.generate_address_of(node)
        if nt == ASTNodeType.DEREFERENCE:
            return self.generate_dereference_load(node)
        if nt == ASTNodeType.MEMBER_ACCESS:
            return self.generate_member_load(node)
        if nt == ASTNodeType.MEMBER_ACCESS_POINTER:
            return self.generate_member_pointer_load(node)
        if nt == ASTNodeType.THIS_KEYWORD:
            return self.generate_this(node)
            
        # Ignored
        if nt in {ASTNodeType.PREPROCESSOR, ASTNodeType.MEMBER_FUNCTION, ASTNodeType.MEMBER_VARIABLE}:
            return None
        
        self.emit(IROpCode.COMMENT, arg1=f"WARN: No IR for {nt.name}")
        return None
    
    # --- Generator Methods ---

    def generate_program(self, node: ASTNode) -> str:
        for child in node.children:
            if child.node_type != ASTNodeType.PREPROCESSOR:
                self.generate_node(child)
        return None
    
    def generate_block(self, node: ASTNode) -> str:
        for child in node.children:
            self.generate_node(child)
        return None
        
    def generate_class_definition(self, node: ASTNode) -> str:
        """✅ FIXED: Generate class with inheritance, access control, and member functions"""
        class_name = node.value.get('name') if isinstance(node.value, dict) else node.value
        base_classes = node.value.get('base_classes', []) if isinstance(node.value, dict) else []
        
        self.emit(IROpCode.COMMENT, arg1="")
        self.emit(IROpCode.CLASS_BEGIN, arg1=class_name)
        
        # ✅ FIXED: Emit inheritance information
        for base in base_classes:
            base_name = base.get('name', base) if isinstance(base, dict) else base
            access = base.get('access', 'public') if isinstance(base, dict) else 'public'
            self.emit(IROpCode.CLASS_INHERIT, arg1=class_name, arg2=base_name,
                    comment=f"{class_name} : {access} {base_name}")
        
        # Look up class symbol for member info
        class_symbol = self.symbol_table.lookup(class_name) if self.symbol_table else None
        self.current_class = class_symbol
        
        for member in node.children:
            if member.node_type == ASTNodeType.MEMBER_VARIABLE:
                decl_node = member.children[0] if member.children else member
                member_name = decl_node.value if hasattr(decl_node, 'value') else 'unknown'
                if isinstance(member_name, dict):
                    member_name = member_name.get('name', 'unknown')
                member_type = decl_node.data_type if hasattr(decl_node, 'data_type') else 'int'
                
                # ✅ FIXED: Look up member symbol for qualifiers and access
                member_symbol = None
                if class_symbol and hasattr(class_symbol, 'members') and member_name in class_symbol.members:
                    member_symbol = class_symbol.members[member_name]
                
                # Emit appropriate declaration based on qualifiers
                if member_symbol:
                    # Handle qualifiers
                    if member_symbol.is_reference:
                        self.emit(IROpCode.REF_DECLARE, arg1=member_name, arg2=member_type,
                                comment="reference member")
                    elif member_symbol.is_const:
                        if member_symbol.is_static:
                            self.emit(IROpCode.STATIC_DECLARE, arg1=member_name, arg2=member_type,
                                    comment="static const member")
                        else:
                            self.emit(IROpCode.CONST_DECLARE, arg1=member_name, arg2=member_type,
                                    comment="const member")
                    elif member_symbol.is_static:
                        self.emit(IROpCode.STATIC_DECLARE, arg1=member_name, arg2=member_type,
                                comment="static member")
                    elif member_symbol.is_volatile:
                        self.emit(IROpCode.VOLATILE_DECLARE, arg1=member_name, arg2=member_type,
                                comment="volatile member")
                    else:
                        self.emit(IROpCode.MEMBER_DECLARE, arg1=member_name, arg2=member_type)
                    
                    # ✅ FIXED: Emit access level
                    if hasattr(member_symbol, 'access_level'):
                        access_level = member_symbol.access_level.name if hasattr(member_symbol.access_level, 'name') else str(member_symbol.access_level)
                        self.emit(IROpCode.MEMBER_ACCESS_LEVEL, arg1=member_name, arg2=access_level,
                                comment=f"{access_level.lower()} access")
                else:
                    self.emit(IROpCode.MEMBER_DECLARE, arg1=member_name, arg2=member_type)
                    
            elif member.node_type == ASTNodeType.MEMBER_FUNCTION:
                # ✅ CRITICAL FIX: Handle member functions properly
                # Check if child is already a FUNCTION node or just a BLOCK
                child = member.children[0] if member.children else None
                
                if child and child.node_type == ASTNodeType.FUNCTION:
                    # Already wrapped in FUNCTION node - generate normally
                    self.generate_node(child)
                elif child and child.node_type == ASTNodeType.BLOCK:
                    # BLOCK only - need to wrap it in FUNCTION manually
                    func_info = member.value
                    func_name = func_info.get('name') if isinstance(func_info, dict) else func_info
                    return_type = func_info.get('return_type', 'void') if isinstance(func_info, dict) else 'void'
                    params = func_info.get('params', []) if isinstance(func_info, dict) else []
                    
                    # Manually generate FUNCTION wrapper
                    self.current_function = func_name
                    
                    self.emit(IROpCode.COMMENT, arg1="")
                    self.emit(IROpCode.FUNC_BEGIN, arg1=func_name, comment=f"Member function: {func_name}")
                    
                    # Generate parameters
                    for param in params:
                        param_type = param['type']
                        param_name = param['name']
                        
                        if '&' in param_type and '*' not in param_type:
                            self.emit(IROpCode.REF_DECLARE, arg1=param_name, arg2=param_type,
                                    comment="reference parameter")
                        else:
                            self.emit(IROpCode.PARAM, arg1=param_name, arg2=param_type)
                    
                    # Generate function body (the BLOCK)
                    self.generate_node(child)
                    
                    self.emit(IROpCode.FUNC_END, arg1=func_name)
                    self.current_function = None
                else:
                    # Fallback - try to generate whatever is there
                    if child:
                        self.generate_node(child)
        
        self.emit(IROpCode.CLASS_END, arg1=class_name)
        self.current_class = None
        return None

    def generate_function(self, node: ASTNode) -> str:
        """✅ ENHANCED: Generate function with reference parameter support"""
        func_info = node.value
        name = func_info['name'] if isinstance(func_info, dict) else func_info
        params = func_info.get('params', []) if isinstance(func_info, dict) else []
        
        self.current_function = name
        
        self.emit(IROpCode.COMMENT, arg1="")
        self.emit(IROpCode.FUNC_BEGIN, arg1=name, comment=f"Function: {name}")
        
        for param in params:
            param_type = param['type']
            param_name = param['name']
            
            # ✅ FIXED: Check if parameter is a reference
            if '&' in param_type and '*' not in param_type:
                self.emit(IROpCode.REF_DECLARE, arg1=param_name, arg2=param_type,
                         comment="reference parameter")
            else:
                self.emit(IROpCode.PARAM, arg1=param_name, arg2=param_type)
        
        for child in node.children:
            self.generate_node(child)
        
        self.emit(IROpCode.FUNC_END, arg1=name)
        self.current_function = None
        return None
    
    def generate_variable_declaration(self, node: ASTNode) -> str:
        """✅ FIXED: Generate variable declaration with full qualifier support"""
        var_name = node.value['name'] if isinstance(node.value, dict) else node.value
        var_type = node.data_type or 'int'
        
        # ✅ ENHANCED DEBUG
        print(f"\n{'='*70}")
        print(f"IR: Processing variable '{var_name}'")
        print(f"  AST data_type: '{var_type}'")
        print(f"  self.symbol_table exists: {self.symbol_table is not None}")
        
        if self.symbol_table:
            print(f"  Symbol table details:")
            print(f"    Total scopes: {len(self.symbol_table.scopes)}")
            print(f"    Current scope: {self.symbol_table.current_scope}")
            
            # Show all symbols in all scopes
            for scope_idx, scope_dict in enumerate(self.symbol_table.scopes):
                if scope_dict:
                    print(f"    Scope {scope_idx}: {list(scope_dict.keys())}")
            
            # Show all variables in all_symbols
            print(f"  All symbols ({len(self.symbol_table.all_symbols)} total):")
            for sym in self.symbol_table.all_symbols:
                if sym.symbol_type.name == 'VARIABLE':
                    print(f"    - {sym.name}: type='{sym.data_type}', scope={sym.scope_level}, const={sym.is_const}, static={sym.is_static}, volatile={sym.is_volatile}")
        
        # ✅ Look up symbol to get qualifiers
        symbol = self.symbol_table.lookup(var_name) if self.symbol_table else None
        
        print(f"  Lookup result for '{var_name}': {symbol is not None}")
        if symbol:
            print(f"  ✅ Symbol found!")
            print(f"    name: {symbol.name}")
            print(f"    data_type: {symbol.data_type}")
            print(f"    scope_level: {symbol.scope_level}")
            print(f"    is_const: {symbol.is_const}")
            print(f"    is_static: {symbol.is_static}")
            print(f"    is_volatile: {symbol.is_volatile}")
            print(f"    is_reference: {symbol.is_reference}")
        else:
            print(f"  ❌ Symbol NOT found in symbol table!")
        print(f"{'='*70}\n")
        
        if symbol:
            # Emit appropriate declaration based on qualifiers
            if symbol.is_reference:
                # References must be initialized - bind to target
                if node.children:
                    target_temp = self.generate_node(node.children[0])
                    self.emit(IROpCode.REF_DECLARE, arg1=var_name, arg2=target_temp,
                            comment=f"reference to {target_temp}")
                else:
                    self.emit(IROpCode.REF_DECLARE, arg1=var_name, arg2='uninitialized',
                            comment="ERROR: uninitialized reference")
                return var_name
                
            elif symbol.is_const and symbol.is_static:
                # static const combination
                self.emit(IROpCode.STATIC_DECLARE, arg1=var_name, arg2=var_type,
                        comment="static const variable")
                
            elif symbol.is_const and symbol.is_volatile:
                # const volatile combination
                self.emit(IROpCode.CONST_DECLARE, arg1=var_name, arg2=var_type,
                        comment="const volatile variable")
                
            elif symbol.is_const:
                self.emit(IROpCode.CONST_DECLARE, arg1=var_name, arg2=var_type,
                        comment="const variable")
                        
            elif symbol.is_static:
                self.emit(IROpCode.STATIC_DECLARE, arg1=var_name, arg2=var_type,
                        comment="static variable")
                        
            elif symbol.is_volatile:
                self.emit(IROpCode.VOLATILE_DECLARE, arg1=var_name, arg2=var_type,
                        comment="volatile variable")
            else:
                self.emit(IROpCode.DECLARE, arg1=var_name, arg2=var_type)
        else:
            # Fallback if no symbol table
            print(f"  ⚠️ WARNING: Using fallback DECLARE for '{var_name}'")
            self.emit(IROpCode.DECLARE, arg1=var_name, arg2=var_type)
        
        # Handle initialization (skip for references as they're initialized above)
        if node.children and not (symbol and symbol.is_reference):
            init_temp = self.generate_node(node.children[0])
            if init_temp:
                self.emit(IROpCode.ASSIGN, result=var_name, arg1=init_temp)
        
        return var_name

    
    def generate_array_declaration(self, node: ASTNode) -> str:
        """✅ ENHANCED: Array declaration with qualifier support"""
        array_info = node.value
        name = array_info['name']
        size = array_info.get('size', 0)
        array_type = node.data_type or 'int'
        
        # Check for qualifiers
        symbol = self.symbol_table.lookup(name) if self.symbol_table else None
        qualifier_comment = ""
        if symbol:
            qualifiers = []
            if symbol.is_const:
                qualifiers.append("const")
            if symbol.is_static:
                qualifiers.append("static")
            if symbol.is_volatile:
                qualifiers.append("volatile")
            if qualifiers:
                qualifier_comment = f"{' '.join(qualifiers)} array"
        
        self.emit(IROpCode.ARRAY_DECL, arg1=name, arg2=array_type, arg3=size, 
                 comment=qualifier_comment)
        
        if node.children:
            init_node = node.children[0]
            for i, elem in enumerate(init_node.children):
                elem_temp = self.generate_node(elem)
                index_temp = self.new_temp()
                self.emit(IROpCode.LOAD_CONST, result=index_temp, arg1=i)
                self.emit(IROpCode.ARRAY_STORE, arg1=name, arg2=index_temp, arg3=elem_temp)
        
        return name

    def generate_assignment(self, node: ASTNode) -> str:
        lvalue_node = node.children[0]
        rvalue_node = node.children[1]
        
        value_temp = self.generate_node(rvalue_node)
        self.generate_store(lvalue_node, value_temp)
        
        return value_temp

    def generate_compound_assignment(self, node: ASTNode) -> str:
        lvalue_node = node.children[0]
        rvalue_node = node.children[1]

        current_val_temp = self.generate_node(lvalue_node)
        rvalue_temp = self.generate_node(rvalue_node)
        
        result_temp = self.new_temp()
        op_map = { '+=': IROpCode.ADD, '-=': IROpCode.SUB, '*=': IROpCode.MUL, 
                  '/=': IROpCode.DIV, '%=': IROpCode.MOD }
        opcode = op_map.get(node.value, IROpCode.ADD)
        
        self.emit(opcode, result=result_temp, arg1=current_val_temp, arg2=rvalue_temp)
        self.generate_store(lvalue_node, result_temp)
        
        return result_temp

    def generate_increment_decrement(self, node: ASTNode) -> str:
        lvalue_node = node.children[0]
        
        current_val_temp = self.generate_node(lvalue_node)
        one_temp = self.new_temp()
        self.emit(IROpCode.LOAD_CONST, result=one_temp, arg1=1)
        
        new_val_temp = self.new_temp()
        opcode = IROpCode.ADD if node.node_type == ASTNodeType.INCREMENT else IROpCode.SUB
        self.emit(opcode, result=new_val_temp, arg1=current_val_temp, arg2=one_temp)
        
        self.generate_store(lvalue_node, new_val_temp)
        
        if node.value == 'prefix':
            return new_val_temp
        else:
            return current_val_temp

    def generate_return_statement(self, node: ASTNode) -> str:
        if node.children:
            return_temp = self.generate_node(node.children[0])
            self.emit(IROpCode.RETURN, arg1=return_temp)
        else:
            self.emit(IROpCode.RETURN)
        return None

    def generate_if_statement(self, node: ASTNode) -> str:
        cond_temp = self.generate_node(node.children[0])
        else_label = self.new_label("ELSE")
        end_label = self.new_label("ENDIF")
        
        self.emit(IROpCode.JUMP_IF_FALSE, arg1=cond_temp, arg2=else_label)
        self.generate_node(node.children[1])
        
        if len(node.children) > 2:
            self.emit(IROpCode.JUMP, arg1=end_label)
            self.emit(IROpCode.LABEL, arg1=else_label)
            self.generate_node(node.children[2])
            self.emit(IROpCode.LABEL, arg1=end_label)
        else:
            self.emit(IROpCode.LABEL, arg1=else_label)
        
        return None

    def generate_binary_op(self, node: ASTNode) -> str:
        op = node.value
        
        # ✅ ENHANCED FIX: Handle cout/cin stream operations (including chained operations)
        # Check if this is a stream operation by examining the data type or identifier
        if op == '<<':
            # Check if left side is cout or ostream type (supports chaining)
            is_cout_operation = False
            
            # First check if left child is explicitly 'cout' (most reliable)
            left_child_value = node.children[0].value
            if isinstance(left_child_value, dict):
                left_child_name = left_child_value.get('name', '')
            else:
                left_child_name = left_child_value if isinstance(left_child_value, str) else ''
            
            if left_child_name == 'cout':
                is_cout_operation = True
            # Check left child's data_type for chained operations (cout << x returns ostream)
            elif node.children[0].data_type == 'ostream':
                is_cout_operation = True
            
            # If it's a cout operation, handle it
            if is_cout_operation:
                # Generate left side (for chained operations)
                if node.children[0].node_type != ASTNodeType.IDENTIFIER or left_child_name != 'cout':
                    self.generate_node(node.children[0])
                
                # Generate right side and print it
                right_temp = self.generate_node(node.children[1])
                self.emit(IROpCode.PRINT, arg1=right_temp, comment="cout")
                return right_temp
        
        if op == '>>':
            # Check if left side is cin or istream type
            is_cin_operation = False
            
            # First check if left child is explicitly 'cin' (most reliable)
            left_child_value = node.children[0].value
            if isinstance(left_child_value, dict):
                left_child_name = left_child_value.get('name', '')
            else:
                left_child_name = left_child_value if isinstance(left_child_value, str) else ''
            
            if left_child_name == 'cin':
                is_cin_operation = True
            # Check left child's data_type for chained operations (cin >> x returns istream)
            elif node.children[0].data_type == 'istream':
                is_cin_operation = True
            
            if is_cin_operation:
                # Generate left side (for chained operations)
                if node.children[0].node_type != ASTNodeType.IDENTIFIER or left_child_name != 'cin':
                    self.generate_node(node.children[0])
                
                self.generate_store(node.children[1], "INPUT_PLACEHOLDER")
                # ✅ FIX: Safely extract right child value
                right_child_value = node.children[1].value
                if isinstance(right_child_value, dict):
                    right_child_name = right_child_value.get('name', '')
                else:
                    right_child_name = right_child_value if isinstance(right_child_value, str) else ''
                self.emit(IROpCode.INPUT, result=right_child_name, comment="cin")
                return right_child_name
        
        # Regular binary operations
        left_temp = self.generate_node(node.children[0])
        right_temp = self.generate_node(node.children[1])
        result_temp = self.new_temp()
        
        opcode_map = {
            '+': IROpCode.ADD, '-': IROpCode.SUB, '*': IROpCode.MUL, '/': IROpCode.DIV,
            '%': IROpCode.MOD, '==': IROpCode.EQ, '!=': IROpCode.NE, '<': IROpCode.LT,
            '>': IROpCode.GT, '<=': IROpCode.LE, '>=': IROpCode.GE, '&&': IROpCode.AND,
            '||': IROpCode.OR,
        }

        opcode = opcode_map.get(op)
        if opcode:
            self.emit(opcode, result=result_temp, arg1=left_temp, arg2=right_temp)
        else:
            self.emit(IROpCode.COMMENT, arg1=f"WARN: Unsupported op {op}")
            return left_temp
        
        return result_temp
    
    def generate_unary_op(self, node: ASTNode) -> str:
        operand_temp = self.generate_node(node.children[0])
        result_temp = self.new_temp()
        
        op = node.value
        if op == '-':
            self.emit(IROpCode.NEG, result=result_temp, arg1=operand_temp)
        elif op == '!':
            self.emit(IROpCode.NOT, result=result_temp, arg1=operand_temp)
        elif op == '+':
            return operand_temp
        
        return result_temp

    def generate_ternary_op(self, node: ASTNode) -> str:
        cond_temp = self.generate_node(node.children[0])
        result_temp = self.new_temp()
        false_label = self.new_label("TERN_F")
        end_label = self.new_label("TERN_E")
        
        self.emit(IROpCode.JUMP_IF_FALSE, arg1=cond_temp, arg2=false_label)
        
        true_temp = self.generate_node(node.children[1])
        self.emit(IROpCode.ASSIGN, result=result_temp, arg1=true_temp)
        self.emit(IROpCode.JUMP, arg1=end_label)
        
        self.emit(IROpCode.LABEL, arg1=false_label)
        false_temp = self.generate_node(node.children[2])
        self.emit(IROpCode.ASSIGN, result=result_temp, arg1=false_temp)
        
        self.emit(IROpCode.LABEL, arg1=end_label)
        return result_temp
    
    def generate_while_loop(self, node: ASTNode) -> str:
        start_label = self.new_label("WHILE_S")
        end_label = self.new_label("WHILE_E")
        self.enter_loop(start_label, end_label)
        
        self.emit(IROpCode.LABEL, arg1=start_label)
        condition_temp = self.generate_node(node.children[0])
        self.emit(IROpCode.JUMP_IF_FALSE, arg1=condition_temp, arg2=end_label)
        self.generate_node(node.children[1])
        self.emit(IROpCode.JUMP, arg1=start_label)
        self.emit(IROpCode.LABEL, arg1=end_label)
        
        self.exit_loop()
        return None

    def generate_do_while_loop(self, node: ASTNode) -> str:
        start_label = self.new_label("DO_S")
        end_label = self.new_label("DO_E")
        self.enter_loop(start_label, end_label)

        self.emit(IROpCode.LABEL, arg1=start_label)
        self.generate_node(node.children[0])
        condition_temp = self.generate_node(node.children[1])
        self.emit(IROpCode.JUMP_IF_FALSE, arg1=condition_temp, arg2=end_label)
        self.emit(IROpCode.JUMP, arg1=start_label)
        self.emit(IROpCode.LABEL, arg1=end_label)

        self.exit_loop()
        return None

    def generate_for_loop(self, node: ASTNode) -> str:
        init_node, cond_node, inc_node, body_node = node.children
        start_label = self.new_label("FOR_S")
        inc_label = self.new_label("FOR_I")
        end_label = self.new_label("FOR_E")
        self.enter_loop(inc_label, end_label)
        
        # Initialization
        if init_node:
            self.generate_node(init_node)
        
        # Start of loop
        self.emit(IROpCode.LABEL, arg1=start_label)
        
        # Condition check
        if cond_node:
            cond_temp = self.generate_node(cond_node)
            self.emit(IROpCode.JUMP_IF_FALSE, arg1=cond_temp, arg2=end_label)
        
        # Body
        if body_node:
            self.generate_node(body_node)
        
        # Increment label (for continue statements)
        self.emit(IROpCode.LABEL, arg1=inc_label)
        
        # Increment
        if inc_node:
            self.generate_node(inc_node)
        
        # Loop back
        self.emit(IROpCode.JUMP, arg1=start_label)
        
        # End label
        self.emit(IROpCode.LABEL, arg1=end_label)
        
        self.exit_loop()
        return None

    def generate_switch_statement(self, node: ASTNode) -> str:
        expr_temp = self.generate_node(node.children[0])
        end_label = self.new_label("SWITCH_E")
        self.enter_switch(end_label)
        
        # Collect all case labels
        case_labels = []
        default_label = None
        
        for i, child in enumerate(node.children[1:]):
            if child.node_type == ASTNodeType.CASE_STATEMENT:
                label = self.new_label(f"CASE")
                case_labels.append((child, label))
            elif child.node_type == ASTNodeType.DEFAULT_STATEMENT:
                default_label = self.new_label("DEFAULT")
        
        # Generate comparison jumps for each case
        for case_node, label in case_labels:
            case_value_temp = self.generate_node(case_node.children[0])
            cmp_temp = self.new_temp()
            self.emit(IROpCode.EQ, result=cmp_temp, arg1=expr_temp, arg2=case_value_temp)
            self.emit(IROpCode.JUMP_IF_FALSE, arg1=cmp_temp, arg2=label, comment=f"Jump if not equal")
        
        # If no case matched, jump to default or end
        if default_label:
            self.emit(IROpCode.JUMP, arg1=default_label)
        else:
            self.emit(IROpCode.JUMP, arg1=end_label)
        
        # Generate case bodies
        for case_node, label in case_labels:
            self.emit(IROpCode.LABEL, arg1=label)
            for stmt in case_node.children[1:]:
                self.generate_node(stmt)
        
        # Generate default body if exists
        if default_label:
            self.emit(IROpCode.LABEL, arg1=default_label)
            for child in node.children[1:]:
                if child.node_type == ASTNodeType.DEFAULT_STATEMENT:
                    for stmt in child.children:
                        self.generate_node(stmt)
        
        self.emit(IROpCode.LABEL, arg1=end_label)
        self.exit_switch()
        return None

    def generate_case_statement(self, node: ASTNode) -> str:
        # Handled by generate_switch_statement
        return None

    def generate_default_statement(self, node: ASTNode) -> str:
        # Handled by generate_switch_statement
        return None

    def generate_break(self, node: ASTNode) -> str:
        # Break jumps to end of innermost loop or switch
        end_label = self.get_loop_end() or self.get_switch_end()
        if end_label:
            self.emit(IROpCode.BREAK, comment=f"jump to {end_label}")
            self.emit(IROpCode.JUMP, arg1=end_label)
        else:
            self.emit(IROpCode.COMMENT, arg1="ERROR: break outside loop/switch")
        return None

    def generate_continue(self, node: ASTNode) -> str:
        # Continue jumps to start of innermost loop
        start_label = self.get_loop_start()
        if start_label:
            self.emit(IROpCode.CONTINUE, comment=f"jump to {start_label}")
            self.emit(IROpCode.JUMP, arg1=start_label)
        else:
            self.emit(IROpCode.COMMENT, arg1="ERROR: continue outside loop")
        return None

    def generate_function_call(self, node: ASTNode) -> str:
        func_name = node.value
        
        # ✅ CHECK: Is this a member function call? (first child is MEMBER_ACCESS)
        if node.children and node.children[0].node_type == ASTNodeType.MEMBER_ACCESS:
            member_access = node.children[0]
            obj_node = member_access.children[0] if member_access.children else None
            method_name = member_access.value
            
            if obj_node:
                # Generate object reference
                obj_temp = self.generate_node(obj_node)
                
                # Pass object as implicit 'this' argument
                self.emit(IROpCode.ARG, arg1=obj_temp, comment="'this' pointer")
                
                # Generate remaining arguments (skip first child which is MEMBER_ACCESS)
                for arg in node.children[1:]:
                    arg_temp = self.generate_node(arg)
                    self.emit(IROpCode.ARG, arg1=arg_temp)
                
                # Call the member function
                result_temp = self.new_temp()
                total_args = len(node.children)  # obj + other args
                self.emit(IROpCode.CALL, result=result_temp, arg1=method_name, 
                        arg2=total_args, comment=f"call member function {method_name}")
                
                return result_temp
        
        # ✅ REGULAR FUNCTION CALL (not a member function)
        # Generate arguments
        for arg in node.children:
            arg_temp = self.generate_node(arg)
            self.emit(IROpCode.ARG, arg1=arg_temp)
        
        # Call function
        result_temp = self.new_temp()
        self.emit(IROpCode.CALL, result=result_temp, arg1=func_name, 
                arg2=len(node.children), comment=f"call {func_name}")
        
        return result_temp

    def generate_literal(self, node: ASTNode) -> str:
        result_temp = self.new_temp()
        self.emit(IROpCode.LOAD_CONST, result=result_temp, arg1=node.value)
        return result_temp

    def generate_identifier_load(self, node: ASTNode) -> str:
        var_name = node.value
        
        # ✅ CHECK: Are we inside a class member function?
        if self.current_function and self.current_class:
            # Check if this identifier is a member of the current class
            if hasattr(self.current_class, 'members') and var_name in self.current_class.members:
                member_symbol = self.current_class.members[var_name]
                
                # Only access through 'this' if it's not a static member
                if not member_symbol.is_static:
                    # Generate: this->member
                    this_temp = self.new_temp()
                    self.emit(IROpCode.LOAD_THIS, result=this_temp, comment="'this' pointer")
                    
                    result_temp = self.new_temp()
                    self.emit(IROpCode.GET_MEMBER, result=result_temp, arg1=this_temp, 
                            arg2=var_name, comment=f"this->{var_name}")
                    return result_temp
        
        # ✅ REGULAR VARIABLE (not a class member or is static)
        result_temp = self.new_temp()
        self.emit(IROpCode.LOAD, result=result_temp, arg1=var_name)
        return result_temp

    def generate_array_load(self, node: ASTNode) -> str:
        array_temp = self.generate_node(node.children[0])
        index_temp = self.generate_node(node.children[1])
        result_temp = self.new_temp()
        self.emit(IROpCode.ARRAY_LOAD, result=result_temp, arg1=array_temp, arg2=index_temp)
        return result_temp

    def generate_address_of(self, node: ASTNode) -> str:
        """Generate address-of operation (&var)"""
        operand_node = node.children[0]
        
        if operand_node.node_type == ASTNodeType.IDENTIFIER:
            result_temp = self.new_temp()
            self.emit(IROpCode.LOAD_ADDRESS, result=result_temp, arg1=operand_node.value,
                     comment=f"&{operand_node.value}")
            return result_temp
        else:
            self.emit(IROpCode.COMMENT, arg1="ERROR: Cannot take address of non-lvalue")
            return self.new_temp()

    def generate_dereference_load(self, node: ASTNode) -> str:
        """Generate dereference load operation (*ptr)"""
        ptr_temp = self.generate_node(node.children[0])
        result_temp = self.new_temp()
        self.emit(IROpCode.DEREF_LOAD, result=result_temp, arg1=ptr_temp,
                 comment="dereference")
        return result_temp

    def generate_member_load(self, node: ASTNode) -> str:
        """Generate member access (obj.member)"""
        obj_temp = self.generate_node(node.children[0])
        member_name = node.value
        result_temp = self.new_temp()
        self.emit(IROpCode.GET_MEMBER, result=result_temp, arg1=obj_temp, 
                 arg2=member_name, comment=f"access .{member_name}")
        return result_temp

    def generate_member_pointer_load(self, node: ASTNode) -> str:
        """Generate member access through pointer (ptr->member)"""
        ptr_temp = self.generate_node(node.children[0])
        member_name = node.value
        result_temp = self.new_temp()
        self.emit(IROpCode.GET_MEMBER_PTR, result=result_temp, arg1=ptr_temp, 
                 arg2=member_name, comment=f"access ->{member_name}")
        return result_temp

    def generate_this(self, node: ASTNode) -> str:
        """Generate 'this' keyword"""
        result_temp = self.new_temp()
        self.emit(IROpCode.LOAD_THIS, result=result_temp, comment="this pointer")
        return result_temp

    # --- Output Methods ---
    
    def print_ir(self):
        """Print all IR instructions"""
        for instr in self.instructions:
            print(instr.to_string())
    
    def save_ir(self, filename: str):
        """Save IR to file"""
        with open(filename, 'w') as f:
            for instr in self.instructions:
                f.write(instr.to_string() + '\n')

    def get_ir_summary(self) -> Dict[str, Any]:
        """Get summary statistics of generated IR"""
        opcode_counts = {}
        label_count = 0
        function_count = 0
        class_count = 0
        
        for instr in self.instructions:
            # Count opcodes
            opcode_name = instr.opcode.name
            opcode_counts[opcode_name] = opcode_counts.get(opcode_name, 0) + 1
            
            # Count specific constructs
            if instr.opcode == IROpCode.LABEL:
                label_count += 1
            elif instr.opcode == IROpCode.FUNC_BEGIN:
                function_count += 1
            elif instr.opcode == IROpCode.CLASS_BEGIN:
                class_count += 1
        
        # Get total instruction count (excluding comments)
        total_instructions = sum(1 for i in self.instructions if i.opcode != IROpCode.COMMENT)
        
        return {
            'total_instructions': len(self.instructions),
            'executable_instructions': total_instructions,
            'temp_variables': self.temp_counter,
            'labels': label_count,
            'functions': function_count,
            'classes': class_count,
            'opcode_distribution': opcode_counts,
            'includes': len(self.includes),
            'defines': len(self.defines)
        }

    def to_string(self) -> str:
        """Convert all IR instructions to a single formatted string"""
        return "\n".join(instr.to_string() for instr in self.instructions)
    
    def get_instructions_as_list(self) -> List[str]:
        """Get all IR instructions as a list of strings"""
        return [instr.to_string() for instr in self.instructions]

        
# ----------------------------------------------------------------------
#  Main Entry Point
# ----------------------------------------------------------------------
def generate_ir(ast: ASTNode, symbol_table=None, includes=None, defines=None) -> List[IRInstruction]:
    """
    Main function to generate IR from AST
    
    Args:
        ast: The abstract syntax tree
        symbol_table: Symbol table from semantic analyzer
        includes: Set of included files from preprocessor
        defines: Dictionary of macro defines from preprocessor
    
    Returns:
        List of IR instructions
    """
    generator = IRGenerator(ast, symbol_table, includes, defines)
    return generator.generate()


# # ----------------------------------------------------------------------
# #  Testing
# # ----------------------------------------------------------------------
# if __name__ == "__main__":
#     print("IR Generator - Enhanced Version")
#     print("=" * 60)
#     print("✅ Full support for:")
#     print("   - const/static/volatile qualifiers")
#     print("   - References")
#     print("   - Inheritance")
#     print("   - Access control (public/private/protected)")
#     print("   - Preprocessor metadata (#include, #define)")
#     print("=" * 60)
    
#     # Example usage (requires actual parser and semantic analyzer)
#     # from parser import parse
#     # from semantic_analyzer import analyze
#     # 
#     # code = """
#     # #include <iostream>
#     # 
#     # class Base {
#     # public:
#     #     int x;
#     # };
#     # 
#     # class Derived : public Base {
#     # private:
#     #     const int y;
#     #     static int count;
#     # public:
#     #     int& getX() { return x; }
#     # };
#     # 
#     # int main() {
#     #     const int a = 5;
#     #     static int b = 10;
#     #     volatile int c = 15;
#     #     int& ref = a;
#     #     return 0;
#     # }
#     # """
#     # 
#     # ast = parse(code)
#     # symbol_table, includes, defines = analyze(ast)
#     # ir_instructions = generate_ir(ast, symbol_table, includes, defines)
#     # 
#     # for instr in ir_instructions:
#     #     print(instr.to_string())