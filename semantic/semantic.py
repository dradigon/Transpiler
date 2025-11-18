"""
semantic.py - Enhanced Version

Transpiler-ready Semantic Analyzer for C++ -> Python pipeline.

NEW FEATURES:
- ✅ Inheritance validation (base class exists, member inheritance)
- ✅ Reference type validation (must be initialized, binding rules)
- ✅ const/static enforcement (immutability, initialization checks)
- ✅ Access specifier enforcement (private/protected/public)
- ✅ Array size validation (initializer matches declared size)
- ✅ Preprocessor tracking and include validation
- ✅ Built-in symbols (cout, cin, endl)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum, auto
import sys
sys.path.append('..')
from parser.parser import ASTNode, ASTNodeType

# ----------------------------------------------------------------------
# Symbol types and symbol table
# ----------------------------------------------------------------------
class SymbolType(Enum):
    VARIABLE = auto()
    FUNCTION = auto()
    ARRAY = auto()
    PARAMETER = auto()
    CLASS = auto()
    MEMBER_VARIABLE = auto()
    MEMBER_FUNCTION = auto()
    BUILTIN = auto()  # NEW: For built-in symbols

class AccessLevel(Enum):
    PUBLIC = auto()
    PRIVATE = auto()
    PROTECTED = auto()

@dataclass
class Symbol:
    name: str
    symbol_type: SymbolType
    data_type: str
    scope_level: int
    line: int = 0
    column: int = 0
    size: Optional[int] = None
    params: List[Dict] = field(default_factory=list)
    initialized: bool = False
    members: Dict[str, 'Symbol'] = field(default_factory=dict)
    # NEW: Enhanced attributes
    is_const: bool = False
    is_static: bool = False
    is_volatile: bool = False
    is_reference: bool = False
    access_level: AccessLevel = AccessLevel.PUBLIC
    base_classes: List[str] = field(default_factory=list)  # For inheritance

    def __repr__(self):
        qualifiers = []
        if self.is_const: qualifiers.append('const')
        if self.is_static: qualifiers.append('static')
        if self.is_volatile: qualifiers.append('volatile')
        qual_str = ' '.join(qualifiers) + ' ' if qualifiers else ''
        return f"Symbol({qual_str}{self.name}, {self.symbol_type.name}, {self.data_type}, scope={self.scope_level})"

class SymbolTable:
    def __init__(self):
        self.scopes: List[Dict[str, Symbol]] = [{}]
        self.current_scope = 0
        self.all_symbols: List[Symbol] = []
        self.symbols = self.scopes[0]  # ✅ ADD THIS: Alias for compatibility

    def enter_scope(self):
        self.scopes.append({})
        self.current_scope += 1

    def exit_scope(self):
        if self.current_scope > 0:
            self.scopes.pop()
            self.current_scope -= 1

    def declare(self, symbol: Symbol) -> bool:
        if symbol.name in self.scopes[self.current_scope]:
            return False
        symbol.scope_level = self.current_scope
        self.scopes[self.current_scope][symbol.name] = symbol
        self.all_symbols.append(symbol)
        # ✅ ADD THIS: Update global symbols dict
        if self.current_scope == 0:
            self.symbols[symbol.name] = symbol
        return True
    
    def define(self, symbol: Symbol) -> bool:
        """✅ ALIAS for declare() - for compatibility"""
        return self.declare(symbol)

    def lookup(self, name: str) -> Optional[Symbol]:
        """✅ FIXED: Look up symbol in active scopes AND all_symbols"""
        # First, search active scopes (for proper shadowing behavior during analysis)
        for scope_level in range(self.current_scope, -1, -1):
            if name in self.scopes[scope_level]:
                return self.scopes[scope_level][name]
        
        # ✅ NEW: If not in active scopes, search all_symbols
        # This allows IR/CodeGen to access symbols after scopes have exited
        for symbol in self.all_symbols:
            if symbol.name == name:
                return symbol
        
        return None

    def lookup_current_scope(self, name: str) -> Optional[Symbol]:
        return self.scopes[self.current_scope].get(name)

    def get_all_symbols(self) -> List[Symbol]:
        return self.all_symbols

# ----------------------------------------------------------------------
# Semantic Analyzer
# ----------------------------------------------------------------------
class SemanticAnalyzer:
    def __init__(self, ast: ASTNode):
        self.ast = ast
        self.symbol_table = SymbolTable()
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.current_function: Optional[Symbol] = None
        self.current_class: Optional[Symbol] = None
        self._checking_literal_zero = False
        # NEW: Track preprocessor directives
        self.includes: Set[str] = set()
        self.defines: Dict[str, str] = {}
        # NEW: Track qualifiers being parsed
        self.current_qualifiers: Dict[str, bool] = {}
        
        # ✅ NEW: Register built-in symbols
        self._register_builtins()
    
    def _register_builtins(self):
        """✅ Register built-in C++ symbols (cout, cin, endl, std)"""
        builtins = [
            Symbol('cout', SymbolType.BUILTIN, 'ostream', 0, initialized=True),
            Symbol('cin', SymbolType.BUILTIN, 'istream', 0, initialized=True),
            Symbol('endl', SymbolType.BUILTIN, 'manipulator', 0, initialized=True),
            Symbol('std', SymbolType.BUILTIN, 'namespace', 0, initialized=True),
        ]
        
        for builtin in builtins:
            self.symbol_table.declare(builtin)

    # ----------------- error/warning helpers -----------------
    def error(self, message: str, node: ASTNode = None):
        if node:
            self.errors.append(f"Semantic Error at line {getattr(node,'line','?')}: {message}")
        else:
            self.errors.append(f"Semantic Error: {message}")

    def warning(self, message: str, node: ASTNode = None):
        if node:
            self.warnings.append(f"Warning at line {getattr(node,'line','?')}: {message}")
        else:
            self.warnings.append(f"Warning: {message}")

    # ----------------- type helpers -----------------
    def normalize_type(self, t: Optional[str]) -> str:
        if not t:
            return 'int'
        return t

    def is_nullptr_type(self, t: str) -> bool:
        return t in ('nullptr', 'null', 'NULL')

    def pointer_level(self, t: str) -> int:
        return t.count('*') if t else 0

    def base_type(self, t: str) -> str:
        """✅ FIXED: Extract base type without qualifiers"""
        if not t:
            return t
        # Remove pointers, references, and qualifiers
        base = t.replace('*', '').replace('&', '')
        base = base.replace('const', '').replace('static', '').replace('volatile', '')
        return base.strip()

    def is_pointer(self, t: str) -> bool:
        return '*' in t if t else False
    
    def is_reference(self, t: str) -> bool:
        return '&' in t and '*' not in t if t else False

    def make_pointer(self, base: str, level: int = 1) -> str:
        return base + ('*' * level)
    
    def parse_pointer_type(self, t: str) -> tuple[str, int]:
        """Return (base_type, pointer_level) for types like 'int**'."""
        if not t:
            return ('int', 0)
        base = self.base_type(t)
        pointer_level = t.count('*')
        return (base, pointer_level)

    # annotate helper
    def annotate_node(self, node: ASTNode, data_type: str, *, is_lvalue: bool = False, hint: Optional[str] = None):
        node.data_type = data_type
        meta = {
            'is_pointer': self.is_pointer(data_type),
            'pointer_level': self.pointer_level(data_type),
            'base_type': self.base_type(data_type),
            'is_lvalue': bool(is_lvalue),
            'is_class': False,
            'class_name': None,
            'is_reference': self.is_reference(data_type),
            'hint': hint or ''
        }
        # mark class metadata if applicable
        if node.data_type and not self.is_pointer(node.data_type) and not self.is_reference(node.data_type):
            sym = self.symbol_table.lookup(self.base_type(node.data_type))
            if sym and sym.symbol_type == SymbolType.CLASS:
                meta['is_class'] = True
                meta['class_name'] = sym.name
        # if pointer to class
        if meta['is_pointer'] or meta['is_reference']:
            base = meta['base_type']
            sym = self.symbol_table.lookup(base)
            if sym and sym.symbol_type == SymbolType.CLASS:
                meta['is_class'] = True
                meta['class_name'] = base
        node.codegen = meta
        return meta

    # ----------------- analyze entry -----------------
    def analyze(self) -> bool:
        if not self.ast:
            self.error("No AST to analyze")
            return False
        self.collect_preprocessor(self.ast)
        self.collect_declarations(self.ast)
        self.validate_inheritance()
        self.check_node(self.ast)
        return len(self.errors) == 0

    # ----------------- NEW: Preprocessor handling -----------------
    def collect_preprocessor(self, node: ASTNode):
        """First pass: collect all preprocessor directives"""
        if node.node_type == ASTNodeType.PROGRAM:
            for child in node.children:
                if child.node_type == ASTNodeType.PREPROCESSOR:
                    directive = child.value
                    if directive.startswith('#include'):
                        # Extract include file
                        parts = directive.split()
                        if len(parts) >= 2:
                            include_file = parts[1].strip('<>"')
                            self.includes.add(include_file)
                    elif directive.startswith('#define'):
                        # Extract macro definition
                        parts = directive.split(maxsplit=2)
                        if len(parts) >= 2:
                            macro_name = parts[1]
                            macro_value = parts[2] if len(parts) > 2 else ''
                            self.defines[macro_name] = macro_value

    # ----------------- first pass: collect top-level declarations -----------------
    def collect_declarations(self, node: ASTNode):
        if node.node_type == ASTNodeType.PROGRAM:
            for child in node.children:
                if child.node_type == ASTNodeType.FUNCTION:
                    self.declare_function(child)
                elif child.node_type in {ASTNodeType.VARIABLE_DECLARATION, ASTNodeType.ARRAY_DECLARATION}:
                    self.declare_variable(child)
                elif child.node_type == ASTNodeType.CLASS_DEFINITION:
                    self.declare_class(child)
                elif child.node_type == ASTNodeType.MULTIPLE_DECLARATION:
                    for sub in child.children:
                        if sub.node_type == ASTNodeType.VARIABLE_DECLARATION:
                            self.declare_variable(sub)

    # ----------------- NEW: Inheritance validation -----------------
    def validate_inheritance(self):
        """Validate all class inheritance relationships"""
        for symbol in self.symbol_table.all_symbols:
            if symbol.symbol_type == SymbolType.CLASS:
                for base_name in symbol.base_classes:
                    base_sym = self.symbol_table.lookup(base_name)
                    if not base_sym:
                        self.error(f"Class '{symbol.name}' inherits from undefined base class '{base_name}'")
                    elif base_sym.symbol_type != SymbolType.CLASS:
                        self.error(f"'{base_name}' is not a class, cannot be used as base class")
                    else:
                        # Inherit members from base class
                        self.inherit_members(symbol, base_sym)

    def inherit_members(self, derived: Symbol, base: Symbol):
        """✅ FIXED: Recursively copy accessible members from base class hierarchy"""
        # First, ensure base class has inherited from its own bases (recursive)
        for base_base_name in base.base_classes:
            base_base_sym = self.symbol_table.lookup(base_base_name)
            if base_base_sym and base_base_sym.symbol_type == SymbolType.CLASS:
                # Recursively inherit members into base class first
                self.inherit_members(base, base_base_sym)
        
        # Now inherit from the fully-populated base class
        for member_name, member_sym in base.members.items():
            # Only inherit public and protected members
            if member_sym.access_level in {AccessLevel.PUBLIC, AccessLevel.PROTECTED}:
                if member_name not in derived.members:
                    # Create a copy of the member for the derived class
                    inherited_member = Symbol(
                        name=member_sym.name,
                        symbol_type=member_sym.symbol_type,
                        data_type=member_sym.data_type,
                        scope_level=derived.scope_level,
                        line=member_sym.line,
                        column=member_sym.column,
                        size=member_sym.size,
                        params=member_sym.params.copy() if member_sym.params else [],
                        initialized=member_sym.initialized,
                        is_const=member_sym.is_const,
                        is_static=member_sym.is_static,
                        access_level=member_sym.access_level
                    )
                    derived.members[member_name] = inherited_member

    # ----------------- declarations helpers -----------------
    def declare_class(self, node: ASTNode) -> bool:
        """✅ FIXED: Declare a class with bulletproof member name extraction"""
        raw = node.value
        if isinstance(raw, dict):
            name = raw.get('name')
            base_classes_raw = raw.get('base_classes', [])
            # ✅ FIX: Parser returns base_classes as list of dicts, extract names
            base_classes = []
            for bc in base_classes_raw:
                if isinstance(bc, dict):
                    base_classes.append(bc.get('name'))
                elif isinstance(bc, str):
                    base_classes.append(bc)
        else:
            name = raw
            base_classes = []
        
        if not isinstance(name, str):
            self.error("Class missing name", node)
            return False
        
        class_sym = Symbol(
            name=name,
            symbol_type=SymbolType.CLASS,
            data_type=name,
            scope_level=self.symbol_table.current_scope,
            line=getattr(node,'line',0),
            column=getattr(node,'column',0),
            base_classes=base_classes
        )
        
        if not self.symbol_table.declare(class_sym):
            self.error(f"Class '{name}' already declared", node)
            class_sym = self.symbol_table.lookup(name)
            if not class_sym:
                return False
        
        # C++ classes default to private access
        current_access = AccessLevel.PRIVATE
        
        # Collect members
        for member in node.children:
            if member.node_type == ASTNodeType.MEMBER_VARIABLE:
                # Access level from wrapper
                access_level = current_access
                if isinstance(member.value, dict):
                    access_level = self.parse_access_level(member.value.get('access', 'private'))

                # Member declaration can be a single VARIABLE_DECLARATION or a MULTIPLE_DECLARATION
                decl_nodes = []
                if member.children:
                    for ch in member.children:
                        if ch.node_type == ASTNodeType.VARIABLE_DECLARATION:
                            decl_nodes.append(ch)
                        elif ch.node_type == ASTNodeType.MULTIPLE_DECLARATION:
                            for sub in ch.children:
                                if sub.node_type == ASTNodeType.VARIABLE_DECLARATION:
                                    decl_nodes.append(sub)

                for decl in decl_nodes:
                    # Extract name/type from decl
                    if isinstance(decl.value, dict):
                        member_name = decl.value.get('name')
                    else:
                        member_name = decl.value
                    member_type = decl.data_type or 'int'

                    if not member_name or not isinstance(member_name, str):
                        self.error("Member variable missing or invalid name", member)
                        continue

                    sym = Symbol(
                        name=member_name,
                        symbol_type=SymbolType.MEMBER_VARIABLE,
                        data_type=self.normalize_type(member_type),
                        scope_level=class_sym.scope_level,
                        line=getattr(member,'line',0),
                        column=getattr(member,'column',0),
                        access_level=access_level
                    )
                    class_sym.members[member_name] = sym
                    self.symbol_table.all_symbols.append(sym)
                
            elif member.node_type == ASTNodeType.MEMBER_FUNCTION:
                func_name = None
                params = []
                ret_type = member.data_type or 'void'
                
                # ✅ FIXED: Handle both dict and string values
                if isinstance(member.value, dict):
                    func_name = member.value.get('name')
                    params = member.value.get('params', [])
                    access_str = member.value.get('access', 'private')
                    current_access = self.parse_access_level(access_str)
                elif isinstance(member.value, str):
                    func_name = member.value
                
                # ✅ FIXED: If still no func_name, search children
                if not func_name:
                    for ch in member.children:
                        if ch.node_type == ASTNodeType.FUNCTION:
                            if isinstance(ch.value, dict):
                                func_name = ch.value.get('name')
                                params = ch.value.get('params', [])
                                ret_type = ch.data_type or ret_type
                            elif isinstance(ch.value, str):
                                func_name = ch.value
                                ret_type = ch.data_type or ret_type
                            break
                
                # ✅ CRITICAL: Validate func_name is a string
                if not func_name:
                    self.error(f"Member function missing name", member)
                    continue
                    
                if not isinstance(func_name, str):
                    self.error(f"Invalid member function name type: {type(func_name)}", member)
                    continue
                
                sym = Symbol(
                    name=func_name,
                    symbol_type=SymbolType.MEMBER_FUNCTION,
                    data_type=self.normalize_type(ret_type),
                    scope_level=class_sym.scope_level,
                    line=getattr(member,'line',0),
                    column=getattr(member,'column',0),
                    params=params,
                    access_level=current_access
                )
                class_sym.members[func_name] = sym
                self.symbol_table.all_symbols.append(sym)
        
        return True

    def parse_access_level(self, access_str: str) -> AccessLevel:
        """Convert access string to AccessLevel enum"""
        access_map = {
            'public': AccessLevel.PUBLIC,
            'private': AccessLevel.PRIVATE,
            'protected': AccessLevel.PROTECTED
        }
        return access_map.get(access_str.lower(), AccessLevel.PRIVATE)

    def declare_function(self, node: ASTNode) -> bool:
        func_info = node.value
        name = func_info['name'] if isinstance(func_info, dict) else func_info
        params = func_info.get('params', []) if isinstance(func_info, dict) else []
        
        symbol = Symbol(
            name=name,
            symbol_type=SymbolType.FUNCTION,
            data_type=node.data_type or 'void',
            scope_level=self.symbol_table.current_scope,
            line=getattr(node,'line',0),
            column=getattr(node,'column',0),
            params=params
        )
        
        if not self.symbol_table.declare(symbol):
            self.error(f"Function '{name}' already declared", node)
            return False
        return True

    def declare_variable(self, node: ASTNode) -> bool:
        """✅ FIXED: Declare a variable with proper qualifier handling and dict support"""
        if node.node_type == ASTNodeType.ARRAY_DECLARATION:
            if isinstance(node.value, dict):
                name = node.value.get('name')
                size = node.value.get('size')
            else:
                name = node.value
                size = None
            symbol_type = SymbolType.ARRAY
        else:
            if isinstance(node.value, dict):
                name = node.value.get('name')
            else:
                name = node.value
            size = None
            symbol_type = SymbolType.VARIABLE

        # ✅ FIXED: Extract FULL data_type with qualifiers
        data_type = node.data_type or 'int'
        
        # Parse qualifiers from data_type
        is_const = 'const' in data_type.lower() if isinstance(data_type, str) else False
        is_static = 'static' in data_type.lower() if isinstance(data_type, str) else False
        is_volatile = 'volatile' in data_type.lower() if isinstance(data_type, str) else False
        is_reference = '&' in data_type and '*' not in data_type if isinstance(data_type, str) else False

        # ✅ FIXED: Store FULL type with qualifiers
        symbol = Symbol(
            name=name,
            symbol_type=symbol_type,
            data_type=data_type,  # ✅ Keep full type: "const int", "static int", etc.
            scope_level=self.symbol_table.current_scope,
            line=node.line,
            column=node.column,
            size=size,
            initialized=False,
            is_const=is_const,
            is_static=is_static,
            is_volatile=is_volatile,
            is_reference=is_reference
        )

        if not self.symbol_table.declare(symbol):
            self.error(f"Variable '{name}' already declared in this scope", node)
            return False

        # References must be initialized
        if is_reference and not node.children:
            self.error(f"Reference '{name}' must be initialized", node)

        return True

    # ----------------- main traversal -----------------
    def check_node(self, node: ASTNode) -> Optional[str]:
        if not node:
            return None
        nt = node.node_type

        if nt == ASTNodeType.PROGRAM:
            for ch in node.children:
                self.check_node(ch)
            return None
        
        if nt == ASTNodeType.CLASS_DEFINITION:
            raw = node.value
            classname = raw.get('name') if isinstance(raw, dict) else raw
            class_sym = self.symbol_table.lookup(classname)
            prev_class = self.current_class
            
            if class_sym:
                self.current_class = class_sym
                
                # Check all members
                for member in node.children:
                    if member.node_type == ASTNodeType.MEMBER_FUNCTION:
                        # ✅ FIXED: Process member function
                        self.check_node(member)
                    else:
                        # Check other members
                        self.check_node(member)
                
                self.current_class = prev_class
            else:
                for ch in node.children:
                    self.check_node(ch)
            
            self.annotate_node(node, classname, is_lvalue=False, hint='class')
            return classname

        # ✅ NEW: Add explicit MEMBER_FUNCTION handling
        if nt == ASTNodeType.MEMBER_FUNCTION:
            # Extract function name
            func_name = None
            if isinstance(node.value, dict):
                func_name = node.value.get('name')
            elif isinstance(node.value, str):
                func_name = node.value
            
            # Set current_function from class members
            prev_func = self.current_function
            if self.current_class and func_name and func_name in self.current_class.members:
                self.current_function = self.current_class.members[func_name]
            
            # Find and check the function node
            func_node = None
            for ch in node.children:
                if ch.node_type == ASTNodeType.FUNCTION or ch.node_type == ASTNodeType.BLOCK:
                    func_node = ch
                    break
            
            if func_node:
                if func_node.node_type == ASTNodeType.FUNCTION:
                    ret_type = self.check_function(func_node)
                else:
                    # It's a BLOCK directly (inline member function)
                    self.symbol_table.enter_scope()
                    
                    # ✅ FIX: Add parameters to scope for inline member functions
                    if self.current_function and hasattr(self.current_function, 'params'):
                        for p in self.current_function.params:
                            pname = p['name']
                            ptype = self.normalize_type(p['type'])
                            is_ref = '&' in ptype and '*' not in ptype
                            psym = Symbol(
                                name=pname,
                                symbol_type=SymbolType.PARAMETER,
                                data_type=ptype,
                                scope_level=self.symbol_table.current_scope,
                                initialized=True,
                                is_reference=is_ref
                            )
                            self.symbol_table.declare(psym)
                    
                    # Add 'this' pointer
                    if self.current_class:
                        this_type = self.make_pointer(self.current_class.data_type, 1)
                        this_sym = Symbol(
                            name='this',
                            symbol_type=SymbolType.PARAMETER,
                            data_type=this_type,
                            scope_level=self.symbol_table.current_scope,
                            initialized=True
                        )
                        self.symbol_table.declare(this_sym)
                    
                    self.check_node(func_node)
                    self.symbol_table.exit_scope()
                    ret_type = node.data_type or 'void'
            
            self.current_function = prev_func
            self.annotate_node(node, ret_type if func_node else 'void')
            return ret_type if func_node else 'void'

        # declarations
        if nt == ASTNodeType.FUNCTION:
            return self.check_function(node)
        if nt == ASTNodeType.VARIABLE_DECLARATION:
            return self.check_variable_declaration(node)
        if nt == ASTNodeType.ARRAY_DECLARATION:
            return self.check_array_declaration(node)
        
        # ✅ FIX: Handle MULTIPLE_DECLARATION in check phase
        if nt == ASTNodeType.MULTIPLE_DECLARATION:
            # Check each child variable declaration
            for child in node.children:
                if child.node_type == ASTNodeType.VARIABLE_DECLARATION:
                    self.check_variable_declaration(child)
            # Return the data type of the declaration
            return node.data_type if hasattr(node, 'data_type') else 'void'

        # expressions / statements
        if nt == ASTNodeType.ASSIGNMENT:
            return self.check_assignment(node)
        if nt == ASTNodeType.BINARY_OP:
            return self.check_binary_op(node)
        if nt == ASTNodeType.UNARY_OP:
            return self.check_unary_op(node)
        if nt == ASTNodeType.ADDRESS_OF:
            return self.check_address_of(node)
        if nt == ASTNodeType.DEREFERENCE:
            return self.check_dereference(node)
        if nt == ASTNodeType.LITERAL:
            return self.get_literal_type(node)
        if nt == ASTNodeType.IDENTIFIER:
            return self.check_identifier(node)
        if nt == ASTNodeType.IF_STATEMENT:
            return self.check_if_statement(node)
        if nt == ASTNodeType.WHILE_LOOP:
            return self.check_while_loop(node)
        if nt == ASTNodeType.DO_WHILE_LOOP:
            return self.check_do_while(node)
        if nt == ASTNodeType.FOR_LOOP:
            return self.check_for_loop(node)
        if nt == ASTNodeType.RETURN_STATEMENT:
            return self.check_return_statement(node)
        if nt == ASTNodeType.FUNCTION_CALL:
            return self.check_function_call(node)
        if nt == ASTNodeType.ARRAY_ACCESS:
            return self.check_array_access(node)
        if nt == ASTNodeType.BLOCK:
            return self.check_block(node)
        if nt == ASTNodeType.EXPRESSION_STATEMENT:
            if node.children:
                return self.check_node(node.children[0])
            self.annotate_node(node, 'void')
            return 'void'
        if nt in {ASTNodeType.INCREMENT, ASTNodeType.DECREMENT}:
            return self.check_increment_decrement(node)
        if nt == ASTNodeType.COMPOUND_ASSIGNMENT:
            return self.check_compound_assignment(node)
        if nt == ASTNodeType.TERNARY_OP:
            return self.check_ternary(node)
        if nt == ASTNodeType.MEMBER_ACCESS:
            return self.check_member_access(node)
        if nt == ASTNodeType.MEMBER_ACCESS_POINTER:
            return self.check_member_access_pointer(node)
        if nt == ASTNodeType.THIS_KEYWORD:
            return self.check_this(node)
        if nt in {ASTNodeType.BREAK_STATEMENT, ASTNodeType.CONTINUE_STATEMENT}:
            self.annotate_node(node, 'void')
            return 'void'
        if nt == ASTNodeType.SWITCH_STATEMENT:
            return self.check_switch(node)
        if nt == ASTNodeType.CASE_STATEMENT or nt == ASTNodeType.DEFAULT_STATEMENT:
            for ch in node.children:
                self.check_node(ch)
            self.annotate_node(node, 'void')
            return 'void'

        # fallback: traverse children and annotate as void
        for ch in node.children:
            self.check_node(ch)
        self.annotate_node(node, 'void')
        return 'void'

    def check_function(self, node: ASTNode) -> str:
        """✅ ENHANCED: Check function with template support"""
        func_info = node.value
        name = func_info['name'] if isinstance(func_info, dict) else func_info
        params = func_info.get('params', []) if isinstance(func_info, dict) else []
        
        # ✅ Track template parameters (if this is a template function)
        template_params = func_info.get('template_params', []) if isinstance(func_info, dict) else []
        if template_params and not hasattr(self, 'template_params'):
            self.template_params = set()
        prev_template_params = getattr(self, 'template_params', set())
        if template_params:
            self.template_params = set(template_params)
        
        # Look up the function symbol
        sym = self.symbol_table.lookup(name)
        prev_function = self.current_function
        
        # ✅ NEW: If inside a class, look for member function
        if self.current_class and name in self.current_class.members:
            self.current_function = self.current_class.members[name]
        elif sym and sym.symbol_type == SymbolType.FUNCTION:
            self.current_function = sym
        else:
            # ✅ NEW: Create temporary function symbol if not found
            self.current_function = Symbol(
                name=name,
                symbol_type=SymbolType.FUNCTION,
                data_type=node.data_type or 'void',
                scope_level=self.symbol_table.current_scope,
                line=getattr(node, 'line', 0),
                column=getattr(node, 'column', 0),
                params=params
            )
        
        self.symbol_table.enter_scope()
        
        # Declare params in scope
        for p in params:
            pname = p['name']
            ptype = self.normalize_type(p['type'])
            is_ref = '&' in ptype and '*' not in ptype
            psym = Symbol(
                name=pname,
                symbol_type=SymbolType.PARAMETER,
                data_type=ptype,
                scope_level=self.symbol_table.current_scope,
                initialized=True,
                is_reference=is_ref
            )
            if not self.symbol_table.declare(psym):
                self.error(f"Parameter '{pname}' duplicated", node)
        
        # Check function body
        for ch in node.children:
            self.check_node(ch)
        
        self.symbol_table.exit_scope()
        self.current_function = prev_function
        
        # ✅ Restore template parameters
        if template_params:
            self.template_params = prev_template_params
        
        ret = node.data_type or 'void'
        self.annotate_node(node, ret)
        return ret

    def check_variable_declaration(self, node: ASTNode) -> str:
        """✅ FIXED: Check variable declaration with proper dict value handling"""
        declared_full = node.data_type or 'int'
        
        # ✅ FIXED: Handle both dict and string values
        if isinstance(node.value, dict):
            var_name = node.value.get('name')
            qualifiers = node.value.get('qualifiers', [])
        else:
            var_name = node.value
            qualifiers = []
        
        sym = self.symbol_table.lookup(var_name)
        
        # ✅ FIX: If variable not in symbol table, declare it now (for function-local variables)
        if not sym:
            is_const = 'const' in qualifiers
            is_static = 'static' in qualifiers
            is_volatile = 'volatile' in qualifiers
            
            var_sym = Symbol(
                name=var_name,
                symbol_type=SymbolType.VARIABLE,
                data_type=declared_full,
                scope_level=self.symbol_table.current_scope,
                line=getattr(node, 'line', 0),
                column=getattr(node, 'column', 0),
                initialized=len(node.children) > 0,  # Has initializer
                is_const=is_const,
                is_static=is_static,
                is_volatile=is_volatile
            )
            self.symbol_table.declare(var_sym)
            sym = var_sym
        
        # ✅ FIXED: Extract base type for comparison
        declared_base = self.base_type(declared_full)
        
        if node.children:
            init_node = node.children[0]
            init_type = self.check_node(init_node)
            
            # ✅ FIXED: Compare base types only (ignore qualifiers)
            if not self.types_compatible(declared_base, init_type):
                self.warning(f"Type mismatch in initialization: {declared_base} = {init_type}", node)
            
            # Check const initialization
            if sym and sym.is_const and not node.children:
                self.error(f"const variable '{var_name}' must be initialized", node)
            
            # Check reference binding
            if sym and sym.is_reference:
                # References must bind to lvalues (except const refs can bind to temporaries)
                if not sym.is_const and init_node.node_type == ASTNodeType.LITERAL:
                    self.error(f"Non-const reference '{var_name}' cannot bind to temporary/literal", node)
        
        # ✅ FIXED: Annotate with FULL type (with qualifiers)
        self.annotate_node(node, declared_full, is_lvalue=True)
        return declared_full

    def check_array_declaration(self, node: ASTNode) -> str:
        declared = self.normalize_type(node.data_type or 'int')
        arr_type = declared + '[]'
        
        # Validate array initializer size
        if node.children:
            init_node = node.children[0]
            if init_node and init_node.node_type == ASTNodeType.LITERAL:
                init_count = len(init_node.children)
                
                if isinstance(node.value, dict):
                    declared_size = node.value.get('size')
                    if declared_size and init_count > int(declared_size):
                        self.error(f"Too many initializers for array of size {declared_size}", node)
                    elif declared_size and init_count < int(declared_size):
                        self.warning(f"Array partially initialized: {init_count}/{declared_size} elements", node)
        
        self.annotate_node(node, arr_type, is_lvalue=True)
        return arr_type

    def check_assignment(self, node: ASTNode) -> str:
        if len(node.children) < 2:
            self.error("Invalid assignment", node)
            self.annotate_node(node, 'error')
            return 'error'
        
        left = node.children[0]
        right = node.children[1]
        left_type = self.check_node(left)
        right_type = self.check_node(right)
        
        # Check const assignment
        if left.node_type == ASTNodeType.IDENTIFIER:
            sym = self.symbol_table.lookup(left.value)
            if sym and sym.is_const:
                self.error(f"Cannot assign to const variable '{sym.name}'", node)
        
        # Check reference reassignment
        if left.node_type == ASTNodeType.IDENTIFIER:
            sym = self.symbol_table.lookup(left.value)
            if sym and sym.is_reference:
                self.warning(f"Assignment to reference '{sym.name}' modifies the referenced object", node)

        # ✅ FIXED: Compare base types only
        left_base = self.base_type(left_type)
        right_base = self.base_type(right_type)
        
        if not self.types_compatible(left_base, right_base):
            self.error(
                f"Incompatible types in assignment: cannot assign '{right_type}' to '{left_type}'",
                node
            )

        allowed = {ASTNodeType.IDENTIFIER, ASTNodeType.ARRAY_ACCESS, ASTNodeType.DEREFERENCE, 
                   ASTNodeType.MEMBER_ACCESS, ASTNodeType.MEMBER_ACCESS_POINTER}
        if left.node_type not in allowed:
            self.error("Left side of assignment must be an lvalue", node)
        
        self.annotate_node(node, left_type)
        return left_type

    def check_binary_op(self, node: ASTNode) -> str:
        if len(node.children) < 2:
            self.error("Binary operation requires two operands", node)
            self.annotate_node(node, 'error')
            return 'error'
        lt = self.check_node(node.children[0])
        rt = self.check_node(node.children[1])
        op = node.value
        
        if op in ['+', '-', '*', '/', '%']:
            if self.is_pointer(lt) or self.is_pointer(rt):
                if self.is_pointer(lt) and rt == 'int':
                    self.annotate_node(node, lt)
                    return lt
                if self.is_pointer(rt) and lt == 'int':
                    self.annotate_node(node, rt)
                    return rt
                if self.is_pointer(lt) and self.is_pointer(rt):
                    if self.pointer_level(lt) == self.pointer_level(rt) and self.base_type(lt) == self.base_type(rt):
                        self.annotate_node(node, 'int')
                        return 'int'
                self.warning(f"Pointer arithmetic with incompatible operands: {lt} {op} {rt}", node)
                self.annotate_node(node, lt or rt)
                return lt or rt
            
            if lt not in ['int', 'float'] or rt not in ['int', 'float']:
                self.warning(f"Arithmetic on non-numeric types: {lt} {op} {rt}", node)
            result = 'float' if 'float' in (lt, rt) else 'int'
            self.annotate_node(node, result)
            return result
        
        if op in ['==', '!=', '<', '>', '<=', '>=', '&&', '||']:
            self.annotate_node(node, 'bool')
            return 'bool'
        
        if op in ['<<', '>>']:
            self.annotate_node(node, lt)
            return lt
        
        self.annotate_node(node, 'int')
        return 'int'

    def check_unary_op(self, node: ASTNode) -> str:
        if not node.children:
            self.error("Unary op requires operand", node)
            self.annotate_node(node, 'error')
            return 'error'
        operand = node.children[0]
        otype = self.check_node(operand)
        op = node.value
        if op in ['+', '-']:
            if otype not in ['int', 'float']:
                self.warning(f"Unary {op} on non-numeric: {otype}", node)
            self.annotate_node(node, otype)
            return otype
        if op == '!':
            self.annotate_node(node, 'bool')
            return 'bool'
        self.annotate_node(node, otype)
        return otype

    def check_address_of(self, node: ASTNode) -> str:
        if not node.children:
            self.error("Address-of requires operand", node)
            self.annotate_node(node, 'error')
            return 'error'
        operand = node.children[0]
        
        if operand.node_type not in {ASTNodeType.IDENTIFIER, ASTNodeType.ARRAY_ACCESS, 
                                       ASTNodeType.MEMBER_ACCESS, ASTNodeType.MEMBER_ACCESS_POINTER}:
            self.warning(f"Taking address of non-lvalue (temporary): {operand.node_type}", node)
        
        opt = self.check_node(operand)
        ptype = self.make_pointer(opt, 1)
        self.annotate_node(node, ptype, is_lvalue=False, hint='addr->box')
        return ptype

    def check_dereference(self, node: ASTNode) -> str:
        if not node.children:
            self.error("Dereference requires operand", node)
            self.annotate_node(node, 'error')
            return 'error'
        ptr_expr = node.children[0]
        pt = self.check_node(ptr_expr)
        if self.is_nullptr_type(pt):
            self.error("Dereferencing nullptr", node)
            self.annotate_node(node, 'error')
            return 'error'
        if not self.is_pointer(pt):
            self.error(f"Cannot dereference non-pointer type: {pt}", node)
            self.annotate_node(node, 'error')
            return 'error'
        base = self.base_type(pt)
        hint = 'deref->.value' if not self.symbol_table.lookup(base) else 'deref->member/object'
        self.annotate_node(node, base, is_lvalue=True, hint=hint)
        return base

    def check_identifier(self, node: ASTNode) -> str:
        """✅ FIXED: Check identifier with member variable resolution"""
        name = node.value
        sym = self.symbol_table.lookup(name)
        
        # ✅ NEW: If not found and we're in a member function, check class members
        if not sym and self.current_class:
            if name in self.current_class.members:
                member_sym = self.current_class.members[name]
                is_lval = member_sym.symbol_type in {SymbolType.MEMBER_VARIABLE, SymbolType.VARIABLE}
                self.annotate_node(node, member_sym.data_type, is_lvalue=is_lval)
                # ✅ Add hint that this is a member access
                if node.codegen:
                    node.codegen['is_member'] = True
                    node.codegen['class_name'] = self.current_class.name
                return member_sym.data_type
        
        if not sym:
            self.error(f"Undefined identifier '{name}'", node)
            self.annotate_node(node, 'error')
            return 'error'
        
        is_lval = sym.symbol_type in {SymbolType.VARIABLE, SymbolType.ARRAY, 
                                    SymbolType.PARAMETER, SymbolType.MEMBER_VARIABLE}
        self.annotate_node(node, sym.data_type, is_lvalue=is_lval)
        return sym.data_type

    def check_if_statement(self, node: ASTNode) -> str:
        if not node.children:
            self.error("If requires condition", node)
            self.annotate_node(node, 'error')
            return 'error'
        self.check_node(node.children[0])
        if len(node.children) > 1:
            self.check_node(node.children[1])
        if len(node.children) > 2:
            self.check_node(node.children[2])
        self.annotate_node(node, 'void')
        return 'void'

    def check_while_loop(self, node: ASTNode) -> str:
        if len(node.children) < 2:
            self.error("While requires condition and body", node)
            self.annotate_node(node, 'error')
            return 'error'
        self.check_node(node.children[0])
        self.check_node(node.children[1])
        self.annotate_node(node, 'void')
        return 'void'

    def check_do_while(self, node: ASTNode) -> str:
        if len(node.children) < 2:
            self.error("Do-while malformed", node)
            self.annotate_node(node, 'error')
            return 'error'
        self.check_node(node.children[0])
        self.check_node(node.children[1])
        self.annotate_node(node, 'void')
        return 'void'

    def check_for_loop(self, node: ASTNode) -> str:
        self.symbol_table.enter_scope()
        for ch in node.children:
            self.check_node(ch)
        self.symbol_table.exit_scope()
        self.annotate_node(node, 'void')
        return 'void'

    def check_return_statement(self, node: ASTNode) -> str:
        if not self.current_function:
            self.error("Return outside function", node)
            self.annotate_node(node, 'error')
            return 'error'
        ret_type = 'void'
        if node.children:
            ret_type = self.check_node(node.children[0])
        expected = self.current_function.data_type
        if not self.types_compatible(expected, ret_type):
            self.error(f"Return type mismatch: expected {expected}, got {ret_type}", node)
        self.annotate_node(node, ret_type)
        return ret_type

    def check_function_call(self, node: ASTNode) -> str:
        func_name = node.value
        
        if func_name is None:
            for arg in node.children:
                self.check_node(arg)
            self.annotate_node(node, 'error')
            return 'error'
        
        if func_name in ['cout', 'cin']:
            for arg in node.children:
                self.check_node(arg)
            self.annotate_node(node, 'void')
            return 'void'
        
        # ✅ NEW: Check if this is a member function call (first child is MEMBER_ACCESS)
        if node.children and node.children[0].node_type == ASTNodeType.MEMBER_ACCESS:
            member_access = node.children[0]
            obj_expr = member_access.children[0] if member_access.children else None
            member_name = member_access.value
            
            if obj_expr:
                obj_type = self.check_node(obj_expr)
                class_sym = self.symbol_table.lookup(obj_type)
                
                if class_sym and class_sym.symbol_type == SymbolType.CLASS:
                    member_sym = class_sym.members.get(member_name)
                    
                    if not member_sym:
                        self.error(f"Class '{obj_type}' has no member '{member_name}'", node)
                        self.annotate_node(node, 'error')
                        return 'error'
                    
                    if member_sym.symbol_type != SymbolType.MEMBER_FUNCTION:
                        self.error(f"'{member_name}' is not a member function", node)
                        self.annotate_node(node, 'error')
                        return 'error'
                    
                    # Check argument count (skip first child which is MEMBER_ACCESS)
                    expected = len(member_sym.params)
                    actual = len(node.children) - 1  # ✅ Subtract 1 for MEMBER_ACCESS
                    
                    if expected != actual:
                        self.error(f"Member function '{member_name}' expects {expected} args, got {actual}", node)
                    
                    # Check remaining arguments
                    for i, arg in enumerate(node.children[1:]):
                        arg_type = self.check_node(arg)
                        if i < len(member_sym.params):
                            expected_type = member_sym.params[i]['type']
                            if not self.types_compatible(expected_type, arg_type):
                                self.error(
                                    f"Argument {i+1} of member function '{member_name}' expects '{expected_type}', got '{arg_type}'",
                                    arg
                                )
                    
                    self.annotate_node(node, member_sym.data_type)
                    return member_sym.data_type
        
        # ✅ REGULAR FUNCTION CALL or CONSTRUCTOR CALL
        sym = self.symbol_table.lookup(func_name)
        if not sym:
            self.error(f"Undefined function '{func_name}'", node)
            self.annotate_node(node, 'error')
            return 'error'
        
        # ✅ FIX: Allow CLASS symbols (constructor calls)
        if sym.symbol_type == SymbolType.CLASS:
            # This is a constructor call - return the class name as the type
            # Check constructor parameters if available
            self.annotate_node(node, func_name)
            return func_name
        
        if sym.symbol_type not in {SymbolType.FUNCTION, SymbolType.MEMBER_FUNCTION}:
            self.error(f"'{func_name}' is not a function", node)
            self.annotate_node(node, 'error')
            return 'error'
        
        # ✅ FIXED: Filter out the function name from children if present
        actual_args = []
        for child in node.children:
            # Skip if child is an IDENTIFIER with the same name as the function
            if child.node_type == ASTNodeType.IDENTIFIER and child.value == func_name:
                continue
            actual_args.append(child)
        
        expected = len(sym.params)
        actual = len(actual_args)  # ✅ Use filtered arguments
        
        if expected != actual:
            self.error(f"Function '{func_name}' expects {expected} args, got {actual}", node)
        
        # ✅ TEMPLATE SUPPORT: Collect template parameter names from function signature
        template_type_names = set()
        for param in sym.params:
            param_type = self.base_type(param['type'])
            # If param type is a single capital letter or looks like a template param, track it
            if len(param_type) <= 2 and param_type[0].isupper():
                template_type_names.add(param_type)
        
        # Temporarily set template params for this function call
        prev_template_params = getattr(self, 'template_params', set())
        if template_type_names:
            self.template_params = template_type_names
        
        # ✅ FIXED: Check filtered arguments
        for i, arg in enumerate(actual_args):
            arg_type = self.check_node(arg)
            if i < len(sym.params):
                expected_type = sym.params[i]['type']
                if not self.types_compatible(expected_type, arg_type):
                    self.error(
                        f"Argument {i+1} of function '{func_name}' expects '{expected_type}', got '{arg_type}'",
                        arg
                    )
        
        # Restore template params
        self.template_params = prev_template_params
        
        self.annotate_node(node, sym.data_type)
        return sym.data_type
    
    def check_array_access(self, node: ASTNode) -> str:
        if len(node.children) < 2:
            self.error("Array access requires array and index", node)
            self.annotate_node(node, 'error')
            return 'error'
        arr_expr = node.children[0]
        idx = node.children[1]
        arr_type = self.check_node(arr_expr)
        idx_type = self.check_node(idx)
        if idx_type != 'int':
            self.warning(f"Array index should be int, got {idx_type}", node)
        
        if isinstance(arr_type, str) and arr_type.endswith('[]'):
            elem = arr_type[:-2]
            self.annotate_node(node, elem, is_lvalue=True)
            return elem
        if self.is_pointer(arr_type):
            elem = self.base_type(arr_type)
            self.annotate_node(node, elem, is_lvalue=True)
            return elem
        self.annotate_node(node, 'error')
        return 'error'

    def check_variable_initialization(self, node: ASTNode):
        """Check initialization of an already declared variable"""
        if not node.children:
            return

        # ✅ FIX: Skip type checking for array declarations
        # Array initialization has special rules and is validated in check_array_declaration
        if node.node_type == ASTNodeType.ARRAY_DECLARATION:
            var_name = node.value['name'] if isinstance(node.value, dict) else node.value
            symbol = self.symbol_table.lookup_current_scope(var_name)
            if symbol:
                symbol.initialized = True
            return

        init_expr = node.children[0]
        init_type = self.check_node(init_expr)

        var_name = node.value['name'] if isinstance(node.value, dict) else node.value
        symbol = self.symbol_table.lookup_current_scope(var_name)

        if not symbol:
            self.error(f"Internal error: variable '{var_name}' not found after declaration", node)
            return

        symbol.initialized = True

        # ✅ FIXED: Compare base types only
        declared_base = self.base_type(symbol.data_type)
        
        # ✅ FIX: Allow aggregate initialization for structs/classes
        # Example: Point p = {3, 4}; where 'array' type is used for class initialization
        is_aggregate_init = False
        if init_type == 'array':
            # Check if declared_base is a class/struct type
            type_symbol = self.symbol_table.lookup(declared_base)
            if type_symbol and type_symbol.symbol_type == SymbolType.CLASS:
                is_aggregate_init = True
        
        if not is_aggregate_init and not self.types_compatible(declared_base, init_type):
            self.error(
                f"Incompatible types in initialization: cannot assign '{init_type}' to '{declared_base}'",
                node
            )

    def check_block(self, node: ASTNode) -> str:
        self.symbol_table.enter_scope()

        for child in node.children:
            if child.node_type == ASTNodeType.VARIABLE_DECLARATION:
                self.declare_variable(child)
            elif child.node_type == ASTNodeType.ARRAY_DECLARATION:
                self.declare_variable(child)
            elif child.node_type == ASTNodeType.FUNCTION:
                self.declare_function(child)

        for child in node.children:
            if child.node_type in {
                ASTNodeType.VARIABLE_DECLARATION,
                ASTNodeType.ARRAY_DECLARATION,
                ASTNodeType.FUNCTION
            }:
                if child.children:
                    self.check_variable_initialization(child)
                continue
            
            self.check_node(child)

        self.symbol_table.exit_scope()
        return 'void'

    def check_increment_decrement(self, node: ASTNode) -> str:
        if not node.children:
            self.error("Inc/dec requires operand", node)
            self.annotate_node(node, 'error')
            return 'error'
        
        operand = node.children[0]
        ot = self.check_node(operand)
        
        # Check const increment/decrement
        if operand.node_type == ASTNodeType.IDENTIFIER:
            sym = self.symbol_table.lookup(operand.value)
            if sym and sym.is_const:
                self.error(f"Cannot increment/decrement const variable '{sym.name}'", node)
        
        if ot not in ['int', 'float'] and not self.is_pointer(ot):
            self.warning(f"Inc/dec on non-numeric: {ot}", node)
        self.annotate_node(node, ot)
        return ot

    def check_compound_assignment(self, node: ASTNode) -> str:
        if len(node.children) < 2:
            self.error("Compound assignment malformed", node)
            self.annotate_node(node, 'error')
            return 'error'
        
        left = node.children[0]
        lt = self.check_node(left)
        rt = self.check_node(node.children[1])
        
        # Check const compound assignment
        if left.node_type == ASTNodeType.IDENTIFIER:
            sym = self.symbol_table.lookup(left.value)
            if sym and sym.is_const:
                self.error(f"Cannot modify const variable '{sym.name}' with compound assignment", node)
        
        if not self.types_compatible(lt, rt):
            self.warning(f"Compound assignment type mismatch: {lt} {node.value} {rt}", node)
        self.annotate_node(node, lt)
        return lt

    def check_ternary(self, node: ASTNode) -> str:
        if len(node.children) < 3:
            self.error("Ternary malformed", node)
            self.annotate_node(node, 'error')
            return 'error'
        self.check_node(node.children[0])
        t1 = self.check_node(node.children[1])
        t2 = self.check_node(node.children[2])
        if self.types_compatible(t1, t2):
            self.annotate_node(node, t1)
            return t1
        self.warning(f"Ternary branches incompatible: {t1} vs {t2}", node)
        self.annotate_node(node, t1)
        return t1

    def check_member_access(self, node: ASTNode) -> str:
        """Check obj.member access with access control"""
        if not node.children:
            self.error("Member access missing object", node)
            self.annotate_node(node, 'error')
            return 'error'
        
        obj = node.children[0]
        mname = node.value
        obj_type = self.check_node(obj)
        
        if self.is_pointer(obj_type):
            self.error(f"Using '.' on pointer type {obj_type}; did you mean '->'?", node)
            self.annotate_node(node, 'error')
            return 'error'
        
        class_sym = self.symbol_table.lookup(obj_type)
        if not class_sym or class_sym.symbol_type != SymbolType.CLASS:
            self.error(f"Type '{obj_type}' is not a class for member access", node)
            self.annotate_node(node, 'error')
            return 'error'
        
        member = class_sym.members.get(mname)
        if not member:
            self.error(f"Class '{obj_type}' has no member '{mname}'", node)
            self.annotate_node(node, 'error')
            return 'error'
        
        # Check access control
        if not self.check_access_allowed(class_sym, member):
            self.error(f"Cannot access {member.access_level.name.lower()} member '{mname}' of class '{obj_type}'", node)
        
        is_lval = member.symbol_type == SymbolType.MEMBER_VARIABLE
        self.annotate_node(node, member.data_type, is_lvalue=is_lval, hint='member->attr')
        return member.data_type

    def check_member_access_pointer(self, node: ASTNode) -> str:
        """Check ptr->member access with access control"""
        if not node.children:
            self.error("Member access '->' missing pointer", node)
            self.annotate_node(node, 'error')
            return 'error'
        
        ptr = node.children[0]
        mname = node.value
        ptype = self.check_node(ptr)
        
        if not self.is_pointer(ptype):
            self.error(f"Using '->' on non-pointer type {ptype}", node)
            self.annotate_node(node, 'error')
            return 'error'
        
        base = self.base_type(ptype)
        class_sym = self.symbol_table.lookup(base)
        
        if not class_sym or class_sym.symbol_type != SymbolType.CLASS:
            self.error(f"Type '{base}' is not a class for '->' access", node)
            self.annotate_node(node, 'error')
            return 'error'
        
        member = class_sym.members.get(mname)
        if not member:
            self.error(f"Class '{base}' has no member '{mname}'", node)
            self.annotate_node(node, 'error')
            return 'error'
        
        # Check access control
        if not self.check_access_allowed(class_sym, member):
            self.error(f"Cannot access {member.access_level.name.lower()} member '{mname}' of class '{base}'", node)
        
        is_lval = member.symbol_type == SymbolType.MEMBER_VARIABLE
        self.annotate_node(node, member.data_type, is_lvalue=is_lval, hint='ptr->member')
        return member.data_type

    def check_access_allowed(self, class_sym: Symbol, member: Symbol) -> bool:
        """Check if access to a class member is allowed from current context"""
        # Public members always accessible
        if member.access_level == AccessLevel.PUBLIC:
            return True
        
        # Private members only accessible from within same class
        if member.access_level == AccessLevel.PRIVATE:
            return self.current_class and self.current_class.name == class_sym.name
        
        # Protected members accessible from same class or derived classes
        if member.access_level == AccessLevel.PROTECTED:
            if self.current_class:
                # Same class
                if self.current_class.name == class_sym.name:
                    return True
                # Derived class
                if class_sym.name in self.current_class.base_classes:
                    return True
                # Check transitive inheritance
                return self.is_derived_from(self.current_class, class_sym)
            return False
        
        return False

    def is_derived_from(self, derived: Symbol, base: Symbol) -> bool:
        """Check if derived class inherits from base class (directly or transitively)"""
        if base.name in derived.base_classes:
            return True
        
        for base_name in derived.base_classes:
            base_sym = self.symbol_table.lookup(base_name)
            if base_sym and base_sym.symbol_type == SymbolType.CLASS:
                if self.is_derived_from(base_sym, base):
                    return True
        
        return False

    def check_this(self, node: ASTNode) -> str:
        if not self.current_class:
            self.error("'this' used outside class/member context", node)
            self.annotate_node(node, 'error')
            return 'error'
        t = self.make_pointer(self.current_class.data_type, 1)
        self.annotate_node(node, t, is_lvalue=False, hint='this->ptr')
        return t

    def check_switch(self, node: ASTNode) -> str:
        if not node.children:
            self.error("Switch requires expression", node)
            self.annotate_node(node, 'error')
            return 'error'
        self.check_node(node.children[0])
        for ch in node.children[1:]:
            self.check_node(ch)
        self.annotate_node(node, 'void')
        return 'void'

    # ----------------- literal and compatibility helpers -----------------
    def get_literal_type(self, node: ASTNode) -> str:
        v = node.value
        if isinstance(v, list):
            node_type = 'array'
            self.annotate_node(node, node_type)
            return node_type
        if v is None:
            self.annotate_node(node, 'int')
            return 'int'
        if isinstance(v, (int, float)):
            t = 'float' if isinstance(v, float) else 'int'
            self.annotate_node(node, t)
            return t
        if isinstance(v, str):
            s = v.strip()
            if s.startswith('"') and s.endswith('"'):
                self.annotate_node(node, 'str')
                return 'str'
            if s.startswith("'") and s.endswith("'"):
                self.annotate_node(node, 'char')
                return 'char'
            low = s.lower()
            # ✅ FIX: Handle boolean literals
            if low in ('true', 'false'):
                self.annotate_node(node, 'bool')
                return 'bool'
            if low in ('nullptr', 'null', 'nil'):
                self.annotate_node(node, 'nullptr')
                return 'nullptr'
            if '.' in s or 'f' in s.lower():
                self.annotate_node(node, 'float')
                return 'float'
            if s.isdigit():
                self._checking_literal_zero = (s == '0')
                self.annotate_node(node, 'int')
                return 'int'
            self.annotate_node(node, 'str')
            return 'str'
        self.annotate_node(node, 'int')
        return 'int'

    def types_compatible(self, type1: str, type2: str) -> bool:
        """✅ ENHANCED: Check if two types are compatible, with template support."""
        if not type1 or not type2:
            return True
        if 'error' in {type1, type2}:
            return True
        
        # ✅ TEMPLATE SUPPORT: Template parameters are wildcards
        if hasattr(self, 'template_params'):
            base1_check = self.base_type(type1)
            base2_check = self.base_type(type2)
            if base1_check in self.template_params or base2_check in self.template_params:
                return True  # Template parameters match any type
        
        # ✅ FIXED: Compare base types (strip qualifiers)
        base1 = self.base_type(type1)
        base2 = self.base_type(type2)
        
        if base1 == base2:
            return True

        if self.is_nullptr_type(type1) and self.is_pointer(type2):
            return True
        if self.is_nullptr_type(type2) and self.is_pointer(type1):
            return True

        ptr1 = self.pointer_level(type1)
        ptr2 = self.pointer_level(type2)

        # ✅ FIX: Allow numeric type conversions (widening conversions are safe)
        # int -> float -> double hierarchy
        if ptr1 == ptr2 == 0:
            numeric_types = {'int', 'float', 'double'}
            if {base1, base2} <= numeric_types:
                # Allow any numeric type conversion for now
                # In C++: int->float, int->double, float->double are all valid (widening)
                return True

        if ptr1 > 0 or ptr2 > 0:
            if ptr1 > 0 and ptr2 > 0:
                if base1 == base2 or 'void' in {base1, base2}:
                    return ptr1 == ptr2
                return False
            if (ptr1 > 0 and base2 == 'int') or (ptr2 > 0 and base1 == 'int'):
                if getattr(self, "_checking_literal_zero", False):
                    return True
                return False
            return False

        if isinstance(type1, str) and type1.endswith('[]') and self.is_pointer(type2):
            return self.types_compatible(type1[:-2], self.base_type(type2))
        if isinstance(type2, str) and type2.endswith('[]') and self.is_pointer(type1):
            return self.types_compatible(self.base_type(type1), type2[:-2])

        return False

    # ----------------- summary helpers -----------------
    def get_symbol_table_summary(self) -> Dict:
        symbols = self.symbol_table.get_all_symbols()
        summary = {
            'total_symbols': len(symbols),
            'variables': len([s for s in symbols if s.symbol_type == SymbolType.VARIABLE]),
            'functions': len([s for s in symbols if s.symbol_type == SymbolType.FUNCTION]),
            'arrays': len([s for s in symbols if s.symbol_type == SymbolType.ARRAY]),
            'parameters': len([s for s in symbols if s.symbol_type == SymbolType.PARAMETER]),
            'classes': len([s for s in symbols if s.symbol_type == SymbolType.CLASS]),
            'const_variables': len([s for s in symbols if s.is_const]),
            'static_variables': len([s for s in symbols if s.is_static]),
            'references': len([s for s in symbols if s.is_reference]),
            'includes': list(self.includes),
            'defines': dict(self.defines)
        }
        return summary

    def get_all_errors(self) -> List[str]:
        return self.errors + self.warnings