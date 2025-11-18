"""
Code Generator - ENHANCED VERSION
Full support for semantic analyzer features:
- ✅ const/static/volatile qualifiers
- ✅ References
- ✅ Inheritance
- ✅ Access control (private members with _prefix)
- ✅ Preprocessor includes (Python imports)
"""

from typing import List, Dict, Set
import sys
sys.path.append('..')
from parser.parser import ASTNode, ASTNodeType

class DirectCodeGenerator:
    """Generates Python code directly from AST with full semantic support"""
    
    def __init__(self, ast, symbol_table=None, includes=None, defines=None):
        self.ast = ast
        self.symbol_table = symbol_table
        self.includes = includes or set()
        self.defines = defines or {}
        self.python_code = []
        self.indent_level = 0
        self.in_loop = False
        self.in_switch = False
        self.current_class = None
        self.current_function = None  # ✅ ADD: Track current function for static variables
        self.class_hierarchy = {}  # Track inheritance
        self.boxed_vars = set()    # Variables whose address is taken
        self.in_lhs = False        # Are we generating the left-hand side of an assignment
        self.type_map = {
            'int': 'int',
            'float': 'float',
            'double': 'float',
            'char': 'str',
            'bool': 'bool',
            'void': 'None',
            'string': 'str'
        }
    
    def indent(self) -> str:
        """Get current indentation"""
        return "    " * self.indent_level
    
    def emit(self, code: str):
        """Emit a line of Python code"""
        if code.strip():
            self.python_code.append(self.indent() + code)
        else:
            self.python_code.append("")
    
    def generate(self) -> str:
        """Generate Python code from AST"""
        self.python_code = []
        self.indent_level = 0
        self.boxed_vars = set()
        self._collect_boxed_vars(self.ast)
        
        # Add header with preprocessor info
        self.emit("# Generated Python code from C++")
        self.emit("")
        
        # NEW: Handle includes as Python imports
        if self.includes:
            self.generate_includes()
            self.emit("")
        
        # NEW: Handle defines as Python constants
        if self.defines:
            self.emit("# Preprocessor defines as constants")
            for macro_name, macro_value in self.defines.items():
                self.emit(f"{macro_name} = {macro_value}")
            self.emit("")
        
        if self.ast:
            self.generate_node(self.ast)
        
        return "\n".join(self.python_code)

    def _collect_boxed_vars(self, node):
        """Prepass: collect variables whose address is taken (&var)"""
        if not node:
            return
        from parser.parser import ASTNodeType
        if node.node_type == ASTNodeType.ADDRESS_OF:
            if node.children and node.children[0].node_type == ASTNodeType.IDENTIFIER:
                self.boxed_vars.add(node.children[0].value)
        for ch in (node.children or []):
            self._collect_boxed_vars(ch)
    
    def generate_includes(self):
        """Convert C++ includes to Python imports"""
        include_map = {
            'iostream': None,  # Built-in print/input
            'string': None,    # Built-in str
            'vector': None,    # Built-in list
            'cmath': 'import math',
            'algorithm': 'import itertools',
            'fstream': 'import io',
        }
        
        for include in sorted(self.includes):
            if include in include_map and include_map[include]:
                self.emit(include_map[include])
            elif include not in include_map:
                # Custom header - comment it out
                self.emit(f"# include: {include}")
    
    def generate_node(self, node):
        """Generate code for a node"""
        if not node:
            return ""
        
        if node.node_type == ASTNodeType.PROGRAM:
            for child in node.children:
                if child.node_type != ASTNodeType.PREPROCESSOR:
                    self.generate_node(child)
        
        elif node.node_type == ASTNodeType.CLASS_DEFINITION:
            self.generate_class(node)
        
        elif node.node_type == ASTNodeType.FUNCTION:
            self.generate_function(node)
        
        elif node.node_type == ASTNodeType.VARIABLE_DECLARATION:
            return self.generate_variable_declaration(node)
        
        elif node.node_type == ASTNodeType.MULTIPLE_DECLARATION:
            return self.generate_multiple_declaration(node)
        
        elif node.node_type == ASTNodeType.ARRAY_DECLARATION:
            return self.generate_array_declaration(node)
        
        elif node.node_type == ASTNodeType.ASSIGNMENT:
            return self.generate_assignment(node)
        
        elif node.node_type == ASTNodeType.BINARY_OP:
            return self.generate_binary_op(node)
        
        elif node.node_type == ASTNodeType.UNARY_OP:
            return self.generate_unary_op(node)
        
        elif node.node_type == ASTNodeType.TERNARY_OP:
            return self.generate_ternary_op(node)
        
        elif node.node_type == ASTNodeType.LITERAL:
            return self.generate_literal(node)
        
        elif node.node_type == ASTNodeType.IDENTIFIER:
            var_name = node.value

            if self.current_class and self.indent_level > 1:  # Inside a method
                # Check if this identifier is a class member
                if hasattr(self.current_class, 'members') and var_name in self.current_class.members:
                    member_symbol = self.current_class.members[var_name]
                    # Only use self. for non-static members
                    if not member_symbol.is_static:
                        # Apply access control
                        if member_symbol.access_level.name == 'PRIVATE':
                            return f"self._{var_name}"
                        return f"self.{var_name}"
            
            # For boxed local variables, use value access unless we're producing an address or similar
            if var_name in self.boxed_vars:
                # In LHS, we want to target the boxed value
                if self.in_lhs:
                    return f"{var_name}[0]"
                return f"{var_name}[0]"
            return var_name
        
        elif node.node_type == ASTNodeType.IF_STATEMENT:
            self.generate_if_statement(node)
        
        elif node.node_type == ASTNodeType.WHILE_LOOP:
            self.generate_while_loop(node)
        
        elif node.node_type == ASTNodeType.DO_WHILE_LOOP:
            self.generate_do_while_loop(node)
        
        elif node.node_type == ASTNodeType.FOR_LOOP:
            self.generate_for_loop(node)
        
        elif node.node_type == ASTNodeType.SWITCH_STATEMENT:
            self.generate_switch_statement(node)
        
        elif node.node_type == ASTNodeType.RETURN_STATEMENT:
            self.generate_return_statement(node)
        
        elif node.node_type == ASTNodeType.BREAK_STATEMENT:
            self.generate_break_statement(node)
        
        elif node.node_type == ASTNodeType.CONTINUE_STATEMENT:
            self.generate_continue_statement(node)
        
        elif node.node_type == ASTNodeType.FUNCTION_CALL:
            return self.generate_function_call(node)
        
        elif node.node_type == ASTNodeType.ARRAY_ACCESS:
            return self.generate_array_access(node)
        
        elif node.node_type == ASTNodeType.MEMBER_ACCESS:
            return self.generate_member_access(node)
        
        elif node.node_type == ASTNodeType.MEMBER_ACCESS_POINTER:
            return self.generate_member_access_pointer(node)
        
        elif node.node_type == ASTNodeType.THIS_KEYWORD:
            return "self"
        
        elif node.node_type == ASTNodeType.ADDRESS_OF:
            return self.generate_address_of(node)
        
        elif node.node_type == ASTNodeType.DEREFERENCE:
            return self.generate_dereference(node)
        
        elif node.node_type == ASTNodeType.BLOCK:
            self.generate_block(node)
        
        elif node.node_type == ASTNodeType.EXPRESSION_STATEMENT:
            if node.children:
                child = node.children[0]
                if child.node_type in {ASTNodeType.INCREMENT, ASTNodeType.DECREMENT}:
                    self.generate_increment_decrement(child)
                elif child.node_type in {ASTNodeType.ASSIGNMENT, ASTNodeType.COMPOUND_ASSIGNMENT}:
                    # ✅ FIXED: Assignments/compound assignments emit themselves, just generate
                    self.generate_node(child)
                else:
                    # For other expressions (function calls, etc.)
                    expr = self.generate_node(child)
                    if expr and expr.strip():
                        self.emit(expr)
        
        elif node.node_type in {ASTNodeType.INCREMENT, ASTNodeType.DECREMENT}:
            return self.generate_increment_decrement_expr(node)
        
        elif node.node_type == ASTNodeType.COMPOUND_ASSIGNMENT:
            return self.generate_compound_assignment(node)
        
        return ""
    
    def generate_class(self, node):
        """Generate class with inheritance and access control"""
        class_info = node.value
        name = class_info.get('name') if isinstance(class_info, dict) else class_info
        base_classes = class_info.get('base_classes', []) if isinstance(class_info, dict) else []
        
        # Look up class symbol
        class_symbol = self.symbol_table.lookup(name) if self.symbol_table else None
        self.current_class = class_symbol
        
        self.emit("")
        
        # NEW: Handle inheritance
        if base_classes:
            base_names = [b['name'] if isinstance(b, dict) else b for b in base_classes]
            bases_str = ", ".join(base_names)
            self.emit(f"class {name}({bases_str}):")
            self.class_hierarchy[name] = base_classes
        else:
            self.emit(f"class {name}:")
        
        self.indent_level += 1
        
        # Collect members and methods
        has_members = False
        member_vars = []
        member_funcs = []
        
        for member in node.children:
            if member.node_type == ASTNodeType.MEMBER_VARIABLE:
                member_vars.append(member)
                has_members = True
            elif member.node_type == ASTNodeType.MEMBER_FUNCTION:
                member_funcs.append(member)
                has_members = True
        
        # Detect user-defined constructor (__init__) to avoid emitting default __init__
        has_user_init = any((isinstance(m.value, dict) and m.value.get('name') == '__init__') or (isinstance(m.value, str) and m.value == '__init__') for m in member_funcs)

        # Generate __init__ if there are member variables and no user-defined constructor
        if member_vars and not has_user_init:
            self.emit("def __init__(self):")
            self.indent_level += 1
            
            # NEW: Initialize base classes
            if base_classes:
                for base in base_classes:
                    self.emit(f"super({name}, self).__init__()")
                    break  # Python only supports single super() call
            
            for member in member_vars:
                self.generate_member_variable(member)
            
            self.indent_level -= 1
            self.emit("")
        
        # Generate member functions
        for member in member_funcs:
            self.generate_member_function(member)
        
        if not has_members:
            self.emit("pass")
        
        self.indent_level -= 1
        self.current_class = None
    
    def generate_member_variable(self, node):
        """Generate member variable initialization with access control"""
        if not node.children:
            return
        
        decl_node = node.children[0]

        # If member is a multiple declaration (e.g., int width, height;)
        if decl_node.node_type == ASTNodeType.MULTIPLE_DECLARATION:
            for sub in decl_node.children:
                if sub.node_type == ASTNodeType.VARIABLE_DECLARATION:
                    self._emit_single_member_init(sub)
            return

        # Single variable declaration
        self._emit_single_member_init(decl_node)

    def _emit_single_member_init(self, decl_node):
        """Emit initialization for a single member variable declaration node"""
        # Extract name from dict or string
        if isinstance(decl_node.value, dict):
            var_name = decl_node.value.get('name', 'unknown')
        else:
            var_name = decl_node.value if hasattr(decl_node, 'value') else 'unknown'
        var_type = decl_node.data_type if hasattr(decl_node, 'data_type') else 'int'

        # Look up member symbol for qualifiers and access
        member_symbol = None
        if self.current_class and hasattr(self.current_class, 'members') and var_name in self.current_class.members:
            member_symbol = self.current_class.members[var_name]

        # Apply access control - prefix private members with _
        display_name = f"_{var_name}" if (member_symbol and member_symbol.access_level.name == 'PRIVATE') else var_name

        # Handle initialization
        if decl_node.children:
            init_value = self.generate_node(decl_node.children[0])
            if member_symbol and member_symbol.is_const:
                self.emit(f"self.{display_name} = {init_value}  # const")
            else:
                self.emit(f"self.{display_name} = {init_value}")
        else:
            default = self.get_default_value(var_type)
            if member_symbol and member_symbol.is_static:
                self.emit(f"self.{display_name} = {default}  # static")
            else:
                self.emit(f"self.{display_name} = {default}")
    
    def generate_member_function(self, node):
        """Generate member function"""
        if not node.children:
            return
        
        # ✅ FIXED: Extract function info from MEMBER_FUNCTION node itself
        func_info = node.value
        name = func_info.get('name') if isinstance(func_info, dict) else func_info
        params = func_info.get('params', []) if isinstance(func_info, dict) else []
        
        # ✅ NEW: Detect destructor (starts with ~) and map to __del__
        is_destructor = False
        if isinstance(name, str) and name.startswith('~'):
            name = '__del__'
            is_destructor = True
            params = []  # Destructors don't take parameters
        
        # NEW: Look up member symbol for access control
        member_symbol = None
        if self.current_class and name in self.current_class.members:
            member_symbol = self.current_class.members[name]
        
        # NEW: Apply access control (but not for __init__ or __del__)
        if member_symbol and member_symbol.access_level.name == 'PRIVATE' and name not in ['__init__', '__del__']:
            name = f"_{name}"
        
        # Build parameter list with 'self'
        param_names = ['self'] + [p['name'] for p in params]
        param_str = ", ".join(param_names)
        
        self.emit(f"def {name}({param_str}):")
        self.indent_level += 1
        
        # ✅ Add warning comment for destructors
        if is_destructor:
            self.emit("# Note: __del__ is called by Python's garbage collector (non-deterministic)")
            self.emit("# Unlike C++ destructors which are called when object goes out of scope")
        
        # ✅ FIXED: Generate the body (first child is either FUNCTION or BLOCK)
        has_body = False
        if node.children:
            child = node.children[0]
            # Skip FUNCTION wrapper if present, go straight to body
            if child.node_type == ASTNodeType.FUNCTION:
                for func_child in child.children:
                    self.generate_node(func_child)
                    has_body = True
            elif child.node_type == ASTNodeType.BLOCK:
                # Direct BLOCK
                self.generate_node(child)
                has_body = True
        
        if not has_body:
            self.emit("pass")
        
        self.indent_level -= 1
        self.emit("")
    
    def generate_function(self, node):
        """Generate function definition"""
        func_info = node.value
        name = func_info['name'] if isinstance(func_info, dict) else func_info
        params = func_info.get('params', []) if isinstance(func_info, dict) else []
        
        # ✅ Track current function for static variables
        prev_function = self.current_function
        self.current_function = name
        
        self.emit("")
        
        if name == 'main':
            self.emit("def main():")
        else:
            # NEW: Handle reference parameters (pass by reference simulation)
            param_list = []
            ref_params = []
            for p in params:
                param_type = p.get('type', 'int')
                param_name = p['name']
                param_list.append(param_name)
                # Track reference parameters
                if '&' in param_type and '*' not in param_type:
                    ref_params.append(param_name)
            
            # Store reference parameter info for this function
            if not hasattr(self, 'function_ref_params'):
                self.function_ref_params = {}
            self.function_ref_params[name] = ref_params
            
            param_str = ", ".join(param_list)
            # Add reference comment after function signature if any params are references
            if ref_params:
                ref_comment = f"  # {', '.join(ref_params)} are references"
                self.emit(f"def {name}({param_str}):{ref_comment}")
            else:
                self.emit(f"def {name}({param_str}):")
        
        self.indent_level += 1
        
        # ✅ Add reference parameters to boxed_vars so they use [0] indexing
        prev_boxed = set(self.boxed_vars)
        if name != 'main' and ref_params:
            for ref_param in ref_params:
                self.boxed_vars.add(ref_param)
        
        for child in node.children:
            self.generate_node(child)
        
        # ✅ Restore boxed_vars state
        self.boxed_vars = prev_boxed
        
        self.indent_level -= 1
        
        # ✅ Restore previous function
        self.current_function = prev_function
        
        # Add main call
        if name == 'main':
            self.emit("")
            self.emit("if __name__ == '__main__':")
            self.indent_level += 1
            self.emit("main()")
            self.indent_level -= 1
    
    def generate_variable_declaration(self, node):
        """Generate variable declaration with qualifier support"""
        var_name = node.value['name'] if isinstance(node.value, dict) else node.value
        
        # Look up symbol for qualifiers
        symbol = self.symbol_table.lookup(var_name) if self.symbol_table else None
        
        # ✅ Handle references - they bind to the target (must share same list)
        if symbol and symbol.is_reference:
            if node.children:
                target_node = node.children[0]
                # For references, we need to ensure both target and reference use the same list
                if target_node.node_type == ASTNodeType.IDENTIFIER:
                    target_name = target_node.value
                    
                    # If target is not already boxed, wrap it in a list
                    if target_name not in self.boxed_vars:
                        self.emit(f"{target_name} = [{target_name}]  # wrap for reference")
                        self.boxed_vars.add(target_name)
                    
                    # Reference points to the same list
                    self.emit(f"{var_name} = {target_name}  # reference binding")
                    self.boxed_vars.add(var_name)
                else:
                    # Complex expression - just bind normally
                    target = self.generate_node(target_node)
                    self.emit(f"{var_name} = {target}  # reference binding")
            else:
                self.emit(f"# ERROR: reference {var_name} must be initialized")
            return var_name
        
        if node.children:
            # ✅ FIX: Handle aggregate initialization for structs/classes
            # Example: Point p = {3, 4}; should create Point() and assign members
            var_type = node.data_type or 'int'
            init_node = node.children[0]
            
            # Check if initializing a class/struct type with an aggregate initializer
            is_class_type = False
            if self.symbol_table:
                # Remove qualifiers to get base type
                base_type = var_type.replace('const ', '').replace('static ', '').replace('volatile ', '').strip()
                type_symbol = self.symbol_table.lookup(base_type)
                if type_symbol and type_symbol.symbol_type.name == 'CLASS':
                    is_class_type = True
            
            # If it's aggregate initialization of a class/struct: Type var = {val1, val2, ...}
            if is_class_type and init_node.node_type == ASTNodeType.LITERAL and isinstance(init_node.value, list):
                # Generate: var = ClassName()
                base_type = var_type.replace('const ', '').replace('static ', '').replace('volatile ', '').strip()
                self.emit(f"{var_name} = {base_type}()")
                
                # Assign member values in declaration order
                if self.symbol_table:
                    type_symbol = self.symbol_table.lookup(base_type)
                    if type_symbol and hasattr(type_symbol, 'members'):
                        # Get member variables (not functions) in order
                        member_vars = [(name, sym) for name, sym in type_symbol.members.items() 
                                      if sym.symbol_type.name in ['VARIABLE', 'MEMBER_VARIABLE']]
                        
                        # Assign values to members in order
                        for i, (member_name, member_sym) in enumerate(member_vars):
                            if i < len(init_node.children):
                                value = self.generate_node(init_node.children[i])
                                # Use actual Python name (with _ prefix if private)
                                python_name = f"_{member_name}" if member_sym.access_level.name == 'PRIVATE' else member_name
                                self.emit(f"{var_name}.{python_name} = {value}")
                return var_name
            
            init_value = self.generate_node(init_node)
            
            # ✅ FIXED: Handle static variables properly using function attributes
            if symbol and symbol.is_static:
                # Static local variable - use function attribute to persist value
                func_name = self.current_function if self.current_function else 'main'
                attr_name = f"{func_name}.{var_name}"
                
                # Initialize only once (on first call)
                self.emit(f"if not hasattr({func_name}, '{var_name}'):")
                self.indent_level += 1
                if var_name in self.boxed_vars:
                    self.emit(f"{func_name}.{var_name} = [{init_value}]")
                else:
                    self.emit(f"{func_name}.{var_name} = {init_value}")
                self.indent_level -= 1
                
                # Use the static variable
                if var_name in self.boxed_vars:
                    self.emit(f"{var_name} = {func_name}.{var_name}")
                else:
                    self.emit(f"{var_name} = {func_name}.{var_name}")
                return var_name
            
            # ✅ Regular variables with qualifier checks
            if symbol:
                comment = ""
                if symbol.is_const:
                    comment = "  # const"
                elif symbol.is_volatile:
                    comment = "  # volatile"
                if var_name in self.boxed_vars:
                    self.emit(f"{var_name} = [{init_value}]{comment}")
                else:
                    self.emit(f"{var_name} = {init_value}{comment}")
            else:
                if var_name in self.boxed_vars:
                    self.emit(f"{var_name} = [{init_value}]")
                else:
                    self.emit(f"{var_name} = {init_value}")
        else:
            var_type = node.data_type or 'int'
    
            # Check if var_type is a class
            is_class = False
            if self.symbol_table:
                type_symbol = self.symbol_table.lookup(var_type)
                if type_symbol and type_symbol.symbol_type.name == 'CLASS':
                    is_class = True
            
            if is_class:
                # Instantiate the class
                self.emit(f"{var_name} = {var_type}()")
            else:
                # Use default primitive value
                default = self.get_default_value(var_type)
                if var_name in self.boxed_vars:
                    self.emit(f"{var_name} = [{default}]")
                else:
                    self.emit(f"{var_name} = {default}")
                
        return var_name
        
    def generate_multiple_declaration(self, node):
        """Generate multiple declarations: int x, y, z;"""
        for var_node in node.children:
            if var_node.node_type == ASTNodeType.VARIABLE_DECLARATION:
                self.generate_variable_declaration(var_node)
        return ""
    
    def generate_array_declaration(self, node):
        """Generate array declaration"""
        # Value is always a dict now
        array_info = node.value if isinstance(node.value, dict) else {'name': node.value, 'size': 0}
        name = array_info.get('name', 'unknown')
        size = array_info.get('size', 0) if isinstance(array_info, dict) else 0
        
        if node.children:
            init_values = []
            for elem in node.children[0].children:
                init_values.append(self.generate_node(elem))
            init_str = ", ".join(init_values)
            self.emit(f"{name} = [{init_str}]")
        else:
            default = self.get_default_value(node.data_type)
            self.emit(f"{name} = [{default}] * {size}")
        
        return name
    
    def generate_assignment(self, node):
        """Generate assignment"""
        if len(node.children) < 2:
            return ""
        
        prev = self.in_lhs
        self.in_lhs = True
        left = self.generate_node(node.children[0])
        self.in_lhs = prev
        right = self.generate_node(node.children[1])
        
        # ✅ FIX: Check if assigning to a static variable
        if node.children[0].node_type == ASTNodeType.IDENTIFIER and self.symbol_table:
            var_name = node.children[0].value
            symbol = self.symbol_table.lookup(var_name)
            if symbol and symbol.is_static:
                # Update both local alias and function attribute
                func_name = self.current_function if self.current_function else 'main'
                self.emit(f"{left} = {right}")
                self.emit(f"{func_name}.{var_name} = {right}")
                return f"{left} = {right}"
        
        self.emit(f"{left} = {right}")
        return f"{left} = {right}"
    
    def generate_binary_op(self, node):
        """Generate binary operation"""
        if len(node.children) < 2:
            return ""
        
        op = node.value
        
        # ✅ Handle cout << chain (output stream)
        if op == '<<':
            left_node = node.children[0]
            right_node = node.children[1]
            
            # Check if left side is cout or another << operation
            if left_node.node_type == ASTNodeType.IDENTIFIER and left_node.value == 'cout':
                # First cout in chain
                right = self.generate_node(right_node)
                if right == 'endl':
                    self.emit("print()")
                else:
                    self.emit(f"print({right}, end='')")
                return ""  # ✅ FIXED: Return empty string, don't return "cout"
            
            elif left_node.node_type == ASTNodeType.BINARY_OP and left_node.value == '<<':
                # Chained cout operation
                self.generate_node(left_node)  # Process the left chain
                right = self.generate_node(right_node)
                if right == 'endl':
                    self.emit("print()")
                else:
                    self.emit(f"print({right}, end='')")
                return ""  # ✅ FIXED: Return empty string
        
        # ✅ Handle cin >> chain (input stream)
        if op == '>>':
            left_node = node.children[0]
            right_node = node.children[1]
            
            if left_node.node_type == ASTNodeType.IDENTIFIER and left_node.value == 'cin':
                # First cin in chain
                right = self.generate_node(right_node)
                self.emit(f"{right} = int(input())")
                return ""  # ✅ FIXED: Return empty string
            
            elif left_node.node_type == ASTNodeType.BINARY_OP and left_node.value == '>>':
                # Chained cin operation
                self.generate_node(left_node)  # Process the left chain
                right = self.generate_node(right_node)
                self.emit(f"{right} = int(input())")
                return ""  # ✅ FIXED: Return empty string
        
        # Regular binary operations
        left = self.generate_node(node.children[0])
        right = self.generate_node(node.children[1])
        
        # Convert C++ operators to Python
        op_map = {
            '&&': 'and',
            '||': 'or',
        }
        
        py_op = op_map.get(op, op)
        
        # Handle integer division
        if op == '/':
            return f"int({left} / {right})"
        
        return f"{left} {py_op} {right}"
    
    def generate_unary_op(self, node):
        """Generate unary operation"""
        if not node.children:
            return ""
        
        operand = self.generate_node(node.children[0])
        op = node.value
        
        if op == '!':
            return f"not {operand}"
        
        return f"{op}{operand}"
    
    def generate_ternary_op(self, node):
        """Generate ternary conditional operator"""
        if len(node.children) < 3:
            return ""
        
        condition = self.generate_node(node.children[0])
        true_expr = self.generate_node(node.children[1])
        false_expr = self.generate_node(node.children[2])
        
        return f"{true_expr} if {condition} else {false_expr}"
    
    def generate_literal(self, node):
        """Generate literal"""
        value = node.value
        
        if isinstance(value, list):
            values = [self.generate_node(child) for child in node.children]
            return f"[{', '.join(values)}]"
        
        if isinstance(value, str) and value.endswith('f'):
            return value[:-1]
        
        # ✅ FIX: Convert C++ boolean literals to Python
        if isinstance(value, str):
            if value.lower() == 'true':
                return 'True'
            elif value.lower() == 'false':
                return 'False'
        
        return str(value)
    
    def generate_member_access(self, node):
        """Generate member access: obj.member"""
        if not node.children:
            return ""
        
        obj = self.generate_node(node.children[0])
        member = node.value
        
        # NEW: Check if member is private (needs _ prefix)
        if self.symbol_table:
            # Try to find the object's class
            obj_node = node.children[0]
            if hasattr(obj_node, 'data_type'):
                class_symbol = self.symbol_table.lookup(obj_node.data_type)
                if class_symbol and member in class_symbol.members:
                    member_symbol = class_symbol.members[member]
                    if member_symbol.access_level.name == 'PRIVATE':
                        member = f"_{member}"
        
        return f"{obj}.{member}"
    
    def generate_member_access_pointer(self, node):
        """Generate pointer member access: ptr->member"""
        if not node.children:
            return ""
        
        ptr = self.generate_node(node.children[0])
        member = node.value
        
        # NEW: Check if member is private
        if self.symbol_table:
            ptr_node = node.children[0]
            if hasattr(ptr_node, 'data_type'):
                # Remove pointer/reference markers
                base_type = ptr_node.data_type.replace('*', '').replace('&', '').strip()
                class_symbol = self.symbol_table.lookup(base_type)
                if class_symbol and member in class_symbol.members:
                    member_symbol = class_symbol.members[member]
                    if member_symbol.access_level.name == 'PRIVATE':
                        member = f"_{member}"
        
        # In Python, -> becomes .
        return f"{ptr}.{member}"
    
    def generate_address_of(self, node):
        """Generate address-of operator (simulated with list wrapper)"""
        if not node.children:
            return ""
        
        operand_node = node.children[0]
        # For identifiers that we box, &x returns the same box variable name
        if operand_node.node_type == ASTNodeType.IDENTIFIER:
            name = operand_node.value
            return name
        # Fallback: evaluate expression (non-lvalue), but we cannot alias — return as-is
        return self.generate_node(operand_node)
    
    def generate_dereference(self, node):
        """Generate dereference operator - smart pointer arithmetic conversion"""
        if not node.children:
            return ""
        
        child = node.children[0]
        
        # ✅ OPTIMIZATION: Detect *(ptr + offset) pattern and convert to ptr[offset]
        if child.node_type == ASTNodeType.BINARY_OP and child.value in ['+', '-']:
            if len(child.children) == 2:
                left = child.children[0]
                right = child.children[1]
                
                # Check if it's ptr + offset or ptr - offset
                if left.node_type == ASTNodeType.IDENTIFIER:
                    ptr_name = self.generate_node(left)
                    offset = self.generate_node(right)
                    
                    if child.value == '+':
                        # *(ptr + n) → ptr[n]
                        return f"{ptr_name}[{offset}]"
                    else:
                        # *(ptr - n) → ptr[-n]
                        return f"{ptr_name}[-{offset}]"
        
        # ✅ FIX: For complex expressions, add parentheses to ensure correct precedence
        ptr = self.generate_node(child)
        
        # Check if we need parentheses (if expression contains operators)
        if any(op in ptr for op in ['+', '-', '*', '/', '%', ' and ', ' or ']):
            return f"({ptr})[0]"
        else:
            # Simple identifier, no parentheses needed
            return f"{ptr}[0]"
    
    def generate_if_statement(self, node):
        """Generate if statement"""
        if not node.children:
            return
        
        condition = self.generate_node(node.children[0])
        self.emit(f"if {condition}:")
        
        self.indent_level += 1
        
        if len(node.children) > 1:
            self.generate_node(node.children[1])
        
        self.indent_level -= 1
        
        if len(node.children) > 2:
            self.emit("else:")
            self.indent_level += 1
            self.generate_node(node.children[2])
            self.indent_level -= 1
    
    def generate_while_loop(self, node):
        """Generate while loop"""
        if len(node.children) < 2:
            return
        
        condition = self.generate_node(node.children[0])
        self.emit(f"while {condition}:")
        
        self.indent_level += 1
        old_in_loop = self.in_loop
        self.in_loop = True
        self.generate_node(node.children[1])
        self.in_loop = old_in_loop
        self.indent_level -= 1
    
    def generate_do_while_loop(self, node):
        """Generate do-while loop"""
        if len(node.children) < 2:
            return
        
        condition = self.generate_node(node.children[1])
        
        self.emit("_do_while_flag = True")
        self.emit("while _do_while_flag:")
        
        self.indent_level += 1
        old_in_loop = self.in_loop
        self.in_loop = True
        
        self.generate_node(node.children[0])
        self.emit(f"_do_while_flag = {condition}")
        
        self.in_loop = old_in_loop
        self.indent_level -= 1
    
    def generate_for_loop(self, node):
        """Generate for loop - try to use Pythonic range() where possible"""
        child_idx = 0
        
        # Try to detect pattern: for (int i = start; i < end; i++)
        # Convert to: for i in range(start, end):
        pythonic_for = False
        loop_var = None
        start_val = None
        end_val = None
        increment = 1
        comparison_op = None
        
        # Check initialization
        if child_idx < len(node.children):
            init_node = node.children[child_idx]
            if init_node.node_type.name == 'VARIABLE_DECLARATION' and init_node.children:
                loop_var = init_node.value.get('name') if isinstance(init_node.value, dict) else init_node.value
                start_val = self.generate_node(init_node.children[0])
                child_idx += 1
        
        # Check condition
        if child_idx < len(node.children):
            cond_node = node.children[child_idx]
            if cond_node.node_type.name == 'BINARY_OP' and cond_node.value in ['<', '<=', '>', '>=']:
                comparison_op = cond_node.value
                if cond_node.children and len(cond_node.children) == 2:
                    left = cond_node.children[0]
                    right = cond_node.children[1]
                    # Check if left is loop variable
                    if left.node_type.name == 'IDENTIFIER' and left.value == loop_var:
                        end_val = self.generate_node(right)
                        child_idx += 1
        
        # Check increment
        inc_node = None
        if len(node.children) > child_idx + 1:
            inc_node = node.children[-2]
            if inc_node.node_type.name in ['INCREMENT', 'DECREMENT']:
                if hasattr(inc_node, 'children') and inc_node.children:
                    inc_var = inc_node.children[0].value
                    if inc_var == loop_var:
                        increment = 1 if inc_node.node_type.name == 'INCREMENT' else -1
                        pythonic_for = True
            elif inc_node.node_type.name == 'COMPOUND_ASSIGNMENT':
                # Check for i += 1 or i -= 1
                if inc_node.children and len(inc_node.children) >= 1:
                    inc_var = inc_node.children[0].value
                    if inc_var == loop_var and inc_node.value in ['+=', '-=']:
                        if len(inc_node.children) > 1:
                            inc_val = self.generate_node(inc_node.children[1])
                            if inc_val == '1':
                                increment = 1 if inc_node.value == '+=' else -1
                                pythonic_for = True
        
        # Generate Pythonic for loop if pattern matches
        if pythonic_for and loop_var and start_val is not None and end_val is not None:
            # Adjust end value based on comparison operator
            if comparison_op == '<':
                range_end = end_val
            elif comparison_op == '<=':
                range_end = f"{end_val} + 1"
            elif comparison_op == '>':
                # For i > end_val with decrement: range(start_val, end_val, -1)
                # e.g., i > 0 means stop at 1, so range(10, 0, -1) gives 10,9,8...1
                range_end = end_val
                # increment is already -1 from decrement detection, don't negate
            elif comparison_op == '>=':
                # For i >= end_val with decrement: range(start_val, end_val-1, -1)
                # e.g., i >= 0 means stop at 0, so range(10, -1, -1) gives 10,9,8...0
                range_end = f"{end_val} - 1"
                # increment is already -1 from decrement detection, don't negate
            else:
                pythonic_for = False
            
            if pythonic_for:
                if increment == 1:
                    self.emit(f"for {loop_var} in range({start_val}, {range_end}):")
                elif increment == -1:
                    self.emit(f"for {loop_var} in range({start_val}, {range_end}, -1):")
                else:
                    self.emit(f"for {loop_var} in range({start_val}, {range_end}, {increment}):")
                
                self.indent_level += 1
                old_in_loop = self.in_loop
                self.in_loop = True
                
                # Generate body
                if node.children:
                    body_node = node.children[-1]
                    self.generate_node(body_node)
                
                self.in_loop = old_in_loop
                self.indent_level -= 1
                return
        
        # Fall back to while loop for complex for loops
        child_idx = 0
        
        # Initialization
        if child_idx < len(node.children):
            init_node = node.children[child_idx]
            if init_node.node_type.name in ['VARIABLE_DECLARATION', 'ASSIGNMENT', 'EXPRESSION_STATEMENT']:
                self.generate_node(init_node)
                child_idx += 1
        
        # Condition
        condition = "True"
        if child_idx < len(node.children):
            cond_node = node.children[child_idx]
            if cond_node.node_type.name != 'BLOCK':
                condition = self.generate_node(cond_node)
                child_idx += 1
        
        self.emit(f"while {condition}:")
        self.indent_level += 1
        old_in_loop = self.in_loop
        self.in_loop = True
        
        if node.children:
            body_node = node.children[-1]
            self.generate_node(body_node)
            
            # Increment
            if len(node.children) > child_idx + 1:
                inc_node = node.children[-2]
                if inc_node.node_type in {ASTNodeType.INCREMENT, ASTNodeType.DECREMENT}:
                    self.generate_increment_decrement(inc_node)
                elif inc_node.node_type == ASTNodeType.COMPOUND_ASSIGNMENT:
                    # ✅ FIX: COMPOUND_ASSIGNMENT already emits, don't emit again
                    self.generate_node(inc_node)
                else:
                    inc_expr = self.generate_node(inc_node)
                    if inc_expr:
                        self.emit(inc_expr)
        
        self.in_loop = old_in_loop
        self.indent_level -= 1
    
    def generate_switch_statement(self, node):
        """Generate switch statement"""
        if not node.children:
            return
        
        switch_expr = self.generate_node(node.children[0])
        first_case = True
        
        for i in range(1, len(node.children)):
            case_node = node.children[i]
            
            if case_node.node_type == ASTNodeType.CASE_STATEMENT:
                case_value = self.generate_node(case_node.value)
                
                if first_case:
                    self.emit(f"if {switch_expr} == {case_value}:")
                    first_case = False
                else:
                    self.emit(f"elif {switch_expr} == {case_value}:")
                
                self.indent_level += 1
                old_in_switch = self.in_switch
                self.in_switch = True
                
                for stmt in case_node.children:
                    self.generate_node(stmt)
                
                self.in_switch = old_in_switch
                self.indent_level -= 1
            
            elif case_node.node_type == ASTNodeType.DEFAULT_STATEMENT:
                self.emit("else:")
                self.indent_level += 1
                old_in_switch = self.in_switch
                self.in_switch = True
                
                for stmt in case_node.children:
                    self.generate_node(stmt)
                
                self.in_switch = old_in_switch
                self.indent_level -= 1
    
    def generate_return_statement(self, node):
        """Generate return statement"""
        if node.children:
            value = self.generate_node(node.children[0])
            self.emit(f"return {value}")
        else:
            self.emit("return")
    
    def generate_break_statement(self, node):
        """Generate break statement"""
        # ✅ FIX: Only emit break in loops, not in switch (which becomes if/elif)
        # In Python, if/elif branches don't fall through, so no break needed
        if self.in_loop:
            self.emit("break")
        # If in switch but not in loop, skip the break (it's not valid in if/elif)
    
    def generate_continue_statement(self, node):
        """Generate continue statement"""
        if self.in_loop:
            self.emit("continue")
    
    def generate_function_call(self, node):
        """Generate function call with member function support"""
        func_name = node.value
        
        # ✅ NEW: Check if this is a member function call
        if node.children and node.children[0].node_type == ASTNodeType.MEMBER_ACCESS:
            member_access = node.children[0]
            obj_node = member_access.children[0] if member_access.children else None
            method_name = member_access.value
            
            if obj_node:
                obj = self.generate_node(obj_node)
                
                # Check for private member access
                if self.symbol_table and hasattr(obj_node, 'data_type'):
                    class_symbol = self.symbol_table.lookup(obj_node.data_type)
                    if class_symbol and method_name in class_symbol.members:
                        member_symbol = class_symbol.members[method_name]
                        if member_symbol.access_level.name == 'PRIVATE':
                            method_name = f"_{method_name}"
                
                # Generate arguments (skip first child which is MEMBER_ACCESS)
                args = [self.generate_node(arg) for arg in node.children[1:]]
                arg_str = ", ".join(args)
                
                return f"{obj}.{method_name}({arg_str})"
        
        # ✅ REGULAR FUNCTION CALL - Handle cout/cin specially
        if func_name == 'cout':
            for arg in node.children:
                if arg.node_type == ASTNodeType.IDENTIFIER and arg.value == 'endl':
                    self.emit("print()")
                else:
                    arg_str = self.generate_node(arg)
                    self.emit(f"print({arg_str}, end='')")
            return ""
        
        if func_name == 'cin':
            for arg in node.children:
                if arg.node_type == ASTNodeType.IDENTIFIER:
                    var_name = arg.value
                    self.emit(f"{var_name} = int(input())")
            return ""
        
        # ✅ FIXED: Regular function call - don't include func_name in args
        # The function name is in node.value, arguments are in node.children
        
        # ✅ Handle reference parameters: wrap before call, unwrap after
        ref_args_to_wrap = []
        if hasattr(self, 'function_ref_params') and func_name in self.function_ref_params:
            ref_params = self.function_ref_params[func_name]
            if ref_params and node.children:
                # Collect reference arguments that need wrapping
                arg_idx = 0
                for arg in node.children:
                    if arg.node_type == ASTNodeType.IDENTIFIER and arg.value != func_name:
                        if arg_idx < len(ref_params):
                            ref_args_to_wrap.append(arg.value)
                        arg_idx += 1
                
                # Emit wrapping code before the call
                for var_name in ref_args_to_wrap:
                    self.emit(f"{var_name} = [{var_name}]")
        
        # Generate arguments (without extra wrapping - already done above)
        args = []
        for arg in node.children:
            # ✅ Skip any child that looks like a function identifier
            if arg.node_type == ASTNodeType.IDENTIFIER and arg.value == func_name:
                continue  # Skip the function name if it's in children
            args.append(self.generate_node(arg))
        
        arg_str = ", ".join(args)
        result = f"{func_name}({arg_str})"
        
        # ✅ If we wrapped arguments, emit the call and unwrap after
        if ref_args_to_wrap:
            self.emit(result)
            # Unwrap reference arguments back
            for var_name in ref_args_to_wrap:
                self.emit(f"{var_name} = {var_name}[0]")
            return ""  # Already emitted
        
        return result
        
    def generate_array_access(self, node):
        """Generate array access"""
        if len(node.children) < 2:
            return ""
        
        array = self.generate_node(node.children[0])
        index = self.generate_node(node.children[1])
        
        return f"{array}[{index}]"
    
    def generate_block(self, node):
        """Generate block"""
        for child in node.children:
            self.generate_node(child)
    
    def generate_increment_decrement(self, node):
        """Generate increment/decrement as statement"""
        if not node.children:
            return ""
        
        operand = self.generate_node(node.children[0])
        
        # ✅ FIX: Check if incrementing/decrementing a static variable
        if node.children[0].node_type == ASTNodeType.IDENTIFIER and self.symbol_table:
            var_name = node.children[0].value
            symbol = self.symbol_table.lookup(var_name)
            if symbol and symbol.is_static:
                func_name = self.current_function if self.current_function else 'main'
                if node.node_type == ASTNodeType.INCREMENT:
                    self.emit(f"{operand} += 1")
                    self.emit(f"{func_name}.{var_name} += 1")
                else:
                    self.emit(f"{operand} -= 1")
                    self.emit(f"{func_name}.{var_name} -= 1")
                return ""
        
        if node.node_type == ASTNodeType.INCREMENT:
            self.emit(f"{operand} += 1")
        else:
            self.emit(f"{operand} -= 1")
        
        return ""
    
    def generate_increment_decrement_expr(self, node):
        """Generate increment/decrement as expression"""
        if not node.children:
            return ""
        
        operand = self.generate_node(node.children[0])
        
        if node.node_type == ASTNodeType.INCREMENT:
            return f"{operand} += 1"
        else:
            return f"{operand} -= 1"
    
    def generate_compound_assignment(self, node):
        """Generate compound assignment"""
        if len(node.children) < 2:
            return ""
        
        left = self.generate_node(node.children[0])
        right = self.generate_node(node.children[1])
        op = node.value
        
        # ✅ FIX: Check if compound assigning to a static variable
        if node.children[0].node_type == ASTNodeType.IDENTIFIER and self.symbol_table:
            var_name = node.children[0].value
            symbol = self.symbol_table.lookup(var_name)
            if symbol and symbol.is_static:
                func_name = self.current_function if self.current_function else 'main'
                self.emit(f"{left} {op} {right}")
                self.emit(f"{func_name}.{var_name} {op} {right}")
                return f"{left} {op} {right}"
        
        self.emit(f"{left} {op} {right}")
        return f"{left} {op} {right}"

    def extract_name(self, value):
        """Helper to extract name from value (dict or string)"""
        if isinstance(value, dict):
            return value.get('name', 'unknown')
        return value if value else 'unknown'
    
    def get_default_value(self, var_type):
        """Get default value for type"""
        if not var_type:
            return "0"
        
        defaults = {
            'int': '0',
            'float': '0.0',
            'double': '0.0',
            'char': "''",
            'bool': 'False',
            'str': "''",
            'string': "''",
        }
        
        # Handle pointer types
        if '*' in var_type:
            return 'None'
        
        return defaults.get(var_type, '0')