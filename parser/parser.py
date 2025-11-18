"""
Parser - COMPLETE FIXED VERSION with proper support for all enhanced features
- ✅ Class inheritance properly stored
- ✅ Access control properly tracked
- ✅ References and pointers correctly parsed
- ✅ Member variables preserve all info
- ✅ ALL METHODS INCLUDED
"""

from dataclasses import dataclass
from typing import List, Optional, Any
from enum import Enum, auto
import sys
sys.path.append('..')
from lexer.lexer import Token, TokenType

class ASTNodeType(Enum):
    PROGRAM = auto()
    FUNCTION = auto()
    VARIABLE_DECLARATION = auto()
    ASSIGNMENT = auto()
    BINARY_OP = auto()
    UNARY_OP = auto()
    LITERAL = auto()
    IDENTIFIER = auto()
    IF_STATEMENT = auto()
    WHILE_LOOP = auto()
    DO_WHILE_LOOP = auto()
    FOR_LOOP = auto()
    RETURN_STATEMENT = auto()
    FUNCTION_CALL = auto()
    ARRAY_DECLARATION = auto()
    ARRAY_ACCESS = auto()
    BLOCK = auto()
    EXPRESSION_STATEMENT = auto()
    PREPROCESSOR = auto()
    COMPOUND_ASSIGNMENT = auto()
    ADDRESS_OF = auto()
    DEREFERENCE = auto()
    INCREMENT = auto()
    DECREMENT = auto()
    BREAK_STATEMENT = auto()
    CONTINUE_STATEMENT = auto()
    SWITCH_STATEMENT = auto()
    CASE_STATEMENT = auto()
    DEFAULT_STATEMENT = auto()
    TERNARY_OP = auto()
    MULTIPLE_DECLARATION = auto()
    CLASS_DEFINITION = auto()
    MEMBER_VARIABLE = auto()
    MEMBER_FUNCTION = auto()
    MEMBER_ACCESS = auto()
    MEMBER_ACCESS_POINTER = auto()
    THIS_KEYWORD = auto()

@dataclass
class ASTNode:
    """Base class for all AST nodes"""
    node_type: ASTNodeType
    value: Any = None
    children: List['ASTNode'] = None
    data_type: str = None
    line: int = 0
    column: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def __repr__(self):
        return f"ASTNode({self.node_type.name}, value={self.value}, children={len(self.children)})"

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = [t for t in tokens if t.type != TokenType.COMMENT]
        self.position = 0
        self.current_token = self.tokens[0] if self.tokens else None
        self.ast = None
        self.errors = []
        self.max_iterations = 10000
        self.iteration_count = 0
    
    def error(self, message: str):
        """Record a parsing error"""
        if self.current_token:
            error_msg = f"Parse Error at line {self.current_token.line}, col {self.current_token.column}: {message}"
        else:
            error_msg = f"Parse Error: {message} (at end of file)"
        self.errors.append(error_msg)
    
    def advance(self):
        """Move to next token"""
        self.iteration_count += 1
        if self.iteration_count > self.max_iterations:
            raise RuntimeError("Parser exceeded maximum iterations - possible infinite loop")
        
        self.position += 1
        if self.position < len(self.tokens):
            self.current_token = self.tokens[self.position]
        else:
            self.current_token = None
    
    def peek(self, offset=1) -> Optional[Token]:
        """Look ahead at token"""
        pos = self.position + offset
        if 0 <= pos < len(self.tokens):
            return self.tokens[pos]
        return None
    
    def expect(self, token_type: TokenType) -> bool:
        """Check if current token matches expected type"""
        if self.current_token and self.current_token.type == token_type:
            self.advance()
            return True
        self.error(f"Expected {token_type.name}, got {self.current_token.type.name if self.current_token else 'EOF'}")
        return False
    
    def parse(self) -> ASTNode:
        """Main parsing entry point"""
        try:
            self.ast = self.parse_program()
            return self.ast
        except RuntimeError as e:
            self.error(str(e))
            return ASTNode(node_type=ASTNodeType.PROGRAM)
    
    def parse_program(self) -> ASTNode:
        """Parse entire program"""
        program = ASTNode(node_type=ASTNodeType.PROGRAM)
        
        while self.current_token and self.current_token.type != TokenType.EOF:
            if self.current_token.type == TokenType.PREPROCESSOR:
                node = ASTNode(
                    node_type=ASTNodeType.PREPROCESSOR,
                    value=self.current_token.value,
                    line=self.current_token.line,
                    column=self.current_token.column
                )
                program.children.append(node)
                self.advance()
                continue
            
            if self.current_token.type == TokenType.USING:
                self.skip_using_namespace()
                continue
            
            # ✅ Handle template declarations
            if self.current_token.type == TokenType.TEMPLATE:
                node = self.parse_template()
            elif self.current_token.type in {TokenType.CLASS, TokenType.STRUCT}:  # ✅ Handle both class and struct
                node = self.parse_class_definition()
            else:
                node = self.parse_declaration()
            if node:
                program.children.append(node)
            else:
                self.error(f"Skipping unparsed token: {self.current_token}")
                self.advance()
        
        return program
    
    def parse_class_definition(self) -> Optional[ASTNode]:
        """✅ ENHANCED: Parses a class or struct definition with proper inheritance support"""
        
        # ✅ Check if it's a class or struct
        is_struct = False
        if self.current_token.type == TokenType.STRUCT:
            is_struct = True
            self.advance()
        elif self.current_token.type == TokenType.CLASS:
            self.advance()
        else:
            self.error("Expected 'class' or 'struct' keyword")
            return None
        
        if not self.current_token or self.current_token.type != TokenType.IDENTIFIER:
            self.error("Expected class name after 'class' keyword")
            return None
        
        name = self.current_token.value
        line = self.current_token.line
        col = self.current_token.column
        self.advance()

        # ✅ FIXED: Initialize value as dictionary
        class_node = ASTNode(
            node_type=ASTNodeType.CLASS_DEFINITION,
            value={'name': name, 'base_classes': []},
            line=line,
            column=col
        )
        
        # ✅ FIXED: Parse inheritance
        if self.current_token and self.current_token.type == TokenType.COLON:
            self.advance()

            while self.current_token:
                access_specifier = 'public'
                
                # Handle access specifier
                if self.current_token.type == TokenType.PUBLIC:
                    access_specifier = 'public'
                    self.advance()
                elif self.current_token.type == TokenType.PRIVATE:
                    access_specifier = 'private'
                    self.advance()
                elif self.current_token.type == TokenType.PROTECTED:
                    access_specifier = 'protected'
                    self.advance()

                if self.current_token.type == TokenType.IDENTIFIER:
                    base_name = self.current_token.value
                    class_node.value['base_classes'].append({
                        'name': base_name,
                        'access': access_specifier
                    })
                    self.advance()
                else:
                    self.error("Expected base class name after ':'")
                    break

                if self.current_token.type == TokenType.COMMA:
                    self.advance()
                else:
                    break
        
        if not self.expect(TokenType.LBRACE):
            return class_node

        # ✅ Default access: public for struct, private for class
        current_access = 'public' if is_struct else 'private'
        
        while self.current_token and self.current_token.type != TokenType.RBRACE:
            # Handle constructors: Identifier equal to class name followed by '(' inside class body
            if (self.current_token.type == TokenType.IDENTIFIER and
                self.current_token.value == name and
                self.peek() and self.peek().type == TokenType.LPAREN):
                ctor_node = self.parse_constructor(name)
                if ctor_node:
                    # Mark as member function with access
                    ctor_node.node_type = ASTNodeType.MEMBER_FUNCTION
                    if isinstance(ctor_node.value, dict):
                        ctor_node.value['access'] = current_access
                    class_node.children.append(ctor_node)
                    continue
            
            # ✅ Handle destructors: ~ClassName() syntax
            if (self.current_token.type == TokenType.TILDE and
                self.peek() and self.peek().type == TokenType.IDENTIFIER and
                self.peek().value == name):
                dtor_node = self.parse_destructor(name)
                if dtor_node:
                    # Mark as member function with access
                    dtor_node.node_type = ASTNodeType.MEMBER_FUNCTION
                    if isinstance(dtor_node.value, dict):
                        dtor_node.value['access'] = current_access
                    class_node.children.append(dtor_node)
                    continue

            if self.current_token.type == TokenType.PUBLIC:
                current_access = 'public'
                self.advance()
                self.expect(TokenType.COLON)
                continue
            elif self.current_token.type == TokenType.PRIVATE:
                current_access = 'private'
                self.advance()
                self.expect(TokenType.COLON)
                continue
            elif self.current_token.type == TokenType.PROTECTED:
                current_access = 'protected'
                self.advance()
                self.expect(TokenType.COLON)
                continue

            member = self.parse_declaration()
            if member:
                if member.node_type == ASTNodeType.FUNCTION:
                    member.node_type = ASTNodeType.MEMBER_FUNCTION
                    if isinstance(member.value, dict):
                        member.value['access'] = current_access
                    elif isinstance(member.value, str):
                        # Extract function info
                        func_name = member.value
                        params = []
                        member.value = {
                            'name': func_name,
                            'params': params,
                            'access': current_access
                        }
                    else:
                        member.value = {'name': member.value, 'access': current_access}
                else:
                    wrapper = ASTNode(
                        node_type=ASTNodeType.MEMBER_VARIABLE,
                        value={'access': current_access},
                        children=[member],
                        line=member.line,
                        column=member.column
                    )
                    member = wrapper
                
                class_node.children.append(member)
            else:
                self.error("Failed to parse class member")
                self.advance()

        self.expect(TokenType.RBRACE)
        self.expect(TokenType.SEMICOLON)
        
        return class_node

    def parse_constructor(self, class_name: str) -> Optional[ASTNode]:
        """Parse a C++-style constructor inside a class and map it to a member function named '__init__'"""
        # current token is IDENTIFIER == class_name
        line = self.current_token.line
        col = self.current_token.column
        self.advance()  # consume class name
        if not self.expect(TokenType.LPAREN):
            return None

        # Parse parameter list (reuse logic from parse_function)
        params = []
        type_tokens = {TokenType.INT, TokenType.FLOAT, TokenType.DOUBLE, 
                         TokenType.CHAR, TokenType.BOOL, TokenType.VOID}

        while self.current_token and self.current_token.type != TokenType.RPAREN:
            param_qualifiers = []
            while self.current_token and self.current_token.type in {TokenType.CONST, TokenType.STATIC, TokenType.VOLATILE}:
                param_qualifiers.append(self.current_token.value)
                self.advance()

            param_type = ""
            if self.current_token.type in type_tokens:
                param_type = self.current_token.value
                self.advance()
            elif self.current_token.type == TokenType.IDENTIFIER:
                param_type = self.current_token.value
                self.advance()
            else:
                self.error(f"Expected parameter type in constructor, got {self.current_token.type.name}")
                break

            while self.current_token and self.current_token.type in {TokenType.MULTIPLY, TokenType.AMPERSAND}:
                if self.current_token.type == TokenType.MULTIPLY:
                    param_type += "*"
                elif self.current_token.type == TokenType.AMPERSAND:
                    param_type += "&"
                self.advance()

            if self.current_token and self.current_token.type == TokenType.IDENTIFIER:
                param_name = self.current_token.value
                params.append({
                    'type': param_type,
                    'name': param_name,
                    'qualifiers': param_qualifiers
                })
                self.advance()
            else:
                self.error("Expected parameter name in constructor")
                break

            if self.current_token and self.current_token.type == TokenType.COMMA:
                self.advance()
            elif self.current_token and self.current_token.type != TokenType.RPAREN:
                self.error("Expected ',' or ')' in constructor parameter list")
                break

        self.expect(TokenType.RPAREN)

        # Body
        body = None
        if self.current_token and self.current_token.type == TokenType.LBRACE:
            body = self.parse_block()

        # Create a FUNCTION-like node wrapped as MEMBER_FUNCTION later
        func_node = ASTNode(
            node_type=ASTNodeType.FUNCTION,
            value={'name': '__init__', 'params': params, 'qualifiers': []},
            data_type='void',
            line=line,
            column=col
        )
        if body:
            func_node.children.append(body)
        return func_node
    
    def parse_destructor(self, class_name: str) -> Optional[ASTNode]:
        """Parse a C++ destructor (~ClassName) and map it to __del__"""
        # current token is TILDE
        line = self.current_token.line
        col = self.current_token.column
        self.advance()  # consume ~
        
        # Expect class name
        if not self.current_token or self.current_token.type != TokenType.IDENTIFIER or self.current_token.value != class_name:
            self.error(f"Expected {class_name} after ~ in destructor")
            return None
        self.advance()  # consume class name
        
        # Expect ()
        if not self.expect(TokenType.LPAREN):
            return None
        if not self.expect(TokenType.RPAREN):
            return None
        
        # Body
        body = None
        if self.current_token and self.current_token.type == TokenType.LBRACE:
            body = self.parse_block()
        
        # Create a FUNCTION-like node for destructor (no parameters)
        func_node = ASTNode(
            node_type=ASTNodeType.FUNCTION,
            value={'name': f'~{class_name}', 'params': [], 'qualifiers': []},
            data_type='void',
            line=line,
            column=col
        )
        if body:
            func_node.children.append(body)
        return func_node
    
    def skip_using_namespace(self):
        """Skip 'using namespace std;' statements"""
        if self.current_token and self.current_token.type == TokenType.USING:
            self.advance()
        if self.current_token and self.current_token.type == TokenType.NAMESPACE:
            self.advance()
        if self.current_token and self.current_token.type in {TokenType.IDENTIFIER, TokenType.STD}:
            self.advance()
        if self.current_token and self.current_token.type == TokenType.SEMICOLON:
            self.advance()
    
    def parse_template(self) -> Optional[ASTNode]:
        """Parse template declaration: template <typename T> ..."""
        line = self.current_token.line
        col = self.current_token.column
        
        # Consume 'template'
        if not self.expect(TokenType.TEMPLATE):
            return None
        
        # Expect '<'
        if not self.expect(TokenType.LESS_THAN):
            self.error("Expected '<' after 'template'")
            return None
        
        # Parse template parameters
        template_params = []
        while self.current_token and self.current_token.type != TokenType.GREATER_THAN:
            # Expect 'typename' or 'class'
            if self.current_token.type in {TokenType.TYPENAME, TokenType.CLASS}:
                self.advance()
            else:
                self.error(f"Expected 'typename' or 'class', got {self.current_token.type.name}")
                return None
            
            # Expect type parameter name
            if self.current_token.type == TokenType.IDENTIFIER:
                template_params.append(self.current_token.value)
                self.advance()
            else:
                self.error("Expected template parameter name")
                return None
            
            # Handle comma-separated parameters
            if self.current_token.type == TokenType.COMMA:
                self.advance()
            elif self.current_token.type != TokenType.GREATER_THAN:
                self.error("Expected ',' or '>' in template parameters")
                break
        
        # Expect '>'
        if not self.expect(TokenType.GREATER_THAN):
            self.error("Expected '>' to close template parameters")
            return None
        
        # Parse the actual declaration (function or class)
        inner_decl = None
        if self.current_token.type in {TokenType.CLASS, TokenType.STRUCT}:
            inner_decl = self.parse_class_definition()
        else:
            inner_decl = self.parse_declaration()
        
        if not inner_decl:
            self.error("Failed to parse template body")
            return None
        
        # Wrap in template node
        template_node = ASTNode(
            node_type=ASTNodeType.FUNCTION,  # Keep as FUNCTION for simplicity
            value={'template_params': template_params, 'is_template': True},
            line=line,
            column=col
        )
        
        # Copy function details from inner_decl to template_node
        if inner_decl.node_type == ASTNodeType.FUNCTION:
            # Merge template info with function info
            if isinstance(inner_decl.value, dict):
                template_node.value.update(inner_decl.value)
            template_node.data_type = inner_decl.data_type
            template_node.children = inner_decl.children
        
        return template_node
    
    def parse_declaration(self) -> Optional[ASTNode]:
        """✅ ENHANCED: Parse variable or function declaration with qualifiers"""
        if not self.current_token:
            return None
        
        qualifiers = []
        while self.current_token and self.current_token.type in {TokenType.CONST, TokenType.STATIC, TokenType.VOLATILE}:
            qualifiers.append(self.current_token.value)
            self.advance()
        
        type_tokens = {TokenType.INT, TokenType.FLOAT, TokenType.DOUBLE, 
                        TokenType.CHAR, TokenType.BOOL, TokenType.VOID}
        
        data_type = ""
        line = self.current_token.line if self.current_token else 0
        col = self.current_token.column if self.current_token else 0

        if self.current_token.type == TokenType.STD:
            if self.peek() and self.peek().type == TokenType.SCOPE:
                self.advance()
                self.advance()
                
                if self.current_token and self.current_token.type == TokenType.IDENTIFIER:
                    data_type = f"std::{self.current_token.value}"
                    self.advance()
                else:
                    self.error("Expected identifier after 'std::'")
                    return None
            else:
                self.error("Invalid use of 'std', expected 'std::'")
                self.advance()
                return None
        
        elif self.current_token.type in type_tokens:
            data_type = self.current_token.value
            self.advance()

        elif self.current_token.type == TokenType.IDENTIFIER:
            data_type = self.current_token.value
            self.advance()
        
        else:
            self.error(f"Expected type declaration, got {self.current_token.type.name}")
            self.advance()
            return None
        
        if qualifiers:
            data_type = ' '.join(qualifiers) + ' ' + data_type
        
        while self.current_token and self.current_token.type in {TokenType.MULTIPLY, TokenType.AMPERSAND}:
            if self.current_token.type == TokenType.MULTIPLY:
                data_type += "*"
            elif self.current_token.type == TokenType.AMPERSAND:
                data_type += "&"
            self.advance()

        if not self.current_token or self.current_token.type != TokenType.IDENTIFIER:
            self.error("Expected identifier after type")
            return None
        
        name = self.current_token.value
        self.advance()
        
        if not self.current_token:
            return None
        
        if self.current_token.type == TokenType.LPAREN:
            # Disambiguate between function declaration/definition and direct-initialization
            # Look ahead to token after matching ')'
            depth = 0
            i = 0
            while True:
                t = self.peek(i)
                if not t:
                    break
                if t.type == TokenType.LPAREN:
                    depth += 1
                elif t.type == TokenType.RPAREN:
                    depth -= 1
                    if depth == 0:
                        # token after matching ')'
                        after = self.peek(i + 1)
                        if after and after.type == TokenType.LBRACE:
                            return self.parse_function(data_type, name, line, col, qualifiers)
                        else:
                            # Treat as variable direct-initialization: Type name(args);
                            return self.parse_direct_init_variable(data_type, name, line, col, qualifiers)
                i += 1
            # Fallback to function parsing
            return self.parse_function(data_type, name, line, col, qualifiers)
        else:
            return self.parse_variable_declaration(data_type, name, line, col, qualifiers)

    def parse_direct_init_variable(self, data_type: str, name: str, line: int, col: int, qualifiers: List[str] = None) -> ASTNode:
        """Parse direct-initialization: Type name(expr, ...);"""
        var_node = ASTNode(
            node_type=ASTNodeType.VARIABLE_DECLARATION,
            value={'name': name, 'qualifiers': qualifiers or []},
            data_type=data_type,
            line=line,
            column=col
        )

        # Current token is LPAREN
        self.expect(TokenType.LPAREN)
        args = []
        while self.current_token and self.current_token.type != TokenType.RPAREN:
            expr = self.parse_expression()
            if expr:
                args.append(expr)
            if self.current_token and self.current_token.type == TokenType.COMMA:
                self.advance()
            elif self.current_token and self.current_token.type != TokenType.RPAREN:
                self.error("Expected ',' or ')' in initializer")
                break
        self.expect(TokenType.RPAREN)

        ctor_call = ASTNode(
            node_type=ASTNodeType.FUNCTION_CALL,
            value=data_type,
            children=args
        )
        var_node.children.append(ctor_call)

        if self.current_token and self.current_token.type == TokenType.SEMICOLON:
            self.advance()

        return var_node
    
    def parse_function(self, return_type: str, name: str, line: int, col: int, qualifiers: List[str] = None) -> ASTNode:
        """✅ ENHANCED: Parse function with qualifiers"""
        func_node = ASTNode(
            node_type=ASTNodeType.FUNCTION,
            value=name,
            data_type=return_type,
            line=line,
            column=col
        )
        
        self.expect(TokenType.LPAREN)
        params = []
        
        type_tokens = {TokenType.INT, TokenType.FLOAT, TokenType.DOUBLE, 
                         TokenType.CHAR, TokenType.BOOL, TokenType.VOID}

        while self.current_token and self.current_token.type != TokenType.RPAREN:
            param_qualifiers = []
            while self.current_token and self.current_token.type in {TokenType.CONST, TokenType.STATIC, TokenType.VOLATILE}:
                param_qualifiers.append(self.current_token.value)
                self.advance()
            
            param_type = ""
            
            if self.current_token.type == TokenType.STD:
                if self.peek() and self.peek().type == TokenType.SCOPE:
                    self.advance()
                    self.advance()
                    if self.current_token.type == TokenType.IDENTIFIER:
                        param_type = f"std::{self.current_token.value}"
                        self.advance()
                    else:
                        self.error("Expected identifier after 'std::' in parameter")
                        break
                else:
                    self.error("Invalid 'std' in parameter")
                    break
            elif self.current_token.type in type_tokens:
                param_type = self.current_token.value
                self.advance()
            elif self.current_token.type == TokenType.IDENTIFIER:
                param_type = self.current_token.value
                self.advance()
            elif self.current_token.type == TokenType.VOID and self.peek() and self.peek().type == TokenType.RPAREN:
                self.advance()
                break
            else:
                self.error(f"Expected parameter type, got {self.current_token.type.name}")
                break

            while self.current_token and self.current_token.type in {TokenType.MULTIPLY, TokenType.AMPERSAND}:
                if self.current_token.type == TokenType.MULTIPLY:
                    param_type += "*"
                elif self.current_token.type == TokenType.AMPERSAND:
                    param_type += "&"
                self.advance()

            if self.current_token and self.current_token.type == TokenType.IDENTIFIER:
                param_name = self.current_token.value
                params.append({
                    'type': param_type,
                    'name': param_name,
                    'qualifiers': param_qualifiers
                })
                self.advance()
            else:
                self.error("Expected parameter name after type")
                break
            
            if self.current_token and self.current_token.type == TokenType.COMMA:
                self.advance()
            elif self.current_token and self.current_token.type != TokenType.RPAREN:
                self.error("Expected comma or ')' after parameter")
                break
        
        self.expect(TokenType.RPAREN)
        func_node.value = {
            'name': name,
            'params': params,
            'qualifiers': qualifiers or []
        }
        
        if self.current_token and self.current_token.type == TokenType.LBRACE:
            body = self.parse_block()
            func_node.children.append(body)
        
        return func_node
        
    def parse_variable_declaration(self, data_type: str, name: str, line: int, col: int, qualifiers: List[str] = None) -> ASTNode:
        """✅ ENHANCED: Parse variable declaration with qualifiers, initializers, and multiple declarations"""
        var_node = ASTNode(
            node_type=ASTNodeType.VARIABLE_DECLARATION,
            value={'name': name, 'qualifiers': qualifiers or []},  # ✅ Always use dict
            data_type=data_type,
            line=line,
            column=col
        )
        
        # Check for array declaration first
        if self.current_token and self.current_token.type == TokenType.LBRACKET:
            self.advance()
            
            if self.current_token and self.current_token.type == TokenType.INTEGER_LITERAL:
                size = self.current_token.value
                var_node.node_type = ASTNodeType.ARRAY_DECLARATION
                var_node.value = {
                    'name': name,
                    'size': size,
                    'qualifiers': qualifiers or []
                }
                self.advance()
            
            if self.current_token and self.current_token.type == TokenType.RBRACKET:
                self.advance()
        
        # Direct-initialization with parentheses: Type name(expr,...);
        if self.current_token and self.current_token.type == TokenType.LPAREN:
            # Parse constructor-arg expressions
            self.advance()
            args = []
            while self.current_token and self.current_token.type != TokenType.RPAREN:
                arg = self.parse_expression()
                if arg:
                    args.append(arg)
                if self.current_token and self.current_token.type == TokenType.COMMA:
                    self.advance()
                elif self.current_token and self.current_token.type != TokenType.RPAREN:
                    self.error("Expected ',' or ')' in initializer")
                    break
            self.expect(TokenType.RPAREN)

            # Create a FUNCTION_CALL node representing constructor call
            ctor_call = ASTNode(
                node_type=ASTNodeType.FUNCTION_CALL,
                value=data_type,
                children=args
            )
            var_node.children.append(ctor_call)

        elif self.current_token and self.current_token.type == TokenType.ASSIGN:
            self.advance()
            
            if self.current_token and self.current_token.type == TokenType.LBRACE:
                init_values = self.parse_array_initializer()
                var_node.children.append(init_values)
            else:
                init_expr = self.parse_expression()
                if init_expr:
                    var_node.children.append(init_expr)
        
        # ✅ FIX: Check for multiple declarations AFTER parsing initialization
        # Handles: int a = 10, b = 5, c = 3;
        if self.current_token and self.current_token.type == TokenType.COMMA:
            multi_node = ASTNode(
                node_type=ASTNodeType.MULTIPLE_DECLARATION,
                value=None,
                data_type=data_type,
                line=line,
                column=col
            )
            multi_node.children.append(var_node)
            
            while self.current_token and self.current_token.type == TokenType.COMMA:
                self.advance()
                
                if self.current_token and self.current_token.type == TokenType.IDENTIFIER:
                    next_name = self.current_token.value
                    next_line = self.current_token.line
                    next_col = self.current_token.column
                    self.advance()
                    
                    next_var = ASTNode(
                        node_type=ASTNodeType.VARIABLE_DECLARATION,
                        value={'name': next_name, 'qualifiers': qualifiers or []},
                        data_type=data_type,
                        line=next_line,
                        column=next_col
                    )
                    
                    # Check for initialization of this variable
                    if self.current_token and self.current_token.type == TokenType.ASSIGN:
                        self.advance()
                        if self.current_token and self.current_token.type == TokenType.LBRACE:
                            init_values = self.parse_array_initializer()
                            next_var.children.append(init_values)
                        else:
                            init_expr = self.parse_expression()
                            if init_expr:
                                next_var.children.append(init_expr)
                    
                    multi_node.children.append(next_var)
                else:
                    self.error("Expected identifier after comma in declaration")
                    break
            
            if self.current_token and self.current_token.type == TokenType.SEMICOLON:
                self.advance()
            
            return multi_node
        
        if self.current_token and self.current_token.type == TokenType.SEMICOLON:
            self.advance()
        
        return var_node
    
    def parse_array_initializer(self) -> ASTNode:
        """Parse array initializer {1, 2, 3}"""
        init_node = ASTNode(node_type=ASTNodeType.LITERAL, value=[])
        
        if self.current_token and self.current_token.type == TokenType.LBRACE:
            self.advance()
        
        while self.current_token and self.current_token.type != TokenType.RBRACE:
            expr = self.parse_expression()
            if expr:
                init_node.children.append(expr)
            
            if self.current_token and self.current_token.type == TokenType.COMMA:
                self.advance()
            elif self.current_token and self.current_token.type != TokenType.RBRACE:
                break
        
        if self.current_token and self.current_token.type == TokenType.RBRACE:
            self.advance()
        
        return init_node
    
    def parse_block(self) -> ASTNode:
        """Parse a block of statements"""
        block = ASTNode(node_type=ASTNodeType.BLOCK)
        
        if self.current_token and self.current_token.type == TokenType.LBRACE:
            self.advance()
        
        while self.current_token and self.current_token.type not in {TokenType.RBRACE, TokenType.EOF}:
            stmt = self.parse_statement()
            if stmt:
                block.children.append(stmt)
        
        if self.current_token and self.current_token.type == TokenType.RBRACE:
            self.advance()
        
        return block
    
    def parse_statement(self) -> Optional[ASTNode]:
        """✅ FIXED: Parse statements with user-defined type support"""
        if not self.current_token or self.current_token.type == TokenType.EOF:
            return None
        
        type_tokens = {TokenType.INT, TokenType.FLOAT, TokenType.DOUBLE, 
                    TokenType.CHAR, TokenType.BOOL, TokenType.CONST, 
                    TokenType.STATIC, TokenType.VOLATILE}
        
        # Check for type declarations
        if self.current_token.type in type_tokens:
            return self.parse_declaration()
        
        # ✅ NEW: Check if IDENTIFIER followed by another IDENTIFIER (user-defined type)
        # Example: Rectangle rect; or MyClass obj;
        if self.current_token.type == TokenType.IDENTIFIER:
            peek_token = self.peek()
            # Check if next token is identifier, *, &, or punctuation that suggests declaration
            if peek_token and (
                peek_token.type == TokenType.IDENTIFIER or
                peek_token.type == TokenType.MULTIPLY or
                peek_token.type == TokenType.AMPERSAND
            ):
                # This looks like: TypeName variableName or TypeName* ptr or TypeName& ref
                return self.parse_declaration()
        
        if self.current_token.type == TokenType.IF:
            return self.parse_if_statement()
        
        if self.current_token.type == TokenType.WHILE:
            return self.parse_while_loop()
        
        if self.current_token.type == TokenType.DO:
            return self.parse_do_while_loop()
        
        if self.current_token.type == TokenType.FOR:
            return self.parse_for_loop()
        
        if self.current_token.type == TokenType.SWITCH:
            return self.parse_switch_statement()
        
        if self.current_token.type == TokenType.RETURN:
            return self.parse_return_statement()
        
        if self.current_token.type == TokenType.BREAK:
            return self.parse_break_statement()
        
        if self.current_token.type == TokenType.CONTINUE:
            return self.parse_continue_statement()
        
        if self.current_token.type == TokenType.LBRACE:
            return self.parse_block()
        
        return self.parse_expression_statement()
    
    def parse_break_statement(self) -> ASTNode:
        """Parse break statement"""
        line = self.current_token.line
        col = self.current_token.column
        self.expect(TokenType.BREAK)
        
        if self.current_token and self.current_token.type == TokenType.SEMICOLON:
            self.advance()
        
        return ASTNode(node_type=ASTNodeType.BREAK_STATEMENT, line=line, column=col)
    
    def parse_continue_statement(self) -> ASTNode:
        """Parse continue statement"""
        line = self.current_token.line
        col = self.current_token.column
        self.expect(TokenType.CONTINUE)
        
        if self.current_token and self.current_token.type == TokenType.SEMICOLON:
            self.advance()
        
        return ASTNode(node_type=ASTNodeType.CONTINUE_STATEMENT, line=line, column=col)
    
    def parse_do_while_loop(self) -> ASTNode:
        """Parse do-while loop"""
        line = self.current_token.line
        col = self.current_token.column
        do_while_node = ASTNode(node_type=ASTNodeType.DO_WHILE_LOOP, line=line, column=col)
        
        self.expect(TokenType.DO)
        
        body = self.parse_statement()
        if body:
            do_while_node.children.append(body)
        
        self.expect(TokenType.WHILE)
        self.expect(TokenType.LPAREN)
        
        condition = self.parse_expression()
        if condition:
            do_while_node.children.append(condition)
        
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.SEMICOLON)
        
        return do_while_node
    
    def parse_switch_statement(self) -> ASTNode:
        """Parse switch statement"""
        line = self.current_token.line
        col = self.current_token.column
        switch_node = ASTNode(node_type=ASTNodeType.SWITCH_STATEMENT, line=line, column=col)
        
        self.expect(TokenType.SWITCH)
        self.expect(TokenType.LPAREN)
        
        expr = self.parse_expression()
        if expr:
            switch_node.children.append(expr)
        
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.LBRACE)
        
        while self.current_token and self.current_token.type != TokenType.RBRACE:
            if self.current_token.type == TokenType.CASE:
                case_node = self.parse_case_statement()
                if case_node:
                    switch_node.children.append(case_node)
            elif self.current_token.type == TokenType.DEFAULT:
                default_node = self.parse_default_statement()
                if default_node:
                    switch_node.children.append(default_node)
            else:
                self.advance()
        
        self.expect(TokenType.RBRACE)
        
        return switch_node
    
    def parse_case_statement(self) -> ASTNode:
        """Parse case statement"""
        line = self.current_token.line
        col = self.current_token.column
        case_node = ASTNode(node_type=ASTNodeType.CASE_STATEMENT, line=line, column=col)
        
        self.expect(TokenType.CASE)
        
        value = self.parse_expression()
        if value:
            case_node.value = value
        
        self.expect(TokenType.COLON)
        
        while self.current_token and self.current_token.type not in {TokenType.CASE, TokenType.DEFAULT, TokenType.RBRACE}:
            stmt = self.parse_statement()
            if stmt:
                case_node.children.append(stmt)
        
        return case_node
    
    def parse_default_statement(self) -> ASTNode:
        """Parse default statement"""
        line = self.current_token.line
        col = self.current_token.column
        default_node = ASTNode(node_type=ASTNodeType.DEFAULT_STATEMENT, line=line, column=col)
        
        self.expect(TokenType.DEFAULT)
        self.expect(TokenType.COLON)
        
        while self.current_token and self.current_token.type not in {TokenType.CASE, TokenType.RBRACE}:
            stmt = self.parse_statement()
            if stmt:
                default_node.children.append(stmt)
        
        return default_node
    
    def parse_if_statement(self) -> ASTNode:
        """Parse if-else statement"""
        line = self.current_token.line
        col = self.current_token.column
        if_node = ASTNode(node_type=ASTNodeType.IF_STATEMENT, line=line, column=col)
        
        self.expect(TokenType.IF)
        self.expect(TokenType.LPAREN)
        
        condition = self.parse_expression()
        if condition:
            if_node.children.append(condition)
        
        self.expect(TokenType.RPAREN)
        
        then_block = self.parse_statement()
        if then_block:
            if_node.children.append(then_block)
        
        if self.current_token and self.current_token.type == TokenType.ELSE:
            self.advance()
            else_block = self.parse_statement()
            if else_block:
                if_node.children.append(else_block)
        
        return if_node
    
    def parse_while_loop(self) -> ASTNode:
        """Parse while loop"""
        line = self.current_token.line
        col = self.current_token.column
        while_node = ASTNode(node_type=ASTNodeType.WHILE_LOOP, line=line, column=col)
        
        self.expect(TokenType.WHILE)
        self.expect(TokenType.LPAREN)
        
        condition = self.parse_expression()
        if condition:
            while_node.children.append(condition)
        
        self.expect(TokenType.RPAREN)
        
        body = self.parse_statement()
        if body:
            while_node.children.append(body)
        
        return while_node
    
    def parse_for_loop(self) -> ASTNode:
        """Parse for loop"""
        line = self.current_token.line
        col = self.current_token.column
        for_node = ASTNode(node_type=ASTNodeType.FOR_LOOP, line=line, column=col)
        
        self.expect(TokenType.FOR)
        self.expect(TokenType.LPAREN)
        
        if self.current_token and self.current_token.type != TokenType.SEMICOLON:
            init = self.parse_statement()
            if init:
                for_node.children.append(init)
        else:
            self.advance()
        
        if self.current_token and self.current_token.type != TokenType.SEMICOLON:
            condition = self.parse_expression()
            if condition:
                for_node.children.append(condition)
        
        if self.current_token and self.current_token.type == TokenType.SEMICOLON:
            self.advance()
        
        if self.current_token and self.current_token.type != TokenType.RPAREN:
            increment = self.parse_expression()
            if increment:
                for_node.children.append(increment)
        
        self.expect(TokenType.RPAREN)
        
        body = self.parse_statement()
        if body:
            for_node.children.append(body)
        
        return for_node
    
    def parse_return_statement(self) -> ASTNode:
        """Parse return statement"""
        line = self.current_token.line
        col = self.current_token.column
        return_node = ASTNode(node_type=ASTNodeType.RETURN_STATEMENT, line=line, column=col)
        
        self.expect(TokenType.RETURN)
        
        if self.current_token and self.current_token.type != TokenType.SEMICOLON:
            expr = self.parse_expression()
            if expr:
                return_node.children.append(expr)
        
        if self.current_token and self.current_token.type == TokenType.SEMICOLON:
            self.advance()
        
        return return_node
    
    def parse_expression_statement(self) -> ASTNode:
        """Parse expression statement"""
        expr = self.parse_expression()
        
        if self.current_token and self.current_token.type == TokenType.SEMICOLON:
            self.advance()
        
        return ASTNode(
            node_type=ASTNodeType.EXPRESSION_STATEMENT,
            children=[expr] if expr else []
        )
    
    def parse_expression(self) -> Optional[ASTNode]:
        """Parse expression (handles ternary)"""
        return self.parse_ternary()
    
    def parse_ternary(self) -> Optional[ASTNode]:
        """Parse ternary conditional operator"""
        expr = self.parse_assignment()
        
        if self.current_token and self.current_token.type == TokenType.QUESTION:
            line = self.current_token.line
            col = self.current_token.column
            self.advance()
            
            true_expr = self.parse_assignment()
            
            self.expect(TokenType.COLON)
            
            false_expr = self.parse_ternary()
            
            ternary_node = ASTNode(
                node_type=ASTNodeType.TERNARY_OP,
                line=line,
                column=col
            )
            ternary_node.children = [expr, true_expr, false_expr]
            return ternary_node
        
        return expr
    
    def parse_assignment(self) -> Optional[ASTNode]:
        """Parse assignment"""
        expr = self.parse_logical_or()
        
        if self.current_token:
            if self.current_token.type == TokenType.ASSIGN:
                line = self.current_token.line
                col = self.current_token.column
                self.advance()
                right = self.parse_assignment()
                
                return ASTNode(
                    node_type=ASTNodeType.ASSIGNMENT,
                    value='=',
                    children=[expr, right] if right else [expr],
                    line=line,
                    column=col
                )
            
            if self.current_token.type in {TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN,
                                          TokenType.MULT_ASSIGN, TokenType.DIV_ASSIGN}:
                op = self.current_token.value
                line = self.current_token.line
                col = self.current_token.column
                self.advance()
                right = self.parse_assignment()
                
                return ASTNode(
                    node_type=ASTNodeType.COMPOUND_ASSIGNMENT,
                    value=op,
                    children=[expr, right] if right else [expr],
                    line=line,
                    column=col
                )
        
        return expr
    
    def parse_logical_or(self) -> Optional[ASTNode]:
        """Parse logical OR"""
        left = self.parse_logical_and()
        
        while self.current_token and self.current_token.type == TokenType.OR:
            op = self.current_token.value
            line = self.current_token.line
            col = self.current_token.column
            self.advance()
            right = self.parse_logical_and()
            
            left = ASTNode(
                node_type=ASTNodeType.BINARY_OP,
                value=op,
                children=[left, right] if right else [left],
                line=line,
                column=col
            )
        
        return left
    
    def parse_logical_and(self) -> Optional[ASTNode]:
        """Parse logical AND"""
        left = self.parse_equality()
        
        while self.current_token and self.current_token.type == TokenType.AND:
            op = self.current_token.value
            line = self.current_token.line
            col = self.current_token.column
            self.advance()
            right = self.parse_equality()
            
            left = ASTNode(
                node_type=ASTNodeType.BINARY_OP,
                value=op,
                children=[left, right] if right else [left],
                line=line,
                column=col
            )
        
        return left
    
    def parse_equality(self) -> Optional[ASTNode]:
        """Parse equality operators"""
        left = self.parse_relational()
        
        while self.current_token and self.current_token.type in {TokenType.EQUAL, TokenType.NOT_EQUAL}:
            op = self.current_token.value
            line = self.current_token.line
            col = self.current_token.column
            self.advance()
            right = self.parse_relational()
            
            left = ASTNode(
                node_type=ASTNodeType.BINARY_OP,
                value=op,
                children=[left, right] if right else [left],
                line=line,
                column=col
            )
        
        return left
    
    def parse_relational(self) -> Optional[ASTNode]:
        """Parse relational operators"""
        left = self.parse_shift()
        
        while self.current_token and self.current_token.type in {TokenType.LESS_THAN, TokenType.GREATER_THAN,
                                                                   TokenType.LESS_EQUAL, TokenType.GREATER_EQUAL}:
            op = self.current_token.value
            line = self.current_token.line
            col = self.current_token.column
            self.advance()
            right = self.parse_shift()
            
            left = ASTNode(
                node_type=ASTNodeType.BINARY_OP,
                value=op,
                children=[left, right] if right else [left],
                line=line,
                column=col
            )
        
        return left
    
    def parse_shift(self) -> Optional[ASTNode]:
        """Parse shift operators (<< and >>)"""
        left = self.parse_additive()
        
        while self.current_token and self.current_token.type in {TokenType.SHIFT_LEFT, TokenType.SHIFT_RIGHT}:
            op = self.current_token.value
            line = self.current_token.line
            col = self.current_token.column
            self.advance()
            right = self.parse_additive()
            
            left = ASTNode(
                node_type=ASTNodeType.BINARY_OP,
                value=op,
                children=[left, right] if right else [left],
                line=line,
                column=col
            )
        
        return left
    
    def parse_additive(self) -> Optional[ASTNode]:
        """Parse addition and subtraction"""
        left = self.parse_multiplicative()
        
        while self.current_token and self.current_token.type in {TokenType.PLUS, TokenType.MINUS}:
            op = self.current_token.value
            line = self.current_token.line
            col = self.current_token.column
            self.advance()
            right = self.parse_multiplicative()
            
            left = ASTNode(
                node_type=ASTNodeType.BINARY_OP,
                value=op,
                children=[left, right] if right else [left],
                line=line,
                column=col
            )
        
        return left
    
    def parse_multiplicative(self) -> Optional[ASTNode]:
        """Parse multiplication, division, modulo"""
        left = self.parse_unary()
        
        while self.current_token and self.current_token.type in {TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO}:
            op = self.current_token.value
            line = self.current_token.line
            col = self.current_token.column
            self.advance()
            right = self.parse_unary()
            
            left = ASTNode(
                node_type=ASTNodeType.BINARY_OP,
                value=op,
                children=[left, right] if right else [left],
                line=line,
                column=col
            )
        
        return left
    
    def parse_unary(self) -> Optional[ASTNode]:
        """Parse unary operators"""
        if self.current_token:
            if self.current_token.type in {TokenType.MINUS, TokenType.PLUS, TokenType.NOT}:
                op = self.current_token.value
                line = self.current_token.line
                col = self.current_token.column
                self.advance()
                operand = self.parse_unary()
                
                return ASTNode(
                    node_type=ASTNodeType.UNARY_OP,
                    value=op,
                    children=[operand] if operand else [],
                    line=line,
                    column=col
                )
            
            if self.current_token.type == TokenType.AMPERSAND:
                op = self.current_token.value
                line = self.current_token.line
                col = self.current_token.column
                self.advance()
                operand = self.parse_unary()
                
                return ASTNode(
                    node_type=ASTNodeType.ADDRESS_OF,
                    value=op,
                    children=[operand] if operand else [],
                    line=line,
                    column=col
                )
            
            if self.current_token.type == TokenType.MULTIPLY:
                op = self.current_token.value
                line = self.current_token.line
                col = self.current_token.column
                self.advance()
                operand = self.parse_unary()
                
                return ASTNode(
                    node_type=ASTNodeType.DEREFERENCE,
                    value=op,
                    children=[operand] if operand else [],
                    line=line,
                    column=col
                )
            
            if self.current_token.type in {TokenType.INCREMENT, TokenType.DECREMENT}:
                op = self.current_token.value
                line = self.current_token.line
                col = self.current_token.column
                self.advance()
                operand = self.parse_postfix()
                
                node_type = ASTNodeType.INCREMENT if op == '++' else ASTNodeType.DECREMENT
                return ASTNode(
                    node_type=node_type,
                    value='prefix',
                    children=[operand] if operand else [],
                    line=line,
                    column=col
                )
        
        return self.parse_postfix()
    
    def parse_postfix(self) -> Optional[ASTNode]:
        """Parse postfix operators"""
        expr = self.parse_primary()
        
        while self.current_token:
            if self.current_token.type == TokenType.LBRACKET:
                line = self.current_token.line
                col = self.current_token.column
                self.advance()
                index = self.parse_expression()
                
                if self.current_token and self.current_token.type == TokenType.RBRACKET:
                    self.advance()
                
                expr = ASTNode(
                    node_type=ASTNodeType.ARRAY_ACCESS,
                    children=[expr, index],
                    line=line,
                    column=col
                )
            
            # ✅ Handle template instantiation: func<Type>(args)
            # Only treat < as template if there's a ( after the matching >
            elif self.current_token.type == TokenType.LESS_THAN and expr and expr.node_type == ASTNodeType.IDENTIFIER:
                # Peek ahead to see if this looks like a template instantiation
                # We need to find the matching > and check if there's a ( after it
                saved_pos = self.position
                saved_token = self.current_token
                
                is_template = False
                self.advance()  # consume '<'
                depth = 1
                while self.current_token and depth > 0:
                    if self.current_token.type == TokenType.LESS_THAN:
                        depth += 1
                    elif self.current_token.type == TokenType.GREATER_THAN:
                        depth -= 1
                        if depth == 0:
                            # Check if next token is (
                            self.advance()
                            if self.current_token and self.current_token.type == TokenType.LPAREN:
                                is_template = True
                            break
                    self.advance()
                
                # Restore position if not a template
                if not is_template:
                    self.position = saved_pos
                    self.current_token = saved_token
                # If it is a template, we've already skipped past the >
                # Continue to parse function call
            
            if self.current_token and self.current_token.type == TokenType.LPAREN:
                line, col = self.current_token.line, self.current_token.column
                self.advance()
                
                args = []
                while self.current_token and self.current_token.type not in {TokenType.RPAREN, TokenType.EOF}:
                    arg = self.parse_expression()
                    if arg:
                        args.append(arg)
                    
                    if self.current_token and self.current_token.type == TokenType.COMMA:
                        self.advance()
                    elif self.current_token and self.current_token.type != TokenType.RPAREN:
                        break
                
                if self.current_token and self.current_token.type == TokenType.RPAREN:
                    self.advance()
                
                # ✅ FIXED: Extract function name from MEMBER_ACCESS or IDENTIFIER
                func_name = None
                if expr:
                    if expr.node_type == ASTNodeType.IDENTIFIER:
                        func_name = expr.value
                    elif expr.node_type == ASTNodeType.MEMBER_ACCESS:
                        func_name = expr.value  # This is the member name
                    elif expr.node_type == ASTNodeType.MEMBER_ACCESS_POINTER:
                        func_name = expr.value
                
                func_call = ASTNode(
                    node_type=ASTNodeType.FUNCTION_CALL,
                    value=func_name,
                    children=[expr] + args if expr else args,  # ✅ Include object as first child
                    line=line,
                    column=col
                )
                expr = func_call
            
            elif self.current_token.type in {TokenType.INCREMENT, TokenType.DECREMENT}:
                op = self.current_token.value
                line = self.current_token.line
                col = self.current_token.column
                self.advance()
                
                node_type = ASTNodeType.INCREMENT if op == '++' else ASTNodeType.DECREMENT
                expr = ASTNode(
                    node_type=node_type,
                    value='postfix',
                    children=[expr],
                    line=line,
                    column=col
                )

            elif self.current_token.type == TokenType.DOT:
                line, col = self.current_token.line, self.current_token.column
                self.advance()
                if self.current_token.type != TokenType.IDENTIFIER:
                    self.error("Expected member name after '.'")
                    return expr
                
                member_name = self.current_token.value
                self.advance()
                
                expr = ASTNode(
                    node_type=ASTNodeType.MEMBER_ACCESS,
                    value=member_name,
                    children=[expr],
                    line=line,
                    column=col
                )
            
            elif self.current_token.type == TokenType.ARROW:
                line, col = self.current_token.line, self.current_token.column
                self.advance()
                if self.current_token.type != TokenType.IDENTIFIER:
                    self.error("Expected member name after '->'")
                    return expr

                member_name = self.current_token.value
                self.advance()
                
                expr = ASTNode(
                    node_type=ASTNodeType.MEMBER_ACCESS_POINTER,
                    value=member_name,
                    children=[expr],
                    line=line,
                    column=col
                )

            else:
                break
        
        return expr
    
    def parse_primary(self) -> Optional[ASTNode]:
        """Parse primary expressions"""
        if not self.current_token or self.current_token.type == TokenType.EOF:
            return None
        
        if self.current_token.type in {TokenType.INTEGER_LITERAL, TokenType.FLOAT_LITERAL, 
                                       TokenType.STRING_LITERAL, TokenType.CHAR_LITERAL,
                                       TokenType.BOOLEAN_LITERAL}:  # ✅ Added boolean literals
            value = self.current_token.value
            line = self.current_token.line
            col = self.current_token.column
            self.advance()
            
            return ASTNode(
                node_type=ASTNodeType.LITERAL,
                value=value,
                line=line,
                column=col
            )
        
        if self.current_token.type in {TokenType.IDENTIFIER, TokenType.COUT, TokenType.CIN, TokenType.ENDL}:
            value = self.current_token.value
            line = self.current_token.line
            col = self.current_token.column
            self.advance()
            
            return ASTNode(
                node_type=ASTNodeType.IDENTIFIER,
                value=value,
                line=line,
                column=col
            )

        if self.current_token.type == TokenType.THIS:
            line = self.current_token.line
            col = self.current_token.column
            self.advance()
            return ASTNode(
                node_type=ASTNodeType.THIS_KEYWORD,
                value='this',
                line=line,
                column=col
            )
        
        if self.current_token.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expression()
            
            if self.current_token and self.current_token.type == TokenType.RPAREN:
                self.advance()
            
            return expr
        
        self.error(f"Unexpected token: {self.current_token.type.name}")
        self.advance()
        return None
    
    def get_ast_summary(self) -> dict:
        """Get summary statistics of the AST"""
        if not self.ast:
            return {}
        
        def count_nodes(node, counts):
            if node:
                counts[node.node_type.name] = counts.get(node.node_type.name, 0) + 1
                for child in node.children:
                    count_nodes(child, counts)
        
        counts = {}
        count_nodes(self.ast, counts)
        return counts