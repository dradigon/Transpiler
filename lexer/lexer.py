import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional

class TokenType(Enum):
    # Keywords
    INT = auto()
    FLOAT = auto()
    DOUBLE = auto()
    CHAR = auto()
    BOOL = auto()
    VOID = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    DO = auto()
    FOR = auto()
    RETURN = auto()
    INCLUDE = auto()
    USING = auto()
    NAMESPACE = auto()
    STD = auto()
    COUT = auto()
    CIN = auto()
    ENDL = auto()
    CLASS = auto()
    STRUCT = auto()  # ✅ Added for C++ structures
    PUBLIC = auto()
    PRIVATE = auto()
    PROTECTED = auto()
    BREAK = auto()
    CONTINUE = auto()
    SWITCH = auto()
    CASE = auto()
    DEFAULT = auto()
    CONST = auto()
    STATIC = auto()
    VOLATILE = auto()
    THIS = auto()
    TEMPLATE = auto()  # ✅ Added for templates
    TYPENAME = auto()  # ✅ Added for template parameters
    
    # Identifiers and literals
    IDENTIFIER = auto()
    INTEGER_LITERAL = auto()
    FLOAT_LITERAL = auto()
    STRING_LITERAL = auto()
    CHAR_LITERAL = auto()
    BOOLEAN_LITERAL = auto()  # ✅ Added for true/false
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()
    ASSIGN = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS_THAN = auto()
    GREATER_THAN = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    INCREMENT = auto()
    DECREMENT = auto()
    PLUS_ASSIGN = auto()
    MINUS_ASSIGN = auto()
    MULT_ASSIGN = auto()
    DIV_ASSIGN = auto()
    SHIFT_LEFT = auto()
    SHIFT_RIGHT = auto()
    QUESTION = auto()
    AMPERSAND = auto()
    TILDE = auto()  # ✅ Added for destructors (~ClassName)
    
    # Delimiters
    SEMICOLON = auto()
    COMMA = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COLON = auto()
    SCOPE = auto()
    DOT = auto()
    ARROW = auto()
    
    # Special
    PREPROCESSOR = auto()
    COMMENT = auto()
    WHITESPACE = auto()
    NEWLINE = auto()
    EOF = auto()
    UNKNOWN = auto()

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int
    
    def __repr__(self):
        return f"Token({self.type.name}, '{self.value}', L{self.line}:C{self.column})"

class Lexer:
    def __init__(self, source_code: str):
        self.source = source_code
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
        
        # Define keywords
        self.keywords = {
            'int': TokenType.INT,
            'float': TokenType.FLOAT,
            'double': TokenType.DOUBLE,
            'char': TokenType.CHAR,
            'bool': TokenType.BOOL,
            'void': TokenType.VOID,
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'while': TokenType.WHILE,
            'do': TokenType.DO,
            'for': TokenType.FOR,
            'return': TokenType.RETURN,
            'include': TokenType.INCLUDE,
            'using': TokenType.USING,
            'namespace': TokenType.NAMESPACE,
            'std': TokenType.STD,
            'cout': TokenType.COUT,
            'cin': TokenType.CIN,
            'endl': TokenType.ENDL,
            'class': TokenType.CLASS,
            'struct': TokenType.STRUCT,  # ✅ Added struct keyword
            'public': TokenType.PUBLIC,
            'private': TokenType.PRIVATE,
            'protected': TokenType.PROTECTED,
            'break': TokenType.BREAK,
            'continue': TokenType.CONTINUE,
            'switch': TokenType.SWITCH,
            'case': TokenType.CASE,
            'default': TokenType.DEFAULT,
            'const': TokenType.CONST,
            'static': TokenType.STATIC,
            'volatile': TokenType.VOLATILE,
            'this': TokenType.THIS,
            'template': TokenType.TEMPLATE,  # ✅ Added template support
            'typename': TokenType.TYPENAME,
            'true': TokenType.BOOLEAN_LITERAL,  # ✅ Added boolean literals
            'false': TokenType.BOOLEAN_LITERAL,
        }
        
    def current_char(self) -> Optional[str]:
        if self.position >= len(self.source):
            return None
        return self.source[self.position]
    
    def peek_char(self, offset: int = 1) -> Optional[str]:
        pos = self.position + offset
        if pos >= len(self.source):
            return None
        return self.source[pos]
    
    def advance(self):
        if self.position < len(self.source):
            if self.source[self.position] == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            self.position += 1
    
    def skip_whitespace(self):
        while self.current_char() and self.current_char() in ' \t\r':
            self.advance()
    
    def skip_comment(self):
        # Single-line comment
        if self.current_char() == '/' and self.peek_char() == '/':
            start_col = self.column
            comment = ''
            while self.current_char() and self.current_char() != '\n':
                comment += self.current_char()
                self.advance()
            return Token(TokenType.COMMENT, comment, self.line, start_col)
        
        # Multi-line comment
        if self.current_char() == '/' and self.peek_char() == '*':
            start_line = self.line
            start_col = self.column
            comment = ''
            self.advance()
            self.advance()
            comment = '/*'
            
            while self.current_char():
                if self.current_char() == '*' and self.peek_char() == '/':
                    comment += '*/'
                    self.advance()
                    self.advance()
                    break
                comment += self.current_char()
                self.advance()
            
            return Token(TokenType.COMMENT, comment, start_line, start_col)
        
        return None
    
    def read_string(self) -> Token:
        start_col = self.column
        quote = self.current_char()
        string_val = quote
        self.advance()
        
        while self.current_char() and self.current_char() != quote:
            if self.current_char() == '\\':
                string_val += self.current_char()
                self.advance()
                if self.current_char():
                    string_val += self.current_char()
                    self.advance()
            else:
                string_val += self.current_char()
                self.advance()
        
        if self.current_char() == quote:
            string_val += quote
            self.advance()
        
        token_type = TokenType.STRING_LITERAL if quote == '"' else TokenType.CHAR_LITERAL
        return Token(token_type, string_val, self.line, start_col)
    
    def read_number(self) -> Token:
        start_col = self.column
        num_str = ''
        is_float = False
        
        while self.current_char() and (self.current_char().isdigit() or self.current_char() == '.'):
            if self.current_char() == '.':
                if is_float:
                    break
                is_float = True
            num_str += self.current_char()
            self.advance()
        
        if self.current_char() and self.current_char() in 'fF':
            num_str += self.current_char()
            is_float = True
            self.advance()
        
        token_type = TokenType.FLOAT_LITERAL if is_float else TokenType.INTEGER_LITERAL
        return Token(token_type, num_str, self.line, start_col)
    
    def read_identifier(self) -> Token:
        start_col = self.column
        identifier = ''
        
        while self.current_char() and (self.current_char().isalnum() or self.current_char() == '_'):
            identifier += self.current_char()
            self.advance()
        
        token_type = self.keywords.get(identifier, TokenType.IDENTIFIER)
        return Token(token_type, identifier, self.line, start_col)
    
    def read_preprocessor(self) -> Token:
        start_col = self.column
        preprocessor = ''
        
        while self.current_char() and self.current_char() != '\n':
            preprocessor += self.current_char()
            self.advance()
        
        return Token(TokenType.PREPROCESSOR, preprocessor, self.line, start_col)
    
    def tokenize(self) -> List[Token]:
        self.tokens = []
        
        while self.current_char():
            # Skip whitespace
            if self.current_char() in ' \t\r':
                self.skip_whitespace()
                continue
            
            # Newline
            if self.current_char() == '\n':
                self.advance()
                continue
            
            # Comments
            comment = self.skip_comment()
            if comment:
                self.tokens.append(comment)
                continue
            
            # Preprocessor directives
            if self.current_char() == '#':
                self.tokens.append(self.read_preprocessor())
                continue
            
            # Strings and chars
            if self.current_char() in '"\'':
                self.tokens.append(self.read_string())
                continue
            
            # Numbers
            if self.current_char().isdigit():
                self.tokens.append(self.read_number())
                continue
            
            # Identifiers and keywords
            if self.current_char().isalpha() or self.current_char() == '_':
                self.tokens.append(self.read_identifier())
                continue
            
            # Operators and delimiters
            start_col = self.column
            char = self.current_char()
            next_char = self.peek_char()
            
            # Two-character operators
            if char == '+' and next_char == '+':
                self.tokens.append(Token(TokenType.INCREMENT, '++', self.line, start_col))
                self.advance()
                self.advance()
            elif char == '-' and next_char == '-':
                self.tokens.append(Token(TokenType.DECREMENT, '--', self.line, start_col))
                self.advance()
                self.advance()
            elif char == '+' and next_char == '=':
                self.tokens.append(Token(TokenType.PLUS_ASSIGN, '+=', self.line, start_col))
                self.advance()
                self.advance()
            elif char == '-' and next_char == '=':
                self.tokens.append(Token(TokenType.MINUS_ASSIGN, '-=', self.line, start_col))
                self.advance()
                self.advance()
            elif char == '*' and next_char == '=':
                self.tokens.append(Token(TokenType.MULT_ASSIGN, '*=', self.line, start_col))
                self.advance()
                self.advance()
            elif char == '/' and next_char == '=':
                self.tokens.append(Token(TokenType.DIV_ASSIGN, '/=', self.line, start_col))
                self.advance()
                self.advance()
            elif char == '=' and next_char == '=':
                self.tokens.append(Token(TokenType.EQUAL, '==', self.line, start_col))
                self.advance()
                self.advance()
            elif char == '!' and next_char == '=':
                self.tokens.append(Token(TokenType.NOT_EQUAL, '!=', self.line, start_col))
                self.advance()
                self.advance()
            elif char == '<' and next_char == '=':
                self.tokens.append(Token(TokenType.LESS_EQUAL, '<=', self.line, start_col))
                self.advance()
                self.advance()
            elif char == '>' and next_char == '=':
                self.tokens.append(Token(TokenType.GREATER_EQUAL, '>=', self.line, start_col))
                self.advance()
                self.advance()
            elif char == '<' and next_char == '<':
                self.tokens.append(Token(TokenType.SHIFT_LEFT, '<<', self.line, start_col))
                self.advance()
                self.advance()
            elif char == '>' and next_char == '>':
                self.tokens.append(Token(TokenType.SHIFT_RIGHT, '>>', self.line, start_col))
                self.advance()
                self.advance()
            elif char == '&' and next_char == '&':
                self.tokens.append(Token(TokenType.AND, '&&', self.line, start_col))
                self.advance()
                self.advance()
            elif char == '|' and next_char == '|':
                self.tokens.append(Token(TokenType.OR, '||', self.line, start_col))
                self.advance()
                self.advance()
            elif char == ':' and next_char == ':':
                self.tokens.append(Token(TokenType.SCOPE, '::', self.line, start_col))
                self.advance()
                self.advance()
            elif char == '-' and next_char == '>':
                self.tokens.append(Token(TokenType.ARROW, '->', self.line, start_col))
                self.advance()
                self.advance()
            # Single-character operators
            elif char == '+':
                self.tokens.append(Token(TokenType.PLUS, '+', self.line, start_col))
                self.advance()
            elif char == '-':
                self.tokens.append(Token(TokenType.MINUS, '-', self.line, start_col))
                self.advance()
            elif char == '*':
                self.tokens.append(Token(TokenType.MULTIPLY, '*', self.line, start_col))
                self.advance()
            elif char == '/':
                self.tokens.append(Token(TokenType.DIVIDE, '/', self.line, start_col))
                self.advance()
            elif char == '%':
                self.tokens.append(Token(TokenType.MODULO, '%', self.line, start_col))
                self.advance()
            elif char == '=':
                self.tokens.append(Token(TokenType.ASSIGN, '=', self.line, start_col))
                self.advance()
            elif char == '<':
                self.tokens.append(Token(TokenType.LESS_THAN, '<', self.line, start_col))
                self.advance()
            elif char == '>':
                self.tokens.append(Token(TokenType.GREATER_THAN, '>', self.line, start_col))
                self.advance()
            elif char == '!':
                self.tokens.append(Token(TokenType.NOT, '!', self.line, start_col))
                self.advance()
            elif char == '?':
                self.tokens.append(Token(TokenType.QUESTION, '?', self.line, start_col))
                self.advance()
            elif char == '&':
                self.tokens.append(Token(TokenType.AMPERSAND, '&', self.line, start_col))
                self.advance()
            elif char == '~':  # ✅ Added for destructor syntax
                self.tokens.append(Token(TokenType.TILDE, '~', self.line, start_col))
                self.advance()
            elif char == ';':
                self.tokens.append(Token(TokenType.SEMICOLON, ';', self.line, start_col))
                self.advance()
            elif char == ',':
                self.tokens.append(Token(TokenType.COMMA, ',', self.line, start_col))
                self.advance()
            elif char == '(':
                self.tokens.append(Token(TokenType.LPAREN, '(', self.line, start_col))
                self.advance()
            elif char == ')':
                self.tokens.append(Token(TokenType.RPAREN, ')', self.line, start_col))
                self.advance()
            elif char == '{':
                self.tokens.append(Token(TokenType.LBRACE, '{', self.line, start_col))
                self.advance()
            elif char == '}':
                self.tokens.append(Token(TokenType.RBRACE, '}', self.line, start_col))
                self.advance()
            elif char == '[':
                self.tokens.append(Token(TokenType.LBRACKET, '[', self.line, start_col))
                self.advance()
            elif char == ']':
                self.tokens.append(Token(TokenType.RBRACKET, ']', self.line, start_col))
                self.advance()
            elif char == ':':
                self.tokens.append(Token(TokenType.COLON, ':', self.line, start_col))
                self.advance()
            elif char == '.':
                self.tokens.append(Token(TokenType.DOT, '.', self.line, start_col))
                self.advance()
            else:
                self.tokens.append(Token(TokenType.UNKNOWN, char, self.line, start_col))
                self.advance()
        
        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return self.tokens
    
    def get_tokens_summary(self):
        """Return a summary of token types and counts"""
        summary = {}
        for token in self.tokens:
            if token.type != TokenType.EOF:
                summary[token.type.name] = summary.get(token.type.name, 0) + 1
        return summary