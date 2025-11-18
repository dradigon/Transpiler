import pandas as pd
from typing import List, Dict
import sys
sys.path.append('..')
from lexer.lexer import Token, TokenType
from parser.parser import ASTNode, ASTNodeType

def visualize_tokens(tokens: List[Token]) -> pd.DataFrame:
    """Convert tokens to a pandas DataFrame for visualization"""
    token_data = []
    for idx, token in enumerate(tokens):
        if token.type != TokenType.EOF:
            token_data.append({
                'Index': idx,
                'Type': token.type.name,
                'Value': token.value if len(token.value) <= 30 else token.value[:27] + '...',
                'Line': token.line,
                'Column': token.column
            })
    
    return pd.DataFrame(token_data)

def get_token_statistics(tokens: List[Token]) -> dict:
    """Get statistics about the tokens"""
    stats = {
        'Total Tokens': len(tokens) - 1,  # Exclude EOF
        'Keywords': 0,
        'Identifiers': 0,
        'Operators': 0,
        'Literals': 0,
        'Comments': 0,
        'Preprocessor': 0
    }
    
    keyword_types = {TokenType.INT, TokenType.FLOAT, TokenType.DOUBLE, TokenType.CHAR, 
                     TokenType.BOOL, TokenType.VOID, TokenType.IF, TokenType.ELSE,
                     TokenType.WHILE, TokenType.FOR, TokenType.RETURN, TokenType.CLASS,
                     TokenType.PUBLIC, TokenType.PRIVATE, TokenType.PROTECTED}
    
    operator_types = {TokenType.PLUS, TokenType.MINUS, TokenType.MULTIPLY, TokenType.DIVIDE,
                      TokenType.MODULO, TokenType.ASSIGN, TokenType.EQUAL, TokenType.NOT_EQUAL,
                      TokenType.LESS_THAN, TokenType.GREATER_THAN, TokenType.LESS_EQUAL,
                      TokenType.GREATER_EQUAL, TokenType.AND, TokenType.OR, TokenType.NOT,
                      TokenType.INCREMENT, TokenType.DECREMENT, TokenType.PLUS_ASSIGN,
                      TokenType.MINUS_ASSIGN, TokenType.SHIFT_LEFT, TokenType.SHIFT_RIGHT}
    
    literal_types = {TokenType.INTEGER_LITERAL, TokenType.FLOAT_LITERAL, 
                     TokenType.STRING_LITERAL, TokenType.CHAR_LITERAL}
    
    for token in tokens:
        if token.type in keyword_types:
            stats['Keywords'] += 1
        elif token.type == TokenType.IDENTIFIER:
            stats['Identifiers'] += 1
        elif token.type in operator_types:
            stats['Operators'] += 1
        elif token.type in literal_types:
            stats['Literals'] += 1
        elif token.type == TokenType.COMMENT:
            stats['Comments'] += 1
        elif token.type == TokenType.PREPROCESSOR:
            stats['Preprocessor'] += 1
    
    return stats

def format_token_for_display(token: Token) -> str:
    """Format a single token for display"""
    return f"{token.type.name}: '{token.value}' (Line {token.line}, Col {token.column})"

def get_token_color(token_type: TokenType) -> str:
    """Get color for syntax highlighting"""
    color_map = {
        # Keywords - blue
        TokenType.INT: '#0000FF',
        TokenType.FLOAT: '#0000FF',
        TokenType.DOUBLE: '#0000FF',
        TokenType.CHAR: '#0000FF',
        TokenType.BOOL: '#0000FF',
        TokenType.VOID: '#0000FF',
        TokenType.IF: '#0000FF',
        TokenType.ELSE: '#0000FF',
        TokenType.WHILE: '#0000FF',
        TokenType.FOR: '#0000FF',
        TokenType.RETURN: '#0000FF',
        TokenType.CLASS: '#0000FF',
        
        # Identifiers - black
        TokenType.IDENTIFIER: '#000000',
        
        # Literals - red/orange
        TokenType.INTEGER_LITERAL: '#FF4500',
        TokenType.FLOAT_LITERAL: '#FF4500',
        TokenType.STRING_LITERAL: '#A31515',
        TokenType.CHAR_LITERAL: '#A31515',
        
        # Operators - dark gray
        TokenType.PLUS: '#666666',
        TokenType.MINUS: '#666666',
        TokenType.MULTIPLY: '#666666',
        TokenType.DIVIDE: '#666666',
        
        # Comments - green
        TokenType.COMMENT: '#008000',
        
        # Preprocessor - purple
        TokenType.PREPROCESSOR: '#800080',
    }
    return color_map.get(token_type, '#000000')

# ==================== AST Visualization Functions ====================

def visualize_ast(ast: ASTNode, max_depth: int = 10) -> List[Dict]:
    """Convert AST to a list of dictionaries for table visualization"""
    ast_data = []
    
    def traverse(node, depth=0, parent_id="root"):
        if depth > max_depth or not node:
            return
        
        node_id = f"{parent_id}.{len(ast_data)}"
        
        # Format node value
        value_str = ""
        if node.value:
            if isinstance(node.value, dict):
                if 'name' in node.value:
                    value_str = node.value['name']
                else:
                    value_str = str(node.value)[:50]
            else:
                value_str = str(node.value)[:50]
        
        ast_data.append({
            'ID': node_id,
            'Depth': depth,
            'Node Type': node.node_type.name,
            'Value': value_str,
            'Data Type': node.data_type if node.data_type else '-',
            'Children': len(node.children),
            'Line': node.line,
            'Column': node.column
        })
        
        # Traverse children
        for child in node.children:
            traverse(child, depth + 1, node_id)
    
    traverse(ast)
    return ast_data

def get_ast_statistics(ast: ASTNode) -> Dict[str, int]:
    """Get statistics about the AST"""
    stats = {
        'Total Nodes': 0,
        'Functions': 0,
        'Variables': 0,
        'If Statements': 0,
        'Loops': 0,
        'Expressions': 0,
        'Literals': 0,
        'Max Depth': 0
    }
    
    def traverse(node, depth=0):
        if not node:
            return
        
        stats['Total Nodes'] += 1
        stats['Max Depth'] = max(stats['Max Depth'], depth)
        
        # Count specific node types
        if node.node_type == ASTNodeType.FUNCTION:
            stats['Functions'] += 1
        elif node.node_type in {ASTNodeType.VARIABLE_DECLARATION, ASTNodeType.ARRAY_DECLARATION}:
            stats['Variables'] += 1
        elif node.node_type == ASTNodeType.IF_STATEMENT:
            stats['If Statements'] += 1
        elif node.node_type in {ASTNodeType.WHILE_LOOP, ASTNodeType.FOR_LOOP}:
            stats['Loops'] += 1
        elif node.node_type in {ASTNodeType.BINARY_OP, ASTNodeType.UNARY_OP, 
                                ASTNodeType.ASSIGNMENT, ASTNodeType.FUNCTION_CALL}:
            stats['Expressions'] += 1
        elif node.node_type == ASTNodeType.LITERAL:
            stats['Literals'] += 1
        
        # Traverse children
        for child in node.children:
            traverse(child, depth + 1)
    
    traverse(ast)
    return stats

def ast_to_tree_string(ast: ASTNode, max_depth: int = 10) -> str:
    """Convert AST to a tree string representation"""
    lines = []
    
    def traverse(node, prefix="", is_last=True, depth=0):
        if depth > max_depth or not node:
            return
        
        # Create the branch
        branch = "└── " if is_last else "├── "
        
        # Format node info
        value_str = ""
        if node.value:
            if isinstance(node.value, dict):
                if 'name' in node.value:
                    value_str = f" [{node.value['name']}]"
            else:
                val = str(node.value)[:30]
                value_str = f" [{val}]"
        
        type_str = f"{node.data_type}" if node.data_type else ""
        
        lines.append(f"{prefix}{branch}{node.node_type.name}{value_str} {type_str}")
        
        # Update prefix for children
        extension = "    " if is_last else "│   "
        new_prefix = prefix + extension
        
        # Traverse children
        for i, child in enumerate(node.children):
            is_last_child = (i == len(node.children) - 1)
            traverse(child, new_prefix, is_last_child, depth + 1)
    
    traverse(ast)
    return "\n".join(lines)

def get_ast_node_color(node_type: ASTNodeType) -> str:
    """Get color for AST node visualization"""
    color_map = {
        ASTNodeType.PROGRAM: '#1f77b4',
        ASTNodeType.FUNCTION: '#ff7f0e',
        ASTNodeType.VARIABLE_DECLARATION: '#2ca02c',
        ASTNodeType.ARRAY_DECLARATION: '#2ca02c',
        ASTNodeType.IF_STATEMENT: '#d62728',
        ASTNodeType.WHILE_LOOP: '#9467bd',
        ASTNodeType.FOR_LOOP: '#9467bd',
        ASTNodeType.RETURN_STATEMENT: '#8c564b',
        ASTNodeType.BINARY_OP: '#e377c2',
        ASTNodeType.UNARY_OP: '#e377c2',
        ASTNodeType.ASSIGNMENT: '#bcbd22',
        ASTNodeType.FUNCTION_CALL: '#17becf',
        ASTNodeType.LITERAL: '#ff4500',
        ASTNodeType.IDENTIFIER: '#000000',
        ASTNodeType.BLOCK: '#7f7f7f',
    }
    return color_map.get(node_type, '#666666')