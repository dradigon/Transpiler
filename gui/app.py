"""
Enhanced Streamlit GUI for C++ to Python Transpiler
Features: Dark/Light mode, improved phases, better UX
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from lexer.lexer import Lexer, TokenType
from parser.parser import Parser
from semantic.semantic import SemanticAnalyzer
from ir.ir import IRGenerator
from codegen.codegen import DirectCodeGenerator
from visualizer.visualize import (
    visualize_tokens,
    get_token_statistics,
    get_token_color,
    visualize_ast,
    get_ast_statistics,
    ast_to_tree_string,
    get_ast_node_color
)

# Page configuration
st.set_page_config(
    page_title="C++ to Python Transpiler",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="collapsed"  # ✅ Sidebar closed by default
)

# Initialize session state for dark mode
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True  # Default to dark mode

# Toggle function
def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Dynamic CSS based on theme
def get_theme_css():
    if st.session_state.dark_mode:
        return """
        <style>
        :root {
            --bg-primary: #0e1117;
            --bg-secondary: #000000;
            --bg-tertiary: #1a1a1a;
            --text-primary: #ffffff;
            --text-secondary: #e0e0e0;
            --accent: #00d4aa;
            --accent-hover: #00b894;
            --border: #444;
            --shadow: rgba(0, 0, 0, 0.5);
            --code-bg: #000000;
            --code-text: #f0f0f0;
        }
        </style>
        """
    else:
        return """
        <style>
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f8f9fa;
            --bg-tertiary: #e9ecef;
            --text-primary: #1e1e1e;
            --text-secondary: #4a4a4a;
            --accent: #0066cc;
            --accent-hover: #0052a3;
            --border: #dee2e6;
            --shadow: rgba(0, 0, 0, 0.1);
            --code-bg: #f8f9fa;
            --code-text: #1e1e1e;
        }
        </style>
        """

# Apply theme CSS
st.markdown(get_theme_css(), unsafe_allow_html=True)

# Enhanced custom CSS
st.markdown("""
    <style>
    /* Main header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent) 0%, #667eea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 1rem;
    }
    
    /* Phase headers */
    .phase-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--accent);
        margin-top: 2rem;
        padding: 1rem;
        background: var(--bg-tertiary);
        border-radius: 10px;
        border-left: 5px solid var(--accent);
        box-shadow: 0 2px 8px var(--shadow);
    }
    
    /* Enhanced text areas */
    .stTextArea textarea {
        font-family: 'Fira Code', 'Courier New', monospace;
        font-size: 14px;
        line-height: 1.6;
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }
    
    /* Code display boxes */
    .code-box {
        font-family: 'Fira Code', 'Courier New', monospace;
        background: var(--code-bg);
        color: var(--code-text);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid var(--border);
        white-space: pre;
        overflow-x: auto;
        font-size: 14px;
        line-height: 1.8;
        box-shadow: 0 2px 8px var(--shadow);
    }
    
    /* Token highlighting */
    .token-inline {
        padding: 2px 6px;
        border-radius: 4px;
        margin: 0 2px;
        font-family: 'Fira Code', monospace;
        font-size: 13px;
        font-weight: 500;
    }
    
    /* Metric cards */
    .metric-card {
        background: var(--bg-tertiary);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid var(--border);
        box-shadow: 0 2px 8px var(--shadow);
        text-align: center;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px var(--shadow);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--accent);
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
        border: 2px solid var(--accent);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px var(--shadow);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    /* DataFrames */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px var(--shadow);
    }
    
    /* Success/Error messages */
    .success-box {
        background: linear-gradient(135deg, #00b894 0%, #00d4aa 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 600;
    }
    
    .error-box {
        background: linear-gradient(135deg, #d63031 0%, #ff7675 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 600;
    }
    
    /* Theme toggle */
    .theme-toggle {
        position: fixed;
        top: 70px;
        right: 20px;
        z-index: 999;
        background: var(--bg-tertiary);
        border: 2px solid var(--border);
        border-radius: 50px;
        padding: 8px 16px;
        cursor: pointer;
        box-shadow: 0 2px 8px var(--shadow);
        transition: all 0.3s;
    }
    
    .theme-toggle:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px var(--shadow);
    }
    
    /* Sidebar enhancements */
    [data-testid="stSidebar"] {
        background: var(--bg-tertiary);
        border-right: 2px solid var(--border);
    }
    
    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stSidebar"] label {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: var(--text-primary) !important;
    }
    
    /* Fix Streamlit code blocks in dark mode */
    .stCodeBlock {
        background: var(--code-bg) !important;
    }
    
    .stCodeBlock code {
        color: var(--code-text) !important;
    }
    
    /* Fix markdown code blocks */
    code {
        background: var(--bg-tertiary) !important;
        color: var(--code-text) !important;
        padding: 2px 6px;
        border-radius: 4px;
    }
    
    pre {
        background: var(--code-bg) !important;
        color: var(--code-text) !important;
        padding: 1rem !important;
        border-radius: 8px !important;
    }
    
    pre code {
        background: transparent !important;
        color: var(--code-text) !important;
    }
    
    /* Progress indicators */
    .phase-indicator {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px;
        background: var(--bg-tertiary);
        border-radius: 8px;
        margin: 5px 0;
    }
    
    .phase-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: var(--accent);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    </style>
""", unsafe_allow_html=True)

def load_sample_code(sample_name):
    """Load sample C++ code"""
    samples = {
        "Simple Addition": """#include <iostream>
using namespace std;

int add(int a, int b) {
    return a + b;
}

int main() {
    int x = 10;
    int y = 20;
    int sum = add(x, y);
    cout << "Sum: " << sum << endl;
    return 0;
}""",
        
        "Array Operations": """#include <iostream>
using namespace std;

int main() {
    int arr[5] = {1, 2, 3, 4, 5};
    int sum = 0;
    
    for (int i = 0; i < 5; i++) {
        sum += arr[i];
    }
    
    cout << "Sum: " << sum << endl;
    return 0;
}""",
        
        "Nested Loops": """#include <iostream>
using namespace std;

int main() {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            cout << i * j << " ";
        }
        cout << endl;
    }
    return 0;
}""",
        
        "Conditionals": """#include <iostream>
using namespace std;

int main() {
    int x = 15;
    
    if (x > 10) {
        cout << "Greater than 10" << endl;
    } else if (x > 5) {
        cout << "Greater than 5" << endl;
    } else {
        cout << "5 or less" << endl;
    }
    
    return 0;
}"""
    }
    return samples.get(sample_name, "")

def display_tokens_enhanced(tokens):
    """Enhanced token display with better syntax highlighting"""
    html_output = "<div style='font-family: monospace; line-height: 2.2; padding: 1rem; background: var(--code-bg); color: var(--code-text); border-radius: 8px;'>"
    
    for token in tokens:
        if token.type != TokenType.EOF:
            # Get base color
            base_color = get_token_color(token.type)
            
            # Adjust colors for dark mode
            if st.session_state.dark_mode:
                color_map = {
                    '#0000FF': '#569cd6',  # Keywords - light blue
                    '#000000': '#d4d4d4',  # Identifiers - light gray
                    '#FF4500': '#ce9178',  # Numeric literals - light orange
                    '#A31515': '#ce9178',  # String literals - light orange/brown
                    '#666666': '#d4d4d4',  # Operators - light gray
                    '#008000': '#6a9955',  # Comments - light green
                    '#800080': '#c586c0',  # Preprocessor - light purple
                }
                color = color_map.get(base_color, '#d4d4d4')
            else:
                color = base_color
            
            value = token.value.replace('<', '&lt;').replace('>', '&gt;').replace(' ', '&nbsp;')
            
            # Add background for better visibility
            if token.type in [TokenType.STRING_LITERAL, TokenType.CHAR_LITERAL]:
                bg_color = 'rgba(206, 145, 120, 0.15)' if st.session_state.dark_mode else 'rgba(163, 21, 21, 0.1)'
            elif token.type in [TokenType.INTEGER_LITERAL, TokenType.FLOAT_LITERAL]:
                bg_color = 'rgba(206, 145, 120, 0.15)' if st.session_state.dark_mode else 'rgba(255, 69, 0, 0.1)'
            elif token.type == TokenType.COMMENT:
                bg_color = 'rgba(106, 153, 85, 0.15)' if st.session_state.dark_mode else 'rgba(0, 128, 0, 0.1)'
            else:
                bg_color = 'transparent'
            
            html_output += f"<span class='token-inline' style='color: {color}; background: {bg_color};' title='{token.type.name} (Line {token.line}, Col {token.column})'>{value}</span>"
    
    html_output += "</div>"
    return html_output

def create_metric_card(label, value, icon="📊"):
    """Create an enhanced metric card"""
    return f"""
    <div class="metric-card">
        <div style="font-size: 2rem;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

def create_phase_indicator(phase_name, status="running"):
    """Create a phase indicator"""
    status_icons = {
        "running": "🔄",
        "complete": "✅",
        "error": "❌"
    }
    return f"""
    <div class="phase-indicator">
        <div class="phase-dot"></div>
        <span style="font-weight: 600;">{status_icons.get(status, "⏳")} {phase_name}</span>
    </div>
    """

def main():
    # Theme toggle button
    col1, col2, col3 = st.columns([6, 1, 1])
    with col3:
        theme_icon = "🌙" if st.session_state.dark_mode else "☀️"
        if st.button(theme_icon, help="Toggle theme", key="theme_toggle"):
            toggle_theme()
            st.rerun()
    
    # Header
    st.markdown('<div class="main-header">🔄 C++ to Python Transpiler</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: var(--text-secondary); font-size: 1.1rem;'>Advanced Multi-Phase Compilation System</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        
        # Input method selection
        input_method = st.radio(
            "📥 Input Method:",
            ["✍️ Write Code", "📋 Load Sample", "📁 Upload File"],
            format_func=lambda x: x
        )
        
        cpp_code = ""
        
        if input_method == "📋 Load Sample":
            sample_choice = st.selectbox(
                "Select Sample:",
                ["Simple Addition", "Array Operations", "Nested Loops", "Conditionals"]
            )
            if st.button("🚀 Load Sample", use_container_width=True):
                cpp_code = load_sample_code(sample_choice)
                st.session_state.cpp_code = cpp_code
                st.success("✅ Sample loaded!")
        
        elif input_method == "📁 Upload File":
            uploaded_file = st.file_uploader("Choose a C++ file", type=['cpp', 'c', 'cc', 'cxx'])
            if uploaded_file is not None:
                cpp_code = uploaded_file.read().decode('utf-8')
                st.session_state.cpp_code = cpp_code
                st.success(f"✅ Loaded {uploaded_file.name}")
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Phases")
        
        show_lexer = st.checkbox("🔬 Phase 1: Lexical Analysis", value=True)
        show_parser = st.checkbox("🌳 Phase 2: Syntax Analysis", value=True)
        show_semantic = st.checkbox("🔍 Phase 3: Semantic Analysis", value=True)
        show_ir = st.checkbox("⚙️ Phase 4: IR Generation", value=True)
        show_codegen = st.checkbox("🐍 Phase 5: Code Generation", value=True)
        
        st.markdown("---")
        st.markdown("### 🎨 Display Options")
        show_statistics = st.checkbox("📈 Show Statistics", value=True)
        show_visualizations = st.checkbox("📊 Show Charts", value=True)
        detailed_view = st.checkbox("🔍 Detailed View", value=False)
        
        st.markdown("---")
        st.info("💡 **Tip:** Click the theme toggle (top-right) to switch between dark and light modes!")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="phase-header">📝 Input: C++ Source Code</div>', unsafe_allow_html=True)
        
        if 'cpp_code' not in st.session_state:
            st.session_state.cpp_code = load_sample_code("Simple Addition")
        
        cpp_input = st.text_area(
            "Enter your C++ code:",
            value=st.session_state.cpp_code,
            height=450,
            key="cpp_input_area",
            placeholder="// Type or paste your C++ code here..."
        )
        st.session_state.cpp_code = cpp_input
        
        # Analyze button with better styling
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            analyze_button = st.button("🚀 Transpile Now", type="primary", use_container_width=True)
    
    with col2:
        st.markdown('<div class="phase-header">🐍 Output: Python Code</div>', unsafe_allow_html=True)
        python_output_placeholder = st.empty()
    
    # Analysis section
    if analyze_button or cpp_input:
        if cpp_input.strip():
            st.markdown("---")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Store data for reuse
            tokens = None
            ast = None
            semantic_analyzer = None
            ir_generator = None
            python_code = None
            
            # Phase 1: Lexical Analysis
            if show_lexer:
                progress_bar.progress(20)
                status_text.markdown(create_phase_indicator("Lexical Analysis", "running"), unsafe_allow_html=True)
                
                st.markdown('<div class="phase-header">🔬 Phase 1: Lexical Analysis</div>', unsafe_allow_html=True)
                
                try:
                    lexer = Lexer(cpp_input)
                    tokens = lexer.tokenize()
                    
                    st.markdown('<div class="success-box">✅ Lexical analysis completed successfully!</div>', unsafe_allow_html=True)
                    
                    if show_statistics:
                        stats = get_token_statistics(tokens)
                        
                        # Display metrics in cards
                        metric_cols = st.columns(6)
                        metrics = [
                            ("Total Tokens", stats['Total Tokens'], "🔢"),
                            ("Keywords", stats['Keywords'], "🔑"),
                            ("Identifiers", stats['Identifiers'], "🏷️"),
                            ("Operators", stats['Operators'], "➕"),
                            ("Literals", stats['Literals'], "💎"),
                            ("Comments", stats['Comments'], "💬")
                        ]
                        
                        for col, (label, value, icon) in zip(metric_cols, metrics):
                            with col:
                                st.markdown(create_metric_card(label, value, icon), unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    lex_tab1, lex_tab2, lex_tab3 = st.tabs(["📋 Token Table", "🎨 Syntax Highlighted", "📊 Distribution"])
                    
                    with lex_tab1:
                        df = visualize_tokens(tokens)
                        st.dataframe(df, use_container_width=True, height=350)
                        csv = df.to_csv(index=False)
                        st.download_button("📥 Download CSV", csv, "tokens.csv", "text/csv", use_container_width=True)
                    
                    with lex_tab2:
                        colored_html = display_tokens_enhanced(tokens)
                        st.markdown(colored_html, unsafe_allow_html=True)
                    
                    with lex_tab3:
                        if show_visualizations:
                            stats = get_token_statistics(tokens)
                            # Remove total and create pie chart
                            chart_data = {k: v for k, v in stats.items() if k != 'Total Tokens' and v > 0}
                            
                            fig = px.pie(
                                values=list(chart_data.values()),
                                names=list(chart_data.keys()),
                                title="Token Distribution",
                                color_discrete_sequence=px.colors.qualitative.Set3
                            )
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(size=12)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    status_text.markdown(create_phase_indicator("Lexical Analysis", "complete"), unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown(f'<div class="error-box">❌ Lexer Error: {str(e)}</div>', unsafe_allow_html=True)
                    if detailed_view:
                        st.exception(e)
                    return
            
            # Phase 2: Parsing
            if show_parser:
                progress_bar.progress(40)
                status_text.markdown(create_phase_indicator("Syntax Analysis", "running"), unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown('<div class="phase-header">🌳 Phase 2: Syntax Analysis (Parsing)</div>', unsafe_allow_html=True)
                
                try:
                    if tokens is None:
                        lexer = Lexer(cpp_input)
                        tokens = lexer.tokenize()
                    
                    parser = Parser(tokens)
                    ast = parser.parse()
                    
                    if parser.errors:
                        st.markdown(f'<div class="error-box">⚠️ Found {len(parser.errors)} parser warnings</div>', unsafe_allow_html=True)
                        if detailed_view:
                            with st.expander("🔍 View Parser Messages"):
                                for error in parser.errors[:10]:
                                    st.code(error)
                    else:
                        st.markdown('<div class="success-box">✅ AST generated successfully!</div>', unsafe_allow_html=True)
                    
                    if show_statistics:
                        ast_stats = get_ast_statistics(ast)
                        
                        metric_cols = st.columns(6)
                        metrics = [
                            ("Total Nodes", ast_stats['Total Nodes'], "🌐"),
                            ("Functions", ast_stats['Functions'], "⚡"),
                            ("Variables", ast_stats['Variables'], "📦"),
                            ("Loops", ast_stats['Loops'], "🔁"),
                            ("Conditionals", ast_stats['If Statements'], "🔀"),
                            ("Max Depth", ast_stats['Max Depth'], "📏")
                        ]
                        
                        for col, (label, value, icon) in zip(metric_cols, metrics):
                            with col:
                                st.markdown(create_metric_card(label, value, icon), unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    ast_tab1, ast_tab2, ast_tab3 = st.tabs(["🌲 Tree View", "📋 Node Table", "📊 Statistics"])
                    
                    with ast_tab1:
                        tree_str = ast_to_tree_string(ast, max_depth=15)
                        # Use styled div for better visibility
                        st.markdown(f'''
                        <div style="
                            font-family: 'Courier New', monospace;
                            background: var(--code-bg);
                            color: var(--code-text);
                            padding: 1.5rem;
                            border-radius: 10px;
                            border: 1px solid var(--border);
                            white-space: pre;
                            overflow-x: auto;
                            font-size: 14px;
                            line-height: 1.8;
                        ">{tree_str}</div>
                        ''', unsafe_allow_html=True)
                        st.download_button("📥 Download Tree", tree_str, "ast_tree.txt", "text/plain", use_container_width=True)
                    
                    with ast_tab2:
                        ast_data = visualize_ast(ast, max_depth=15)
                        ast_df = pd.DataFrame(ast_data)
                        st.dataframe(ast_df, use_container_width=True, height=350)
                        csv = ast_df.to_csv(index=False)
                        st.download_button("📥 Download CSV", csv, "ast_nodes.csv", "text/csv", use_container_width=True)
                    
                    with ast_tab3:
                        if show_visualizations:
                            ast_stats = get_ast_statistics(ast)
                            
                            # Create bar chart
                            chart_data = {
                                'Category': ['Functions', 'Variables', 'If Statements', 'Loops', 'Expressions', 'Literals'],
                                'Count': [
                                    ast_stats['Functions'],
                                    ast_stats['Variables'],
                                    ast_stats['If Statements'],
                                    ast_stats['Loops'],
                                    ast_stats['Expressions'],
                                    ast_stats['Literals']
                                ]
                            }
                            
                            fig = px.bar(
                                chart_data,
                                x='Category',
                                y='Count',
                                title="AST Node Type Distribution",
                                color='Count',
                                color_continuous_scale='Viridis'
                            )
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(size=12)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    status_text.markdown(create_phase_indicator("Syntax Analysis", "complete"), unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown(f'<div class="error-box">❌ Parser Error: {str(e)}</div>', unsafe_allow_html=True)
                    if detailed_view:
                        st.exception(e)
                    return
            
            # Phase 3: Semantic Analysis
            if show_semantic:
                progress_bar.progress(60)
                status_text.markdown(create_phase_indicator("Semantic Analysis", "running"), unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown('<div class="phase-header">🔍 Phase 3: Semantic Analysis</div>', unsafe_allow_html=True)
                
                try:
                    if ast is None:
                        lexer = Lexer(cpp_input)
                        tokens = lexer.tokenize()
                        parser = Parser(tokens)
                        ast = parser.parse()
                    
                    semantic_analyzer = SemanticAnalyzer(ast)
                    success = semantic_analyzer.analyze()
                    
                    if success:
                        st.markdown('<div class="success-box">✅ Semantic analysis passed - No errors found!</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="error-box">⚠️ Found {len(semantic_analyzer.errors)} semantic issues</div>', unsafe_allow_html=True)
                    
                    sem_tab1, sem_tab2 = st.tabs(["📊 Symbol Table", "⚠️ Messages"])
                    
                    with sem_tab1:
                        symbols = semantic_analyzer.symbol_table.get_all_symbols()
                        if symbols:
                            symbol_data = []
                            for sym in symbols:
                                symbol_data.append({
                                    'Name': sym.name,
                                    'Type': sym.symbol_type.name,
                                    'Data Type': sym.data_type,
                                    'Scope Level': sym.scope_level,
                                    'Line': sym.line
                                })
                            sym_df = pd.DataFrame(symbol_data)
                            st.dataframe(sym_df, use_container_width=True, height=300)
                            
                            if show_visualizations:
                                # Symbol type distribution
                                type_counts = sym_df['Type'].value_counts()
                                fig = px.pie(
                                    values=type_counts.values,
                                    names=type_counts.index,
                                    title="Symbol Type Distribution",
                                    color_discrete_sequence=px.colors.qualitative.Pastel
                                )
                                fig.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("ℹ️ No symbols declared in this program")
                    
                    with sem_tab2:
                        if semantic_analyzer.errors:
                            st.subheader("❌ Errors")
                            for i, error in enumerate(semantic_analyzer.errors, 1):
                                st.error(f"{i}. {error}")
                        
                        if semantic_analyzer.warnings:
                            st.subheader("⚠️ Warnings")
                            for i, warning in enumerate(semantic_analyzer.warnings, 1):
                                st.warning(f"{i}. {warning}")
                        
                        if not semantic_analyzer.errors and not semantic_analyzer.warnings:
                            st.success("🎉 Perfect! No errors or warnings detected!")
                    
                    status_text.markdown(create_phase_indicator("Semantic Analysis", "complete"), unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown(f'<div class="error-box">❌ Semantic Error: {str(e)}</div>', unsafe_allow_html=True)
                    if detailed_view:
                        st.exception(e)
            
            # Phase 4: IR Generation
            if show_ir:
                progress_bar.progress(80)
                status_text.markdown(create_phase_indicator("IR Generation", "running"), unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown('<div class="phase-header">⚙️ Phase 4: Intermediate Representation</div>', unsafe_allow_html=True)
                
                try:
                    if ast is None:
                        lexer = Lexer(cpp_input)
                        tokens = lexer.tokenize()
                        parser = Parser(tokens)
                        ast = parser.parse()
                    
                    if semantic_analyzer is None:
                        semantic_analyzer = SemanticAnalyzer(ast)
                        semantic_analyzer.analyze()
                    
                    ir_generator = IRGenerator(
                        ast,
                        symbol_table=semantic_analyzer.symbol_table,
                        includes=semantic_analyzer.includes,
                        defines=semantic_analyzer.defines
                    )
                    ir_instructions = ir_generator.generate()
                    
                    st.markdown('<div class="success-box">✅ IR generation completed successfully!</div>', unsafe_allow_html=True)
                    
                    if show_statistics:
                        ir_summary = ir_generator.get_ir_summary()
                        
                        metric_cols = st.columns(4)
                        metrics = [
                            ("Instructions", ir_summary.get('total_instructions', 0), "📝"),
                            ("Temp Variables", ir_summary.get('temp_variables', 0), "🔧"),
                            ("Labels", ir_summary.get('labels', 0), "🏷️"),
                            ("Operations", len(ir_instructions), "⚡")
                        ]
                        
                        for col, (label, value, icon) in zip(metric_cols, metrics):
                            with col:
                                st.markdown(create_metric_card(label, value, icon), unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    ir_tab1, ir_tab2 = st.tabs(["📋 IR Code", "📊 Statistics"])
                    
                    with ir_tab1:
                        ir_str = ir_generator.to_string()
                        # Use styled div for better visibility
                        st.markdown(f'''
                        <div style="
                            font-family: 'Courier New', monospace;
                            background: var(--code-bg);
                            color: var(--code-text);
                            padding: 1.5rem;
                            border-radius: 10px;
                            border: 1px solid var(--border);
                            white-space: pre;
                            overflow-x: auto;
                            font-size: 14px;
                            line-height: 1.8;
                        ">{ir_str}</div>
                        ''', unsafe_allow_html=True)
                        st.download_button("📥 Download IR", ir_str, "ir_code.txt", "text/plain", use_container_width=True)
                    
                    with ir_tab2:
                        if show_visualizations:
                            ir_summary = ir_generator.get_ir_summary()
                            # ✅ FIX: Filter out non-numeric values and nested dicts before comparison
                            breakdown = {k.replace('_count', '').replace('_', ' ').title(): v 
                                       for k, v in ir_summary.items() 
                                       if k not in ['total_instructions', 'temp_variables', 'labels', 'opcode_distribution'] 
                                       and isinstance(v, (int, float)) and v > 0}
                            
                            if breakdown:
                                fig = px.bar(
                                    x=list(breakdown.keys()),
                                    y=list(breakdown.values()),
                                    title="IR Instruction Breakdown",
                                    labels={'x': 'Instruction Type', 'y': 'Count'},
                                    color=list(breakdown.values()),
                                    color_continuous_scale='Blues'
                                )
                                fig.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    showlegend=False
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("ℹ️ No detailed instruction breakdown available")
                    
                    status_text.markdown(create_phase_indicator("IR Generation", "complete"), unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown(f'<div class="error-box">❌ IR Generation Error: {str(e)}</div>', unsafe_allow_html=True)
                    if detailed_view:
                        st.exception(e)
            
            # Phase 5: Code Generation
            if show_codegen:
                progress_bar.progress(95)
                status_text.markdown(create_phase_indicator("Code Generation", "running"), unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown('<div class="phase-header">🐍 Phase 5: Python Code Generation</div>', unsafe_allow_html=True)
                
                try:
                    if ast is None:
                        lexer = Lexer(cpp_input)
                        tokens = lexer.tokenize()
                        parser = Parser(tokens)
                        ast = parser.parse()
                    
                    if semantic_analyzer is None:
                        semantic_analyzer = SemanticAnalyzer(ast)
                        semantic_analyzer.analyze()
                    
                    code_generator = DirectCodeGenerator(
                        ast,
                        symbol_table=semantic_analyzer.symbol_table,
                        includes=semantic_analyzer.includes,
                        defines=semantic_analyzer.defines
                    )
                    python_code = code_generator.generate()
                    
                    st.markdown('<div class="success-box">✅ Python code generated successfully!</div>', unsafe_allow_html=True)
                    
                    codegen_tab1, codegen_tab2, codegen_tab3 = st.tabs(["🐍 Python Code", "▶️ Execute", "📊 Metrics"])
                    
                    with codegen_tab1:
                        # Display with better styling
                        st.markdown(f'''
                        <div style="
                            font-family: 'Courier New', monospace;
                            background: var(--code-bg);
                            color: var(--code-text);
                            padding: 1.5rem;
                            border-radius: 10px;
                            border: 1px solid var(--border);
                            white-space: pre;
                            overflow-x: auto;
                            font-size: 14px;
                            line-height: 1.8;
                        ">{python_code}</div>
                        ''', unsafe_allow_html=True)
                        
                        st.download_button(
                                "📥 Download Python File",
                                python_code,
                                "transpiled_code.py",
                                "text/x-python",
                                use_container_width=True
                        )
                    
                    with codegen_tab2:
                        st.markdown("#### 🚀 Execute Generated Code")
                        
                        if st.button("▶️ Run Python Code", use_container_width=True):
                            with st.spinner("Executing..."):
                                try:
                                    import io
                                    import contextlib
                                    
                                    # Capture stdout
                                    output_buffer = io.StringIO()
                                    with contextlib.redirect_stdout(output_buffer):
                                        # ✅ FIX: Set __name__ to '__main__' so if __name__ check works
                                        exec_globals = {'__name__': '__main__', '__builtins__': __builtins__}
                                        exec(python_code, exec_globals)
                                    
                                    output = output_buffer.getvalue()
                                    
                                    st.markdown('<div class="success-box">✅ Execution completed successfully!</div>', unsafe_allow_html=True)
                                    
                                    if output:
                                        st.markdown("**Program Output:**")
                                        st.code(output, language='text')
                                    else:
                                        st.info("ℹ️ Program executed successfully (no output)")
                                    
                                except Exception as exec_error:
                                    st.markdown(f'<div class="error-box">❌ Execution Error: {str(exec_error)}</div>', unsafe_allow_html=True)
                                    if detailed_view:
                                        st.exception(exec_error)
                        
                        st.warning("⚠️ **Note:** Execution is sandboxed and may not work for all programs")
                    
                    with codegen_tab3:
                        lines = len(python_code.split('\n'))
                        chars = len(python_code)
                        words = len(python_code.split())
                        
                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            st.markdown(create_metric_card("Lines", lines, "📏"), unsafe_allow_html=True)
                        with metric_cols[1]:
                            st.markdown(create_metric_card("Characters", chars, "🔤"), unsafe_allow_html=True)
                        with metric_cols[2]:
                            st.markdown(create_metric_card("Words", words, "📝"), unsafe_allow_html=True)
                        with metric_cols[3]:
                            functions = python_code.count('def ')
                            st.markdown(create_metric_card("Functions", functions, "⚡"), unsafe_allow_html=True)
                    
                    # Update output in main area
                    python_output_placeholder.code(python_code, language='python', line_numbers=True)
                    
                    status_text.markdown(create_phase_indicator("Code Generation", "complete"), unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown(f'<div class="error-box">❌ Code Generation Error: {str(e)}</div>', unsafe_allow_html=True)
                    if detailed_view:
                        st.exception(e)
            
            # Complete
            progress_bar.progress(100)
            status_text.markdown('<div class="success-box">🎉 All phases completed successfully!</div>', unsafe_allow_html=True)
            
            # Summary Section
            st.markdown("---")
            st.markdown('<div class="phase-header">📈 Compilation Summary</div>', unsafe_allow_html=True)
            
            summary_cols = st.columns(5)
            
            summary_data = []
            if tokens:
                summary_data.append(("Tokens", len(tokens) - 1, "🔢"))
            if ast:
                ast_stats = get_ast_statistics(ast)
                summary_data.append(("AST Nodes", ast_stats['Total Nodes'], "🌐"))
            if semantic_analyzer:
                symbols = len(semantic_analyzer.symbol_table.get_all_symbols())
                summary_data.append(("Symbols", symbols, "📦"))
            if ir_generator:
                summary_data.append(("IR Instructions", len(ir_generator.instructions), "⚙️"))
            if python_code:
                lines = len(python_code.split('\n'))
                summary_data.append(("Python Lines", lines, "🐍"))
            
            for col, (label, value, icon) in zip(summary_cols, summary_data):
                with col:
                    st.markdown(create_metric_card(label, value, icon), unsafe_allow_html=True)
            
            # Comparison chart
            if show_visualizations and len(summary_data) >= 3:
                st.markdown("---")
                chart_df = pd.DataFrame({
                    'Phase': [x[0] for x in summary_data],
                    'Count': [x[1] for x in summary_data]
                })
                
                fig = go.Figure(data=[
                    go.Scatter(
                        x=chart_df['Phase'],
                        y=chart_df['Count'],
                        mode='lines+markers',
                        line=dict(color='#00d4aa', width=3),
                        marker=dict(size=12, color='#00d4aa'),
                        fill='tozeroy',
                        fillcolor='rgba(0, 212, 170, 0.2)'
                    )
                ])
                
                fig.update_layout(
                    title="Compilation Phases Overview",
                    xaxis_title="Phase",
                    yaxis_title="Count",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("⚠️ Please enter some C++ code to analyze")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: var(--text-secondary); padding: 2rem; background: var(--bg-tertiary); border-radius: 10px; margin-top: 2rem;'>
            <p style='font-size: 1.3rem; font-weight: 700; margin-bottom: 0.5rem;'>🔄 C++ to Python Transpiler v4.0</p>
            <p style='font-size: 0.9rem; margin-bottom: 1rem;'>Complete Multi-Phase Compilation System</p>
            <div style='display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;'>
                <span style='padding: 5px 15px; background: var(--accent); color: white; border-radius: 20px; font-weight: 600;'>✅ Lexer</span>
                <span style='padding: 5px 15px; background: var(--accent); color: white; border-radius: 20px; font-weight: 600;'>✅ Parser</span>
                <span style='padding: 5px 15px; background: var(--accent); color: white; border-radius: 20px; font-weight: 600;'>✅ Semantic</span>
                <span style='padding: 5px 15px; background: var(--accent); color: white; border-radius: 20px; font-weight: 600;'>✅ IR Gen</span>
                <span style='padding: 5px 15px; background: var(--accent); color: white; border-radius: 20px; font-weight: 600;'>✅ CodeGen</span>
            </div>
            <p style='margin-top: 1rem; font-size: 0.85rem;'>Built with Streamlit • Powered by Python</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()