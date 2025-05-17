import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import random
import yfinance as yf
import ta
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# Theme configurations
THEMES = {
    "Angry Birds": {
        "primary_color": "#FF6B6B",  # Red Bird color
        "secondary_color": "#4ECDC4",  # Blue Bird color
        "background_color": "#F7F9FC",  # Light sky blue
        "text_color": "#2D3436",  # Dark gray for readability
        "accent_color": "#FFD93D",  # Yellow Bird color
        "font": "Comic Sans MS",  # Child-friendly font
        "gif": "https://media.giphy.com/media/3o7TKz2eMXx7dn95FS/giphy.gif",
        "loading_gif": "https://media.giphy.com/media/3o7TKz2eMXx7dn95FS/giphy.gif",
        "success_gif": "https://media.giphy.com/media/3o7TKz2eMXx7dn95FS/giphy.gif",
        "background_gif": "https://media.giphy.com/media/3o7TKz2eMXx7dn95FS/giphy.gif",
        "characters": {
            "red": "https://media.giphy.com/media/3o7TKz2eMXx7dn95FS/giphy.gif",
            "blue": "https://media.giphy.com/media/3o7TKz2eMXx7dn95FS/giphy.gif",
            "yellow": "https://media.giphy.com/media/3o7TKz2eMXx7dn95FS/giphy.gif",
            "black": "https://media.giphy.com/media/3o7TKz2eMXx7dn95FS/giphy.gif",
            "white": "https://media.giphy.com/media/3o7TKz2eMXx7dn95FS/giphy.gif"
        },
        "stages": {
            "poached_eggs": "https://media.giphy.com/media/3o7TKz2eMXx7dn95FS/giphy.gif",
            "mighty_hoax": "https://media.giphy.com/media/3o7TKz2eMXx7dn95FS/giphy.gif",
            "danger_above": "https://media.giphy.com/media/3o7TKz2eMXx7dn95FS/giphy.gif"
        }
    }
}

# Game mechanics
class GameState:
    def __init__(self):
        self.bird_position = [0, 0]  # [x, y]
        self.bird_velocity = [0, 0]  # [vx, vy]
        self.target_position = [0, 0]
        self.obstacles = []
        self.score = 0
        self.power = 0
        self.angle = 0
        self.is_launched = False
        self.game_over = False
        self.level_complete = False

# Financial Analysis Functions
def get_stock_data(ticker, period="1y"):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def calculate_technical_indicators(df):
    """Calculate technical indicators for the stock"""
    if df is None or df.empty:
        return None
    
    # Add technical indicators
    df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    
    # RSI
    df['RSI'] = RSIIndicator(close=df['Close']).rsi()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['Close'])
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_middle'] = bb.bollinger_mavg()
    df['BB_lower'] = bb.bollinger_lband()
    
    return df

def plot_stock_analysis(df, ticker):
    """Create interactive stock analysis plots"""
    if df is None or df.empty:
        return
    
    # Price and Moving Averages
    fig1 = go.Figure()
    fig1.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Price'))
    fig1.add_trace(go.Scatter(x=df.index, y=df['SMA_20'],
                             name='SMA 20',
                             line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=df.index, y=df['SMA_50'],
                             name='SMA 50',
                             line=dict(color='red')))
    
    fig1.update_layout(title=f'{ticker} Price and Moving Averages',
                      yaxis_title='Price',
                      xaxis_title='Date',
                      template='plotly_white')
    
    # RSI
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df['RSI'],
                             name='RSI',
                             line=dict(color='purple')))
    fig2.add_hline(y=70, line_dash="dash", line_color="red")
    fig2.add_hline(y=30, line_dash="dash", line_color="green")
    
    fig2.update_layout(title='Relative Strength Index (RSI)',
                      yaxis_title='RSI',
                      xaxis_title='Date',
                      template='plotly_white')
    
    # Bollinger Bands
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df.index, y=df['Close'],
                             name='Price',
                             line=dict(color='black')))
    fig3.add_trace(go.Scatter(x=df.index, y=df['BB_upper'],
                             name='Upper Band',
                             line=dict(color='gray', dash='dash')))
    fig3.add_trace(go.Scatter(x=df.index, y=df['BB_lower'],
                             name='Lower Band',
                             line=dict(color='gray', dash='dash')))
    
    fig3.update_layout(title='Bollinger Bands',
                      yaxis_title='Price',
                      xaxis_title='Date',
                      template='plotly_white')
    
    return fig1, fig2, fig3

# Initialize session state variables
if 'current_theme' not in st.session_state:
    st.session_state.current_theme = "Angry Birds"
if 'selected_character' not in st.session_state:
    st.session_state.selected_character = "Red Bird"
if 'current_level' not in st.session_state:
    st.session_state.current_level = 1
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'stars' not in st.session_state:
    st.session_state.stars = 0
if 'unlocked_levels' not in st.session_state:
    st.session_state.unlocked_levels = [1]
if 'game_state' not in st.session_state:
    st.session_state.game_state = GameState()
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = "AAPL"

# Set page configuration
st.set_page_config(
    page_title="Angry Birds Financial Adventure",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
def apply_theme_style():
    """Apply custom CSS based on the selected theme"""
    theme = st.session_state.current_theme
    theme_config = THEMES[theme]
    
    css = f"""
    <style>
        .main {{
            background-color: {theme_config['background_color']};
            font-family: {theme_config['font']}, cursive;
        }}
        
        .stButton>button {{
            background-color: {theme_config['primary_color']};
            color: white;
            border-radius: 20px;
            padding: 10px 20px;
            font-family: {theme_config['font']}, cursive;
            font-size: 1.2rem;
            border: none;
            transition: transform 0.3s;
        }}
        
        .stButton>button:hover {{
            transform: scale(1.05);
            background-color: {theme_config['accent_color']};
        }}
        
        .game-container {{
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            position: relative;
            height: 400px;
            overflow: hidden;
        }}
        
        .analysis-container {{
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        .stock-card {{
            background-color: {theme_config['secondary_color']};
            border-radius: 15px;
            padding: 15px;
            margin: 10px 0;
            text-align: center;
        }}
        
        .indicator-value {{
            font-size: 1.2rem;
            font-weight: bold;
            color: {theme_config['primary_color']};
        }}
        
        .level-card {{
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        
        .level-card:hover {{
            transform: scale(1.02);
        }}
        
        .character-card {{
            background-color: {theme_config['secondary_color']};
            border-radius: 15px;
            padding: 15px;
            margin: 10px 0;
            text-align: center;
        }}
        
        .score-display {{
            background-color: {theme_config['accent_color']};
            color: {theme_config['text_color']};
            padding: 10px 20px;
            border-radius: 20px;
            font-size: 1.2rem;
            font-weight: bold;
        }}
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)

# Apply the theme
apply_theme_style()

# Game functions
def initialize_level(level):
    game_state = st.session_state.game_state
    game_state.bird_position = [50, 200]
    game_state.bird_velocity = [0, 0]
    game_state.is_launched = False
    game_state.game_over = False
    game_state.level_complete = False
    
    if level == 1:
        game_state.target_position = [400, 200]
        game_state.obstacles = [
            {"x": 300, "y": 150, "width": 20, "height": 100},
            {"x": 350, "y": 200, "width": 20, "height": 100}
        ]
    elif level == 2:
        game_state.target_position = [500, 150]
        game_state.obstacles = [
            {"x": 300, "y": 100, "width": 20, "height": 150},
            {"x": 400, "y": 200, "width": 20, "height": 100},
            {"x": 450, "y": 150, "width": 20, "height": 150}
        ]

def update_game_state():
    game_state = st.session_state.game_state
    if game_state.is_launched:
        # Update bird position based on velocity
        game_state.bird_position[0] += game_state.bird_velocity[0]
        game_state.bird_position[1] += game_state.bird_velocity[1]
        
        # Apply gravity
        game_state.bird_velocity[1] += 0.5
        
        # Check collisions
        check_collisions()
        
        # Check if bird is out of bounds
        if (game_state.bird_position[0] > 600 or 
            game_state.bird_position[0] < 0 or 
            game_state.bird_position[1] > 400 or 
            game_state.bird_position[1] < 0):
            game_state.game_over = True

def check_collisions():
    game_state = st.session_state.game_state
    bird_x, bird_y = game_state.bird_position
    
    # Check target collision
    target_x, target_y = game_state.target_position
    if (abs(bird_x - target_x) < 30 and 
        abs(bird_y - target_y) < 30):
        game_state.level_complete = True
        st.session_state.score += 100
        st.session_state.stars += 1
        if st.session_state.current_level + 1 not in st.session_state.unlocked_levels:
            st.session_state.unlocked_levels.append(st.session_state.current_level + 1)
    
    # Check obstacle collisions
    for obstacle in game_state.obstacles:
        if (bird_x > obstacle["x"] and 
            bird_x < obstacle["x"] + obstacle["width"] and
            bird_y > obstacle["y"] and 
            bird_y < obstacle["y"] + obstacle["height"]):
            game_state.game_over = True

def launch_bird():
    game_state = st.session_state.game_state
    if not game_state.is_launched:
        game_state.is_launched = True
        game_state.bird_velocity[0] = game_state.power * np.cos(np.radians(game_state.angle))
        game_state.bird_velocity[1] = -game_state.power * np.sin(np.radians(game_state.angle))

# Sidebar
with st.sidebar:
    st.markdown("### 🎨 Character Selection")
    
    # Character selection
    character = st.radio(
        "Choose your character:",
        ["Red Bird", "Blue Bird", "Yellow Bird", "Black Bird", "White Bird"],
        horizontal=True
    )
    
    if character != st.session_state.selected_character:
        st.session_state.selected_character = character
    
    # Display character preview
    st.markdown(f"""
    <div class="character-card">
        <h4 style="color: {THEMES['Angry Birds']['primary_color']}; margin-top: 0;">Your Character</h4>
        <img src="{THEMES['Angry Birds']['characters'][character.lower().split()[0]]}" 
             style="width: 100px; height: 100px; border-radius: 50%; margin: 10px auto;">
        <p style="color: {THEMES['Angry Birds']['text_color']}; 
                  font-size: 1.1rem; 
                  margin: 0;">
            {character}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stock selection
    st.markdown("### 📈 Stock Selection")
    ticker = st.text_input("Enter Stock Ticker", value=st.session_state.selected_stock)
    if ticker:
        st.session_state.selected_stock = ticker.upper()
    
    # Time period selection
    period = st.selectbox("Select Time Period",
                         ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                         index=3)
    
    # Score display
    st.markdown(f"""
    <div class="score-display">
        <p style="margin: 0;">🌟 Stars: {st.session_state.stars}</p>
        <p style="margin: 0;">🎯 Score: {st.session_state.score}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Level selection
    st.markdown("### 🎮 Select Level")
    for level in range(1, 6):  # Show 5 levels
        if level in st.session_state.unlocked_levels:
            if st.button(f"Level {level}", key=f"level_{level}"):
                st.session_state.current_level = level
                initialize_level(level)
        else:
            st.button(f"Level {level} 🔒", disabled=True, key=f"level_{level}")

# Main content
st.markdown("<h1 style='text-align: center; color: #FF6B6B;'>Angry Birds Financial Adventure</h1>", unsafe_allow_html=True)

# Stock Analysis Section
st.markdown("### 📊 Stock Analysis")
df = get_stock_data(st.session_state.selected_stock, period)
if df is not None:
    df = calculate_technical_indicators(df)
    
    # Display current stock information
    current_price = df['Close'].iloc[-1]
    price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
    percent_change = (price_change / df['Close'].iloc[-2]) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${current_price:.2f}",
                 f"{price_change:.2f} ({percent_change:.2f}%)")
    with col2:
        st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
    with col3:
        st.metric("SMA 20", f"{df['SMA_20'].iloc[-1]:.2f}")
    
    # Display stock analysis plots
    fig1, fig2, fig3 = plot_stock_analysis(df, st.session_state.selected_stock)
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Trading signals
    st.markdown("### 🎯 Trading Signals")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rsi_signal = "Oversold" if df['RSI'].iloc[-1] < 30 else "Overbought" if df['RSI'].iloc[-1] > 70 else "Neutral"
        st.metric("RSI Signal", rsi_signal)
    
    with col2:
        bb_signal = "Oversold" if df['Close'].iloc[-1] < df['BB_lower'].iloc[-1] else "Overbought" if df['Close'].iloc[-1] > df['BB_upper'].iloc[-1] else "Neutral"
        st.metric("Bollinger Bands Signal", bb_signal)
    
    with col3:
        ma_signal = "Bullish" if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] else "Bearish"
        st.metric("Moving Average Signal", ma_signal)

# Game Section
st.markdown("### 🎮 Trading Game")
st.markdown("""
<div class="game-container">
    <div class="bird" style="left: {st.session_state.game_state.bird_position[0]}px; 
                            top: {st.session_state.game_state.bird_position[1]}px;">
        <img src="{THEMES['Angry Birds']['characters'][st.session_state.selected_character.lower().split()[0]]}" 
             style="width: 100%; height: 100%;">
    </div>
    <div class="target" style="left: {st.session_state.game_state.target_position[0]}px; 
                              top: {st.session_state.game_state.target_position[1]}px;">
    </div>
</div>
""", unsafe_allow_html=True)

# Level content
if st.session_state.current_level == 1:
    st.markdown("""
    <div class="level-card">
        <h3 style="color: #FF6B6B;">Welcome to Level 1!</h3>
        <p>Help the Angry Birds rescue their eggs from the evil pigs!</p>
        <div style="text-align: center; margin: 20px 0;">
            <img src="https://media.giphy.com/media/3o7TKz2eMXx7dn95FS/giphy.gif" 
                 style="width: 200px; border-radius: 10px;">
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Level 1 game content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Objective")
        st.markdown("""
        - Launch your bird to hit the target
        - Score points for each successful hit
        - Collect stars to unlock new levels
        """)
    
    with col2:
        st.markdown("### 🎮 Controls")
        st.markdown("""
        - Adjust power and angle
        - Click Launch to send your bird flying
        - Avoid obstacles and hit the target
        """)

elif st.session_state.current_level == 2:
    st.markdown("""
    <div class="level-card">
        <h3 style="color: #4ECDC4;">Level 2: The Mighty Hoax</h3>
        <p>Face new challenges and defeat the mighty king pig!</p>
        <div style="text-align: center; margin: 20px 0;">
            <img src="https://media.giphy.com/media/3o7TKz2eMXx7dn95FS/giphy.gif" 
                 style="width: 200px; border-radius: 10px;">
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Level 2 game content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 New Challenges")
        st.markdown("""
        - More complex structures
        - New enemy types
        - Special power-ups
        """)
    
    with col2:
        st.markdown("### 💪 Tips")
        st.markdown("""
        - Use the Blue Bird's splitting ability
        - Time your shots carefully
        - Look for weak points
        """)

# Game controls
if not st.session_state.game_state.is_launched and not st.session_state.game_state.game_over:
    st.session_state.game_state.power = st.slider("Investment Amount", 0, 100, 50)
    st.session_state.game_state.angle = st.slider("Risk Level", 0, 90, 45)
    
    if st.button("Make Trade!", use_container_width=True):
        launch_bird()

# Game state update
if st.session_state.game_state.is_launched:
    update_game_state()
    time.sleep(0.1)
    st.experimental_rerun()

# Game messages
if st.session_state.game_state.game_over:
    st.error("Game Over! Try again!")
    if st.button("Restart Level", use_container_width=True):
        initialize_level(st.session_state.current_level)
        st.experimental_rerun()

if st.session_state.game_state.level_complete:
    st.success(f"Level {st.session_state.current_level} completed! 🌟")
    if st.button("Next Level", use_container_width=True):
        st.session_state.current_level += 1
        initialize_level(st.session_state.current_level)
        st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Created with ❤️ by Angry Birds Financial Team</p>
    <p>© 2024 Angry Birds Financial Adventure</p>
</div>
""", unsafe_allow_html=True)
