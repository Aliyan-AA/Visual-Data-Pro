import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import random

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

# Set page configuration
st.set_page_config(
    page_title="Angry Birds Learning Adventure",
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
        
        .bird {{
            position: absolute;
            width: 50px;
            height: 50px;
            transition: all 0.1s linear;
        }}
        
        .target {{
            position: absolute;
            width: 30px;
            height: 30px;
            background-color: {theme_config['accent_color']};
            border-radius: 50%;
        }}
        
        .obstacle {{
            position: absolute;
            background-color: {theme_config['secondary_color']};
            border-radius: 5px;
        }}
        
        .power-meter {{
            width: 100%;
            height: 20px;
            background-color: #ddd;
            border-radius: 10px;
            margin: 10px 0;
        }}
        
        .power-fill {{
            height: 100%;
            background-color: {theme_config['primary_color']};
            border-radius: 10px;
            transition: width 0.1s linear;
        }}
        
        .angle-indicator {{
            position: absolute;
            width: 2px;
            background-color: {theme_config['text_color']};
            transform-origin: bottom center;
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
st.markdown("<h1 style='text-align: center; color: #FF6B6B;'>Angry Birds Learning Adventure</h1>", unsafe_allow_html=True)

# Level display
st.markdown(f"""
<div style="text-align: center; margin: 20px 0;">
    <h2 style="color: {THEMES['Angry Birds']['primary_color']};">Level {st.session_state.current_level}</h2>
</div>
""", unsafe_allow_html=True)

# Game interface
st.markdown(f"""
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
    st.session_state.game_state.power = st.slider("Power", 0, 100, 50)
    st.session_state.game_state.angle = st.slider("Angle", 0, 90, 45)
    
    if st.button("Launch!", use_container_width=True):
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
    <p>Created with ❤️ by Angry Birds Learning Team</p>
    <p>© 2024 Angry Birds Learning Adventure</p>
</div>
""", unsafe_allow_html=True)
