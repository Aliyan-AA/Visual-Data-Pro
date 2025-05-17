import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

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
        - Click and drag to aim
        - Release to launch
        - Use special powers when available
        """)
    
    # Game interface
    st.markdown("### 🎮 Play Now!")
    if st.button("Start Level 1", use_container_width=True):
        st.session_state.score += 100
        st.session_state.stars += 1
        if 2 not in st.session_state.unlocked_levels:
            st.session_state.unlocked_levels.append(2)
        st.success("Level 1 completed! Level 2 unlocked! 🌟")

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
    
    # Game interface
    st.markdown("### 🎮 Play Now!")
    if st.button("Start Level 2", use_container_width=True):
        st.session_state.score += 200
        st.session_state.stars += 2
        if 3 not in st.session_state.unlocked_levels:
            st.session_state.unlocked_levels.append(3)
        st.success("Level 2 completed! Level 3 unlocked! 🌟🌟")

# Add more levels as needed...

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Created with ❤️ by Angry Birds Learning Team</p>
    <p>© 2024 Angry Birds Learning Adventure</p>
</div>
""", unsafe_allow_html=True)
