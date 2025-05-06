import streamlit as st
import pandas as pd
import yfinance as yf
import time
import requests
from typing import Dict, Any, Optional

def load_animation_url(animation_type: str) -> str:
    """
    Return URL for a finance-related GIF animation based on type.
    
    Parameters:
    -----------
    animation_type : str
        Type of animation to load ('finance', 'celebrate', etc.)
        
    Returns:
    --------
    str
        URL to the animation
    """
    if not isinstance(animation_type, str):
        raise ValueError("animation_type must be a string")
    
    animation_urls: Dict[str, str] = {
        'finance': "https://media.giphy.com/media/JtBZm3Getg3dqxK0zP/giphy.gif",
        'celebrate': "https://media.giphy.com/media/lMVNl6XxTvXgs/giphy.gif",
        'loading': "https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif",
        'stock_market': "https://media.giphy.com/media/JtBZm3Getg3dqxK0zP/giphy.gif",
        'success': "https://media.giphy.com/media/g9582DNuQppxC/giphy.gif",
        'error': "https://media.giphy.com/media/TqiwHbFBaZ4ti/giphy.gif"
    }
    
    if animation_type not in animation_urls:
        st.warning(f"Unknown animation type: {animation_type}. Using default finance animation.")
        return animation_urls['finance']
    
    return animation_urls[animation_type]

def validate_stock_ticker(ticker: str) -> bool:
    """
    Validate if a stock ticker exists in Yahoo Finance.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
        
    Returns:
    --------
    bool
        True if valid, False otherwise
    """
    if not isinstance(ticker, str):
        raise ValueError("ticker must be a string")
    
    if not ticker.strip():
        raise ValueError("ticker cannot be empty")
    
    try:
        ticker_info = yf.Ticker(ticker).info
        return 'regularMarketPrice' in ticker_info and ticker_info['regularMarketPrice'] is not None
    except Exception as e:
        st.error(f"Error validating ticker {ticker}: {str(e)}")
        return False

def show_notification(notification_type: str, message: str) -> None:
    """
    Display a notification message in Streamlit.
    
    Parameters:
    -----------
    notification_type : str
        Type of notification ('success', 'info', 'warning', 'error')
    message : str
        Message to display
    """
    if not isinstance(notification_type, str) or not isinstance(message, str):
        raise ValueError("notification_type and message must be strings")
    
    if not message.strip():
        raise ValueError("message cannot be empty")
    
    valid_types = ['success', 'info', 'warning', 'error']
    if notification_type not in valid_types:
        raise ValueError(f"notification_type must be one of {valid_types}")
    
    try:
        if notification_type == 'success':
            st.success(message)
        elif notification_type == 'info':
            st.info(message)
        elif notification_type == 'warning':
            st.warning(message)
        elif notification_type == 'error':
            st.error(message)
    except Exception as e:
        st.write(f"Error displaying notification: {str(e)}")
        st.write(message)  # Fallback to basic message display
