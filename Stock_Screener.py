import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class TechnicalIndicators:
    """Custom technical indicators without TA-Lib dependency"""
    
    @staticmethod
    def sma(data, window):
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data, window):
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data, window=14):
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """MACD Indicator"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data, window=20, num_std=2):
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band
    
    @staticmethod
    def stochastic(high, low, close, k_window=14, d_window=3):
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    @staticmethod
    def atr(high, low, close, window=14):
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr
    
    @staticmethod
    def adx(high, low, close, window=14):
        """Average Directional Index"""
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()
        
        tr = TechnicalIndicators.atr(high, low, close, 1)
        plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / tr.ewm(alpha=1/window).mean())
        minus_di = 100 * (minus_dm.ewm(alpha=1/window).mean() / tr.ewm(alpha=1/window).mean())
        
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = dx.ewm(alpha=1/window).mean()
        return adx, plus_di, minus_di
    
    @staticmethod
    def detect_support_resistance(data, n1=2, n2=2, backcandles=40):
        """
        Detect support and resistance levels using local minima/maxima
        """
        def is_support(df, i, n1, n2):
            """Check if candle at index i is a support level"""
            if i < n1 or i >= len(df) - n2:
                return False
            
            current_low = df['Low'].iloc[i]
            
            # Check if current low is lower than surrounding lows
            for j in range(i - n1, i + n2 + 1):
                if j != i and df['Low'].iloc[j] < current_low:
                    return False
            
            # Check for significant lower wick
            candle_body = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
            lower_wick = min(df['Open'].iloc[i], df['Close'].iloc[i]) - df['Low'].iloc[i]
            
            return lower_wick > candle_body * 0.1
        
        def is_resistance(df, i, n1, n2):
            """Check if candle at index i is a resistance level"""
            if i < n1 or i >= len(df) - n2:
                return False
            
            current_high = df['High'].iloc[i]
            
            # Check if current high is higher than surrounding highs
            for j in range(i - n1, i + n2 + 1):
                if j != i and df['High'].iloc[j] > current_high:
                    return False
            
            # Check for significant upper wick
            candle_body = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
            upper_wick = df['High'].iloc[i] - max(df['Open'].iloc[i], df['Close'].iloc[i])
            
            return upper_wick > candle_body * 0.1
        
        # Find support and resistance levels
        support_levels = []
        resistance_levels = []
        
        start_idx = max(n1, len(data) - backcandles)
        end_idx = len(data) - n2
        
        for i in range(start_idx, end_idx):
            if is_support(data, i, n1, n2):
                support_levels.append(data['Low'].iloc[i])
            if is_resistance(data, i, n1, n2):
                resistance_levels.append(data['High'].iloc[i])
        
        # Remove duplicate levels that are too close
        def merge_close_levels(levels, threshold=0.001):
            if not levels:
                return []
            
            levels = sorted(levels)
            merged = [levels[0]]
            
            for level in levels[1:]:
                if abs(level - merged[-1]) / merged[-1] > threshold:
                    merged.append(level)
            
            return merged
        
        support_levels = merge_close_levels(support_levels)
        resistance_levels = merge_close_levels(resistance_levels)
        
        return support_levels, resistance_levels
    
    @staticmethod
    def calculate_pivot_points(data):
        """Calculate pivot points and support/resistance levels"""
        if len(data) < 1:
            return None
        
        # Use the last complete day's data
        high = data['High'].iloc[-1]
        low = data['Low'].iloc[-1]
        close = data['Close'].iloc[-1]
        
        # Calculate pivot point
        pivot_point = (high + low + close) / 3
        
        # Calculate support and resistance levels
        support_1 = (pivot_point * 2) - high
        support_2 = pivot_point - (high - low)
        resistance_1 = (pivot_point * 2) - low
        resistance_2 = pivot_point + (high - low)
        
        return {
            'pivot_point': pivot_point,
            'support_1': support_1,
            'support_2': support_2,
            'resistance_1': resistance_1,
            'resistance_2': resistance_2
        }
    
    @staticmethod
    def check_level_proximity(current_price, levels, threshold=0.02):
        """Check if current price is near any support/resistance level"""
        for level in levels:
            if abs(current_price - level) / current_price <= threshold:
                return level, abs(current_price - level) / current_price
        return None, None
    
    @staticmethod
    def detect_macd_divergence(data, macd_line, lookback_periods=20):
        """
        Detect MACD divergence patterns
        """
        def find_peaks_and_troughs(series, window=5):
            """Find local peaks and troughs in a series"""
            peaks = []
            troughs = []
            
            for i in range(window, len(series) - window):
                # Check for peak
                if all(series.iloc[i] >= series.iloc[i-j] for j in range(1, window+1)) and \
                   all(series.iloc[i] >= series.iloc[i+j] for j in range(1, window+1)):
                    peaks.append((i, series.iloc[i]))
                
                # Check for trough
                if all(series.iloc[i] <= series.iloc[i-j] for j in range(1, window+1)) and \
                   all(series.iloc[i] <= series.iloc[i+j] for j in range(1, window+1)):
                    troughs.append((i, series.iloc[i]))
            
            return peaks, troughs
        
        # Get recent data for analysis
        recent_data = data.tail(lookback_periods)
        recent_macd = macd_line.tail(lookback_periods)
        
        # Find peaks and troughs in price and MACD
        price_peaks, price_troughs = find_peaks_and_troughs(recent_data['Close'])
        macd_peaks, macd_troughs = find_peaks_and_troughs(recent_macd)
        
        divergences = {
            'bullish_divergence': False,
            'bearish_divergence': False,
            'hidden_bullish': False,
            'hidden_bearish': False,
            'divergence_strength': 0
        }
        
        # Check for regular bullish divergence
        if len(price_troughs) >= 2 and len(macd_troughs) >= 2:
            latest_price_trough = price_troughs[-1]
            prev_price_trough = price_troughs[-2]
            latest_macd_trough = macd_troughs[-1]
            prev_macd_trough = macd_troughs[-2]
            
            # Price making lower lows, MACD making higher lows
            if (latest_price_trough[1] < prev_price_trough[1] and 
                latest_macd_trough[1] > prev_macd_trough[1]):
                divergences['bullish_divergence'] = True
                divergences['divergence_strength'] += 2
        
        # Check for regular bearish divergence
        if len(price_peaks) >= 2 and len(macd_peaks) >= 2:
            latest_price_peak = price_peaks[-1]
            prev_price_peak = price_peaks[-2]
            latest_macd_peak = macd_peaks[-1]
            prev_macd_peak = macd_peaks[-2]
            
            # Price making higher highs, MACD making lower highs
            if (latest_price_peak[1] > prev_price_peak[1] and 
                latest_macd_peak[1] < prev_macd_peak[1]):
                divergences['bearish_divergence'] = True
                divergences['divergence_strength'] -= 2
        
        # Check for hidden divergences (trend continuation signals)
        if len(price_troughs) >= 2 and len(macd_troughs) >= 2:
            latest_price_trough = price_troughs[-1]
            prev_price_trough = price_troughs[-2]
            latest_macd_trough = macd_troughs[-1]
            prev_macd_trough = macd_troughs[-2]
            
            # Hidden bullish: Price higher lows, MACD lower lows (uptrend continuation)
            if (latest_price_trough[1] > prev_price_trough[1] and 
                latest_macd_trough[1] < prev_macd_trough[1]):
                divergences['hidden_bullish'] = True
                divergences['divergence_strength'] += 1
        
        if len(price_peaks) >= 2 and len(macd_peaks) >= 2:
            latest_price_peak = price_peaks[-1]
            prev_price_peak = price_peaks[-2]
            latest_macd_peak = macd_peaks[-1]
            prev_macd_peak = macd_peaks[-2]
            
            # Hidden bearish: Price lower highs, MACD higher highs (downtrend continuation)
            if (latest_price_peak[1] < prev_price_peak[1] and 
                latest_macd_peak[1] > prev_macd_peak[1]):
                divergences['hidden_bearish'] = True
                divergences['divergence_strength'] -= 1
        
        return divergences
    
    @staticmethod
    def enhanced_macd_signals(data, macd_line, signal_line, histogram):
        """
        Enhanced MACD signal generation including divergence analysis
        """
        signals = {
            'crossover_signal': 0,
            'histogram_signal': 0,
            'zero_line_signal': 0,
            'divergence_signal': 0,
            'overall_macd_signal': 0
        }
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        prev_signal = signal_line.iloc[-2]
        current_histogram = histogram.iloc[-1]
        
        # 1. Crossover signals
        if current_macd > current_signal and prev_macd <= prev_signal:
            signals['crossover_signal'] = 1  # Bullish crossover
        elif current_macd < current_signal and prev_macd >= prev_signal:
            signals['crossover_signal'] = -1  # Bearish crossover
        
        # 2. Histogram signals (momentum strength)
        if current_histogram > 0:
            signals['histogram_signal'] = 1  # Above zero - bullish momentum
        else:
            signals['histogram_signal'] = -1  # Below zero - bearish momentum
        
        # 3. Zero line crossover
        if current_macd > 0:
            signals['zero_line_signal'] = 1  # Above zero line - bullish
        else:
            signals['zero_line_signal'] = -1  # Below zero line - bearish
        
        # 4. Divergence analysis
        divergence_data = TechnicalIndicators.detect_macd_divergence(data, macd_line)
        
        if divergence_data['bullish_divergence']:
            signals['divergence_signal'] = 2  # Strong bullish signal
        elif divergence_data['bearish_divergence']:
            signals['divergence_signal'] = -2  # Strong bearish signal
        elif divergence_data['hidden_bullish']:
            signals['divergence_signal'] = 1  # Moderate bullish (trend continuation)
        elif divergence_data['hidden_bearish']:
            signals['divergence_signal'] = -1  # Moderate bearish (trend continuation)
        
        # 5. Overall MACD signal (weighted combination)
        signals['overall_macd_signal'] = (
            signals['crossover_signal'] * 2 +  # Crossovers get double weight
            signals['histogram_signal'] * 1 +
            signals['zero_line_signal'] * 1 +
            signals['divergence_signal'] * 2   # Divergences get double weight
        )
        
        # Store divergence details for display
        signals['divergence_details'] = divergence_data
        
        return signals

class StockAnalyzer:
    def __init__(self, symbol, period="1y"):
        self.symbol = symbol.upper()
        self.period = period
        self.data = None
        self.info = None
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            self.info = ticker.info
            
            if self.data.empty:
                st.error(f"No data found for symbol {self.symbol}")
                return False
            return True
        except Exception as e:
            st.error(f"Error fetching data for {self.symbol}: {str(e)}")
            return False
    
    def calculate_technical_indicators(self):
        """Calculate various technical indicators using custom functions"""
        if self.data is None or len(self.data) < 50:
            return None
            
        indicators = {}
        
        # Price data
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        volume = self.data['Volume']
        
        # Moving Averages
        indicators['SMA_20'] = TechnicalIndicators.sma(close, 20)
        indicators['SMA_50'] = TechnicalIndicators.sma(close, 50)
        indicators['EMA_12'] = TechnicalIndicators.ema(close, 12)
        indicators['EMA_26'] = TechnicalIndicators.ema(close, 26)
        
        # RSI
        indicators['RSI'] = TechnicalIndicators.rsi(close)
        
        # MACD
        macd, signal, histogram = TechnicalIndicators.macd(close)
        indicators['MACD'] = macd
        indicators['MACD_signal'] = signal
        indicators['MACD_histogram'] = histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(close)
        indicators['BB_upper'] = bb_upper
        indicators['BB_middle'] = bb_middle
        indicators['BB_lower'] = bb_lower
        
        # Stochastic
        stoch_k, stoch_d = TechnicalIndicators.stochastic(high, low, close)
        indicators['STOCH_K'] = stoch_k
        indicators['STOCH_D'] = stoch_d
        
        # ATR
        indicators['ATR'] = TechnicalIndicators.atr(high, low, close)
        
        # ADX
        adx, plus_di, minus_di = TechnicalIndicators.adx(high, low, close)
        indicators['ADX'] = adx
        indicators['PLUS_DI'] = plus_di
        indicators['MINUS_DI'] = minus_di
        
        # Support and Resistance Detection
        support_levels, resistance_levels = TechnicalIndicators.detect_support_resistance(self.data)
        indicators['support_levels'] = support_levels
        indicators['resistance_levels'] = resistance_levels
        
        # Pivot Points
        pivot_data = TechnicalIndicators.calculate_pivot_points(self.data)
        if pivot_data:
            indicators.update(pivot_data)
        
        return indicators
    
    def generate_technical_signals(self, indicators):
        """Generate buy/sell signals based on technical indicators"""
        signals = {}
        current_price = self.data['Close'].iloc[-1]
        
        # Moving Average signals
        sma_20 = indicators['SMA_20'].iloc[-1]
        sma_50 = indicators['SMA_50'].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            signals['MA_signal'] = 1  # Buy
        elif current_price < sma_20 < sma_50:
            signals['MA_signal'] = -1  # Sell
        else:
            signals['MA_signal'] = 0  # Neutral
        
        # RSI signals
        rsi = indicators['RSI'].iloc[-1]
        if rsi < 30:
            signals['RSI_signal'] = 1  # Oversold - Buy
        elif rsi > 70:
            signals['RSI_signal'] = -1  # Overbought - Sell
        else:
            signals['RSI_signal'] = 0  # Neutral
        
        # Enhanced MACD signals with divergence analysis
        macd_signals = TechnicalIndicators.enhanced_macd_signals(
            self.data, 
            indicators['MACD'], 
            indicators['MACD_signal'], 
            indicators['MACD_histogram']
        )
        
        # Use the overall MACD signal instead of just crossover
        if macd_signals['overall_macd_signal'] >= 2:
            signals['MACD_signal'] = 1  # Strong buy
        elif macd_signals['overall_macd_signal'] <= -2:
            signals['MACD_signal'] = -1  # Strong sell
        else:
            signals['MACD_signal'] = 0  # Neutral
        
        # Store detailed MACD analysis for display
        signals['macd_details'] = macd_signals
        
        # Bollinger Bands signals
        bb_upper = indicators['BB_upper'].iloc[-1]
        bb_lower = indicators['BB_lower'].iloc[-1]
        
        if current_price <= bb_lower:
            signals['BB_signal'] = 1  # Oversold - Buy
        elif current_price >= bb_upper:
            signals['BB_signal'] = -1  # Overbought - Sell
        else:
            signals['BB_signal'] = 0  # Neutral
        
        # Stochastic signals
        stoch_k = indicators['STOCH_K'].iloc[-1]
        stoch_d = indicators['STOCH_D'].iloc[-1]
        
        if stoch_k < 20 and stoch_d < 20:
            signals['STOCH_signal'] = 1  # Oversold - Buy
        elif stoch_k > 80 and stoch_d > 80:
            signals['STOCH_signal'] = -1  # Overbought - Sell
        else:
            signals['STOCH_signal'] = 0  # Neutral
        
        # Support and Resistance Signals
        support_levels = indicators.get('support_levels', [])
        resistance_levels = indicators.get('resistance_levels', [])
        
        # Check proximity to support/resistance
        near_support, support_distance = TechnicalIndicators.check_level_proximity(
            current_price, support_levels, threshold=0.02
        )
        near_resistance, resistance_distance = TechnicalIndicators.check_level_proximity(
            current_price, resistance_levels, threshold=0.02
        )
        
        signals['SR_signal'] = 0
        if near_support:
            signals['SR_signal'] = 1  # Buy signal near support
            signals['near_support'] = near_support
            signals['support_distance'] = support_distance
        elif near_resistance:
            signals['SR_signal'] = -1  # Sell signal near resistance
            signals['near_resistance'] = near_resistance
            signals['resistance_distance'] = resistance_distance
        
        # Pivot point signals
        pivot_point = indicators.get('pivot_point', current_price)
        if current_price > pivot_point:
            signals['pivot_signal'] = 1  # Above pivot - bullish
        else:
            signals['pivot_signal'] = -1  # Below pivot - bearish
        
        # ADX trend strength signal
        adx_value = indicators['ADX'].iloc[-1]
        plus_di = indicators['PLUS_DI'].iloc[-1]
        minus_di = indicators['MINUS_DI'].iloc[-1]
        
        if adx_value > 25:  # Strong trend
            if plus_di > minus_di:
                signals['ADX_signal'] = 1  # Strong uptrend
            else:
                signals['ADX_signal'] = -1  # Strong downtrend
        else:
            signals['ADX_signal'] = 0  # No clear trend
        
        return signals

def create_price_chart(data, indicators):
    """Create comprehensive price chart with technical indicators"""
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=('Price, Moving Averages & Support/Resistance', 'MACD', 'RSI', 'ADX', 'Volume'),
        row_heights=[0.35, 0.2, 0.15, 0.15, 0.15]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Moving averages
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['BB_upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['BB_lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add Support and Resistance Levels
    support_levels = indicators.get('support_levels', [])
    resistance_levels = indicators.get('resistance_levels', [])
    
    # Plot support levels
    for level in support_levels:
        fig.add_hline(
            y=level, 
            line_dash="dot", 
            line_color="green", 
            line_width=2,
            annotation_text=f"Support: ${level:.2f}",
            annotation_position="bottom right",
            row=1, col=1
        )
    
    # Plot resistance levels
    for level in resistance_levels:
        fig.add_hline(
            y=level, 
            line_dash="dot", 
            line_color="red", 
            line_width=2,
            annotation_text=f"Resistance: ${level:.2f}",
            annotation_position="top right",
            row=1, col=1
        )
    
    # Add pivot points if available
    if 'pivot_point' in indicators:
        pivot_point = indicators['pivot_point']
        fig.add_hline(
            y=pivot_point, 
            line_dash="dash", 
            line_color="yellow", 
            line_width=3,
            annotation_text=f"Pivot: ${pivot_point:.2f}",
            annotation_position="top left",
            row=1, col=1
        )
        
        # Add pivot support and resistance levels
        for level_name, level_value in [
            ('S1', indicators.get('support_1')),
            ('S2', indicators.get('support_2')),
            ('R1', indicators.get('resistance_1')),
            ('R2', indicators.get('resistance_2'))
        ]:
            if level_value:
                color = "lightgreen" if level_name.startswith('S') else "lightcoral"
                fig.add_hline(
                    y=level_value, 
                    line_dash="dashdot", 
                    line_color=color, 
                    line_width=1,
                    annotation_text=f"{level_name}: ${level_value:.2f}",
                    annotation_position="bottom left" if level_name.startswith('S') else "top left",
                    row=1, col=1
                )
    
    # MACD
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['MACD_signal'],
            mode='lines',
            name='MACD Signal',
            line=dict(color='red', width=2)
        ),
        row=2, col=1
    )
    
    # MACD Histogram
    colors = ['green' if val >= 0 else 'red' for val in indicators['MACD_histogram']]
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=indicators['MACD_histogram'],
            name='MACD Histogram',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Add zero line for MACD
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2)
        ),
        row=3, col=1
    )
    
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
    
    # ADX
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['ADX'],
            mode='lines',
            name='ADX',
            line=dict(color='black', width=2)
        ),
        row=4, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['PLUS_DI'],
            mode='lines',
            name='+DI',
            line=dict(color='green', width=1)
        ),
        row=4, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['MINUS_DI'],
            mode='lines',
            name='-DI',
            line=dict(color='red', width=1)
        ),
        row=4, col=1
    )
    
    # ADX trend strength levels
    fig.add_hline(y=25, line_dash="dash", line_color="orange", row=4, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="red", row=4, col=1)
    
    # Volume
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color='lightblue',
            opacity=0.7
        ),
        row=5, col=1
    )
    
    # Update layout
    fig.update_layout(
        title="Advanced Technical Analysis Dashboard",
        xaxis_rangeslider_visible=False,
        height=1000,
        showlegend=True,
        template="plotly_dark"
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="ADX", row=4, col=1)
    fig.update_yaxes(title_text="Volume", row=5, col=1)
    
    return fig

def calculate_position_sizing(current_price, atr, portfolio_value=10000, risk_percent=0.01):
    """Calculate recommended position size based on risk management"""
    risk_amount = portfolio_value * risk_percent
    stop_loss_distance = atr * 2  # 2x ATR for stop loss
    stop_loss_price = current_price - stop_loss_distance
    
    # Calculate shares based on risk amount
    if stop_loss_distance > 0:
        shares = int(risk_amount / stop_loss_distance)
        position_value = shares * current_price
    else:
        shares = 0
        position_value = 0
    
    return {
        'shares': shares,
        'position_value': position_value,
        'risk_amount': risk_amount,
        'stop_loss_price': stop_loss_price,
        'stop_loss_distance': stop_loss_distance
    }

def get_news_sentiment(symbol):
    """Get news sentiment for the stock (placeholder function)"""
    news_items = [
        {
            "title": f"Market Analysis: {symbol} shows strong fundamentals",
            "sentiment": "Positive",
            "date": "2024-01-15"
        },
        {
            "title": f"{symbol} quarterly earnings beat expectations",
            "sentiment": "Positive", 
            "date": "2024-01-14"
        },
        {
            "title": f"Analysts upgrade {symbol} price target",
            "sentiment": "Positive",
            "date": "2024-01-13"
        }
    ]
    return news_items

def main():
    st.title("📈 Advanced Stock Analysis Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📊 Analysis Settings")
    
    # Stock symbol input
    symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL", help="Enter a valid stock ticker symbol")
    
    # Time period selection
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo", 
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y"
    }
    
    selected_period = st.sidebar.selectbox("Select Time Period", list(period_options.keys()), index=3)
    period = period_options[selected_period]
    
    # Portfolio settings for position sizing
    st.sidebar.subheader("💰 Portfolio Settings")
    portfolio_value = st.sidebar.number_input("Portfolio Value ($)", value=10000, min_value=1000, step=1000)
    risk_percent = st.sidebar.slider("Risk per Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1) / 100
    
    # Analysis type
    analysis_type = st.sidebar.multiselect(
        "Select Analysis Type",
        ["Technical Analysis", "Position Sizing", "Fundamental Analysis", "News Sentiment"],
        default=["Technical Analysis", "Position Sizing"]
    )
    
    if st.sidebar.button("🔍 Analyze Stock", type="primary"):
        if symbol:
            with st.spinner(f"Analyzing {symbol}..."):
                # Initialize analyzer
                analyzer = StockAnalyzer(symbol, period)
                
                # Fetch data
                if analyzer.fetch_data():
                    # Display basic info
                    col1, col2, col3, col4 = st.columns(4)
                    
                    current_price = analyzer.data['Close'].iloc[-1]
                    prev_close = analyzer.data['Close'].iloc[-2]
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
                    
                    with col2:
                        st.metric("Volume", f"{analyzer.data['Volume'].iloc[-1]:,.0f}")
                    
                    with col3:
                        high_52w = analyzer.data['High'].max()
                        low_52w = analyzer.data['Low'].min()
                        st.metric("52W High", f"${high_52w:.2f}")
                    
                    with col4:
                        st.metric("52W Low", f"${low_52w:.2f}")
                    
                    # Technical Analysis
                    if "Technical Analysis" in analysis_type:
                        st.markdown("## 🔧 Technical Analysis")
                        
                        # Calculate indicators
                        indicators = analyzer.calculate_technical_indicators()
                        
                        if indicators:
                            # Generate signals
                            tech_signals = analyzer.generate_technical_signals(indicators)
                            
                            # Create chart
                            fig = create_price_chart(analyzer.data, indicators)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Technical indicators table
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("📊 Technical Indicators")
                                
                                # Get MACD details for enhanced display
                                macd_details = tech_signals.get('macd_details', {})
                                divergence_info = macd_details.get('divergence_details', {})
                                
                                # Create more detailed MACD description
                                macd_description = f"MACD: {indicators['MACD'].iloc[-1]:.3f}"
                                if divergence_info.get('bullish_divergence'):
                                    macd_description += " (Bullish Divergence)"
                                elif divergence_info.get('bearish_divergence'):
                                    macd_description += " (Bearish Divergence)"
                                elif divergence_info.get('hidden_bullish'):
                                    macd_description += " (Hidden Bullish)"
                                elif divergence_info.get('hidden_bearish'):
                                    macd_description += " (Hidden Bearish)"
                                
                                # Create technical indicators dataframe
                                tech_df = pd.DataFrame({
                                    'Indicator': ['Moving Averages', 'MACD', 'RSI', 'Bollinger Bands', 'Stochastic', 'Support/Resistance', 'Pivot Point', 'ADX'],
                                    'Signal': [
                                        '🟢 Buy' if tech_signals.get('MA_signal', 0) > 0 else '🔴 Sell' if tech_signals.get('MA_signal', 0) < 0 else '🟡 Neutral',
                                        '🟢 Buy' if tech_signals.get('MACD_signal', 0) > 0 else '🔴 Sell' if tech_signals.get('MACD_signal', 0) < 0 else '🟡 Neutral',
                                        '🟢 Buy' if tech_signals.get('RSI_signal', 0) > 0 else '🔴 Sell' if tech_signals.get('RSI_signal', 0) < 0 else '🟡 Neutral',
                                        '🟢 Buy' if tech_signals.get('BB_signal', 0) > 0 else '🔴 Sell' if tech_signals.get('BB_signal', 0) < 0 else '🟡 Neutral',
                                        '🟢 Buy' if tech_signals.get('STOCH_signal', 0) > 0 else '🔴 Sell' if tech_signals.get('STOCH_signal', 0) < 0 else '🟡 Neutral',
                                        '🟢 Buy' if tech_signals.get('SR_signal', 0) > 0 else '🔴 Sell' if tech_signals.get('SR_signal', 0) < 0 else '🟡 Neutral',
                                        '🟢 Bullish' if tech_signals.get('pivot_signal', 0) > 0 else '🔴 Bearish' if tech_signals.get('pivot_signal', 0) < 0 else '🟡 Neutral',
                                        '🟢 Strong Trend' if tech_signals.get('ADX_signal', 0) > 0 else '🔴 Strong Trend' if tech_signals.get('ADX_signal', 0) < 0 else '🟡 No Trend'
                                    ],
                                    'Value': [
                                        f"SMA20: ${indicators['SMA_20'].iloc[-1]:.2f}" if not pd.isna(indicators['SMA_20'].iloc[-1]) else "N/A",
                                        macd_description,
                                        f"RSI: {indicators['RSI'].iloc[-1]:.1f}" if not pd.isna(indicators['RSI'].iloc[-1]) else "N/A",
                                        f"Price vs BB: {'Upper' if current_price >= indicators['BB_upper'].iloc[-1] else 'Lower' if current_price <= indicators['BB_lower'].iloc[-1] else 'Middle'}",
                                        f"Stoch %K: {indicators['STOCH_K'].iloc[-1]:.1f}" if not pd.isna(indicators['STOCH_K'].iloc[-1]) else "N/A",
                                        f"Near: ${tech_signals.get('near_support', tech_signals.get('near_resistance', 'None')):.2f}" if tech_signals.get('near_support') or tech_signals.get('near_resistance') else "No levels nearby",
                                        f"Pivot: ${indicators.get('pivot_point', 0):.2f}" if indicators.get('pivot_point') else "N/A",
                                        f"ADX: {indicators['ADX'].iloc[-1]:.1f}" if not pd.isna(indicators['ADX'].iloc[-1]) else "N/A"
                                    ]
                                })
                                
                                st.dataframe(tech_df, use_container_width=True)
                            
                            with col2:
                                st.subheader("🎯 Trading Signal")
                                
                                # Overall signal calculation
                                tech_score = sum([
                                    tech_signals.get('MA_signal', 0),
                                    tech_signals.get('MACD_signal', 0),
                                    tech_signals.get('RSI_signal', 0),
                                    tech_signals.get('BB_signal', 0),
                                    tech_signals.get('STOCH_signal', 0),
                                    tech_signals.get('SR_signal', 0),
                                    tech_signals.get('pivot_signal', 0),
                                    tech_signals.get('ADX_signal', 0)
                                ])
                                
                                if tech_score >= 4:
                                    signal_color = "green"
                                    signal_text = "🟢 STRONG BUY"
                                elif tech_score >= 2:
                                    signal_color = "lightgreen"
                                    signal_text = "🟢 BUY"
                                elif tech_score <= -4:
                                    signal_color = "red"
                                    signal_text = "🔴 STRONG SELL"
                                elif tech_score <= -2:
                                    signal_color = "lightcoral"
                                    signal_text = "🔴 SELL"
                                else:
                                    signal_color = "yellow"
                                    signal_text = "🟡 NEUTRAL"
                                
                                st.markdown(f"<h2 style='color: {signal_color}'>{signal_text}</h2>", unsafe_allow_html=True)
                                st.markdown(f"**Signal Strength:** {tech_score}/8")
                                
                                # Signal reasoning
                                st.subheader("📝 Signal Analysis")
                                reasoning = []
                                
                                if tech_signals.get('MA_signal', 0) > 0:
                                    reasoning.append("✅ **Moving Averages**: Price above short-term MA, bullish trend")
                                elif tech_signals.get('MA_signal', 0) < 0:
                                    reasoning.append("❌ **Moving Averages**: Price below short-term MA, bearish trend")
                                
                                if tech_signals.get('RSI_signal', 0) > 0:
                                    reasoning.append("✅ **RSI**: Oversold condition, potential bounce")
                                elif tech_signals.get('RSI_signal', 0) < 0:
                                    reasoning.append("❌ **RSI**: Overbought condition, potential correction")
                                
                                # Enhanced MACD reasoning
                                macd_details = tech_signals.get('macd_details', {})
                                divergence_info = macd_details.get('divergence_details', {})
                                
                                if tech_signals.get('MACD_signal', 0) > 0:
                                    if divergence_info.get('bullish_divergence'):
                                        reasoning.append("✅ **MACD**: Strong bullish divergence detected - potential reversal")
                                    elif divergence_info.get('hidden_bullish'):
                                        reasoning.append("✅ **MACD**: Hidden bullish divergence - trend continuation")
                                    else:
                                        reasoning.append("✅ **MACD**: Bullish momentum confirmed")
                                elif tech_signals.get('MACD_signal', 0) < 0:
                                    if divergence_info.get('bearish_divergence'):
                                        reasoning.append("❌ **MACD**: Strong bearish divergence detected - potential reversal")
                                    elif divergence_info.get('hidden_bearish'):
                                        reasoning.append("❌ **MACD**: Hidden bearish divergence - trend continuation")
                                    else:
                                        reasoning.append("❌ **MACD**: Bearish momentum confirmed")
                                
                                if tech_signals.get('BB_signal', 0) > 0:
                                    reasoning.append("✅ **Bollinger Bands**: Price at lower band, oversold")
                                elif tech_signals.get('BB_signal', 0) < 0:
                                    reasoning.append("❌ **Bollinger Bands**: Price at upper band, overbought")
                                
                                if tech_signals.get('STOCH_signal', 0) > 0:
                                    reasoning.append("✅ **Stochastic**: Oversold territory, potential reversal")
                                elif tech_signals.get('STOCH_signal', 0) < 0:
                                    reasoning.append("❌ **Stochastic**: Overbought territory, potential reversal")
                                
                                if tech_signals.get('SR_signal', 0) > 0:
                                    reasoning.append(f"✅ **Support/Resistance**: Price near support level ${tech_signals.get('near_support', 0):.2f}, potential bounce")
                                elif tech_signals.get('SR_signal', 0) < 0:
                                    reasoning.append(f"❌ **Support/Resistance**: Price near resistance level ${tech_signals.get('near_resistance', 0):.2f}, potential rejection")
                                
                                if tech_signals.get('pivot_signal', 0) > 0:
                                    reasoning.append("✅ **Pivot Point**: Price above pivot point, indicating bullish sentiment")
                                elif tech_signals.get('pivot_signal', 0) < 0:
                                    reasoning.append("❌ **Pivot Point**: Price below pivot point, indicating bearish sentiment")
                                
                                if tech_signals.get('ADX_signal', 0) > 0:
                                    reasoning.append("✅ **ADX**: Strong uptrend confirmed")
                                elif tech_signals.get('ADX_signal', 0) < 0:
                                    reasoning.append("❌ **ADX**: Strong downtrend confirmed")
                                else:
                                    reasoning.append("⚠️ **ADX**: No clear trend, sideways movement")
                                
                                for reason in reasoning:
                                    st.markdown(reason)
                    
                    # Position Sizing
                    if "Position Sizing" in analysis_type and indicators:
                        st.markdown("## 💰 Recommended Position")
                        
                        atr_value = indicators['ATR'].iloc[-1]
                        position_data = calculate_position_sizing(current_price, atr_value, portfolio_value, risk_percent)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            action = "BUY" if tech_score >= 2 else "SELL" if tech_score <= -2 else "HOLD"
                            action_color = "green" if action == "BUY" else "red" if action == "SELL" else "orange"
                            st.markdown(f"**Action:** <span style='color: {action_color}'>{action}</span>", unsafe_allow_html=True)
                        
                        with col2:
                            st.metric("Quantity", f"{position_data['shares']} shares")
                        
                        with col3:
                            st.metric("Position Value", f"${position_data['position_value']:.2f}")
                        
                        with col4:
                            st.metric("Risk Amount", f"${position_data['risk_amount']:.2f} ({risk_percent*100:.1f}%)")
                        
                        st.markdown("---")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Stop Loss", f"${position_data['stop_loss_price']:.2f}")
                        
                        with col2:
                            st.metric("ATR (2x)", f"${position_data['stop_loss_distance']:.2f}")
                        
                        with col3:
                            risk_reward = 2.0  # Typical 1:2 risk-reward ratio
                            target_price = current_price + (position_data['stop_loss_distance'] * risk_reward)
                            st.metric("Target Price", f"${target_price:.2f}")
                    
                    # Fundamental Analysis
                    if "Fundamental Analysis" in analysis_type and analyzer.info:
                        st.markdown("## 📊 Fundamental Analysis")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.subheader("📈 Valuation Metrics")
                            pe_ratio = analyzer.info.get('trailingPE', 'N/A')
                            pb_ratio = analyzer.info.get('priceToBook', 'N/A')
                            ps_ratio = analyzer.info.get('priceToSalesTrailing12Months', 'N/A')
                            
                            st.metric("P/E Ratio", f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else pe_ratio)
                            st.metric("P/B Ratio", f"{pb_ratio:.2f}" if isinstance(pb_ratio, (int, float)) else pb_ratio)
                            st.metric("P/S Ratio", f"{ps_ratio:.2f}" if isinstance(ps_ratio, (int, float)) else ps_ratio)
                        
                        with col2:
                            st.subheader("💰 Financial Health")
                            market_cap = analyzer.info.get('marketCap', 'N/A')
                            debt_to_equity = analyzer.info.get('debtToEquity', 'N/A')
                            roe = analyzer.info.get('returnOnEquity', 'N/A')
                            
                            if isinstance(market_cap, (int, float)):
                                if market_cap >= 1e12:
                                    market_cap_str = f"${market_cap/1e12:.2f}T"
                                elif market_cap >= 1e9:
                                    market_cap_str = f"${market_cap/1e9:.2f}B"
                                else:
                                    market_cap_str = f"${market_cap/1e6:.2f}M"
                            else:
                                market_cap_str = market_cap
                            
                            st.metric("Market Cap", market_cap_str)
                            st.metric("Debt/Equity", f"{debt_to_equity:.2f}" if isinstance(debt_to_equity, (int, float)) else debt_to_equity)
                            st.metric("ROE", f"{roe:.2%}" if isinstance(roe, (int, float)) else roe)
                        
                        with col3:
                            st.subheader("📋 Company Info")
                            sector = analyzer.info.get('sector', 'N/A')
                            industry = analyzer.info.get('industry', 'N/A')
                            employees = analyzer.info.get('fullTimeEmployees', 'N/A')
                            
                            st.write(f"**Sector:** {sector}")
                            st.write(f"**Industry:** {industry}")
                            st.write(f"**Employees:** {employees:,}" if isinstance(employees, (int, float)) else f"**Employees:** {employees}")
                    
                    # News Sentiment
                    if "News Sentiment" in analysis_type:
                        st.markdown("## 📰 News Sentiment Analysis")
                        
                        news_items = get_news_sentiment(symbol)
                        
                        for item in news_items:
                            with st.expander(f"📰 {item['title']} - {item['date']}"):
                                sentiment_color = "green" if item['sentiment'] == "Positive" else "red" if item['sentiment'] == "Negative" else "yellow"
                                st.markdown(f"**Sentiment:** <span style='color: {sentiment_color}'>{item['sentiment']}</span>", unsafe_allow_html=True)
                
                else:
                    st.error("Failed to fetch stock data. Please check the symbol and try again.")
        else:
            st.warning("Please enter a stock symbol.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Disclaimer:** This analysis is for educational purposes only and should not be considered as financial advice.")

if __name__ == "__main__":
    main()
