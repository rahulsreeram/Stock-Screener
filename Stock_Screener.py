import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import warnings
from scipy.stats import norm
import math
warnings.filterwarnings('ignore')

# Install required packages if not available
try:
    from textblob import TextBlob
except ImportError:
    st.error("Please install textblob: pip install textblob")

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
        """Detect support and resistance levels using local minima/maxima"""
        def is_support(df, i, n1, n2):
            if i < n1 or i >= len(df) - n2:
                return False
            
            current_low = df['Low'].iloc[i]
            
            for j in range(i - n1, i + n2 + 1):
                if j != i and df['Low'].iloc[j] < current_low:
                    return False
            
            candle_body = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
            lower_wick = min(df['Open'].iloc[i], df['Close'].iloc[i]) - df['Low'].iloc[i]
            
            return lower_wick > candle_body * 0.1
        
        def is_resistance(df, i, n1, n2):
            if i < n1 or i >= len(df) - n2:
                return False
            
            current_high = df['High'].iloc[i]
            
            for j in range(i - n1, i + n2 + 1):
                if j != i and df['High'].iloc[j] > current_high:
                    return False
            
            candle_body = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
            upper_wick = df['High'].iloc[i] - max(df['Open'].iloc[i], df['Close'].iloc[i])
            
            return upper_wick > candle_body * 0.1
        
        support_levels = []
        resistance_levels = []
        
        start_idx = max(n1, len(data) - backcandles)
        end_idx = len(data) - n2
        
        for i in range(start_idx, end_idx):
            if is_support(data, i, n1, n2):
                support_levels.append(data['Low'].iloc[i])
            if is_resistance(data, i, n1, n2):
                resistance_levels.append(data['High'].iloc[i])
        
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
        
        high = data['High'].iloc[-1]
        low = data['Low'].iloc[-1]
        close = data['Close'].iloc[-1]
        
        pivot_point = (high + low + close) / 3
        
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
        """Detect MACD divergence patterns"""
        def find_peaks_and_troughs(series, window=5):
            peaks = []
            troughs = []
            
            for i in range(window, len(series) - window):
                if all(series.iloc[i] >= series.iloc[i-j] for j in range(1, window+1)) and \
                   all(series.iloc[i] >= series.iloc[i+j] for j in range(1, window+1)):
                    peaks.append((i, series.iloc[i]))
                
                if all(series.iloc[i] <= series.iloc[i-j] for j in range(1, window+1)) and \
                   all(series.iloc[i] <= series.iloc[i+j] for j in range(1, window+1)):
                    troughs.append((i, series.iloc[i]))
            
            return peaks, troughs
        
        recent_data = data.tail(lookback_periods)
        recent_macd = macd_line.tail(lookback_periods)
        
        price_peaks, price_troughs = find_peaks_and_troughs(recent_data['Close'])
        macd_peaks, macd_troughs = find_peaks_and_troughs(recent_macd)
        
        divergences = {
            'bullish_divergence': False,
            'bearish_divergence': False,
            'hidden_bullish': False,
            'hidden_bearish': False,
            'divergence_strength': 0
        }
        
        if len(price_troughs) >= 2 and len(macd_troughs) >= 2:
            latest_price_trough = price_troughs[-1]
            prev_price_trough = price_troughs[-2]
            latest_macd_trough = macd_troughs[-1]
            prev_macd_trough = macd_troughs[-2]
            
            if (latest_price_trough[1] < prev_price_trough[1] and 
                latest_macd_trough[1] > prev_macd_trough[1]):
                divergences['bullish_divergence'] = True
                divergences['divergence_strength'] += 2
        
        if len(price_peaks) >= 2 and len(macd_peaks) >= 2:
            latest_price_peak = price_peaks[-1]
            prev_price_peak = price_peaks[-2]
            latest_macd_peak = macd_peaks[-1]
            prev_macd_peak = macd_peaks[-2]
            
            if (latest_price_peak[1] > prev_price_peak[1] and 
                latest_macd_peak[1] < prev_macd_peak[1]):
                divergences['bearish_divergence'] = True
                divergences['divergence_strength'] -= 2
        
        if len(price_troughs) >= 2 and len(macd_troughs) >= 2:
            latest_price_trough = price_troughs[-1]
            prev_price_trough = price_troughs[-2]
            latest_macd_trough = macd_troughs[-1]
            prev_macd_trough = macd_troughs[-2]
            
            if (latest_price_trough[1] > prev_price_trough[1] and 
                latest_macd_trough[1] < prev_macd_trough[1]):
                divergences['hidden_bullish'] = True
                divergences['divergence_strength'] += 1
        
        if len(price_peaks) >= 2 and len(macd_peaks) >= 2:
            latest_price_peak = price_peaks[-1]
            prev_price_peak = price_peaks[-2]
            latest_macd_peak = macd_peaks[-1]
            prev_macd_peak = macd_peaks[-2]
            
            if (latest_price_peak[1] < prev_price_peak[1] and 
                latest_macd_peak[1] > prev_macd_peak[1]):
                divergences['hidden_bearish'] = True
                divergences['divergence_strength'] -= 1
        
        return divergences
    
    @staticmethod
    def enhanced_macd_signals(data, macd_line, signal_line, histogram):
        """Enhanced MACD signal generation including divergence analysis"""
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
        
        if current_macd > current_signal and prev_macd <= prev_signal:
            signals['crossover_signal'] = 1
        elif current_macd < current_signal and prev_macd >= prev_signal:
            signals['crossover_signal'] = -1
        
        if current_histogram > 0:
            signals['histogram_signal'] = 1
        else:
            signals['histogram_signal'] = -1
        
        if current_macd > 0:
            signals['zero_line_signal'] = 1
        else:
            signals['zero_line_signal'] = -1
        
        divergence_data = TechnicalIndicators.detect_macd_divergence(data, macd_line)
        
        if divergence_data['bullish_divergence']:
            signals['divergence_signal'] = 2
        elif divergence_data['bearish_divergence']:
            signals['divergence_signal'] = -2
        elif divergence_data['hidden_bullish']:
            signals['divergence_signal'] = 1
        elif divergence_data['hidden_bearish']:
            signals['divergence_signal'] = -1
        
        signals['overall_macd_signal'] = (
            signals['crossover_signal'] * 2 +
            signals['histogram_signal'] * 1 +
            signals['zero_line_signal'] * 1 +
            signals['divergence_signal'] * 2
        )
        
        signals['divergence_details'] = divergence_data
        
        return signals

    @staticmethod
    def ema_ribbon_signal(close):
        """Generate EMA ribbon signals based on EMA 10, 21, 50, 200 alignment"""
        ema_10 = TechnicalIndicators.ema(close, 10)
        ema_21 = TechnicalIndicators.ema(close, 21)
        ema_50 = TechnicalIndicators.ema(close, 50)
        ema_200 = TechnicalIndicators.ema(close, 200)
        
        latest = -1
        
        # Check for perfect bullish alignment
        if (ema_10.iloc[latest] > ema_21.iloc[latest] > 
            ema_50.iloc[latest] > ema_200.iloc[latest]):
            return 1  # Bullish
        
        # Check for perfect bearish alignment
        elif (ema_10.iloc[latest] < ema_21.iloc[latest] < 
              ema_50.iloc[latest] < ema_200.iloc[latest]):
            return -1  # Bearish
        
        else:
            return 0  # Neutral

class TechnicalProbabilityCalculator:
    """Calculate technical probability using proper signal weighting and direction"""
    
    @staticmethod
    def calculate_signal_strength_score(signals):
        """Calculate weighted signal strength considering direction and confidence"""
        
        # Define signal weights based on reliability and strength
        signal_weights = {
            'EMA_ribbon_signal': 3.0,  # High weight for trend confirmation
            'MACD_signal': 2.5,        # High weight for momentum
            'RSI_signal': 2.0,         # Medium-high weight for overbought/oversold
            'BB_signal': 2.0,          # Medium-high weight for volatility
            'STOCH_signal': 1.5,       # Medium weight for momentum
            'SR_signal': 2.5,          # High weight for key levels
            'pivot_signal': 1.5,       # Medium weight for daily levels
            'ADX_signal': 2.0          # Medium-high weight for trend strength
        }
        
        # Calculate weighted bullish and bearish scores
        bullish_score = 0
        bearish_score = 0
        total_weight = 0
        
        for signal_name, signal_value in signals.items():
            if signal_name in signal_weights:
                weight = signal_weights[signal_name]
                total_weight += weight
                
                if signal_value > 0:
                    bullish_score += weight * signal_value
                elif signal_value < 0:
                    bearish_score += weight * abs(signal_value)
        
        # Calculate net signal strength (-1 to +1)
        if total_weight > 0:
            net_score = (bullish_score - bearish_score) / total_weight
        else:
            net_score = 0
        
        return net_score, bullish_score, bearish_score, total_weight
    
    @staticmethod
    def calculate_technical_probability(signals, position_type="long"):
        """Calculate technical probability based on signal alignment and strength"""
        
        net_score, bullish_score, bearish_score, total_weight = TechnicalProbabilityCalculator.calculate_signal_strength_score(signals)
        
        # Base probability starts at 50% (neutral)
        base_probability = 0.50
        
        # Adjust probability based on signal strength and direction
        if position_type.lower() == "long":
            # For long positions, positive signals increase probability
            if net_score > 0:
                # Strong bullish signals
                probability_adjustment = min(0.35, net_score * 0.35)  # Max 35% boost
            else:
                # Bearish signals decrease probability
                probability_adjustment = max(-0.35, net_score * 0.35)  # Max 35% reduction
        else:
            # For short positions, negative signals increase probability
            if net_score < 0:
                # Strong bearish signals
                probability_adjustment = min(0.35, abs(net_score) * 0.35)  # Max 35% boost
            else:
                # Bullish signals decrease probability
                probability_adjustment = max(-0.35, -net_score * 0.35)  # Max 35% reduction
        
        technical_probability = base_probability + probability_adjustment
        
        # Ensure probability stays within reasonable bounds
        technical_probability = max(0.15, min(0.85, technical_probability))
        
        return technical_probability, net_score, bullish_score, bearish_score
    
    @staticmethod
    def get_signal_analysis(signals):
        """Get detailed analysis of individual signals"""
        analysis = {
            'strong_bullish': [],
            'weak_bullish': [],
            'neutral': [],
            'weak_bearish': [],
            'strong_bearish': []
        }
        
        signal_descriptions = {
            'EMA_ribbon_signal': 'EMA Ribbon Alignment',
            'MACD_signal': 'MACD Momentum',
            'RSI_signal': 'RSI Overbought/Oversold',
            'BB_signal': 'Bollinger Bands Position',
            'STOCH_signal': 'Stochastic Oscillator',
            'SR_signal': 'Support/Resistance Levels',
            'pivot_signal': 'Pivot Point Position',
            'ADX_signal': 'Trend Strength (ADX)'
        }
        
        for signal_name, signal_value in signals.items():
            if signal_name in signal_descriptions:
                desc = signal_descriptions[signal_name]
                
                if signal_value >= 2:
                    analysis['strong_bullish'].append(desc)
                elif signal_value == 1:
                    analysis['weak_bullish'].append(desc)
                elif signal_value == 0:
                    analysis['neutral'].append(desc)
                elif signal_value == -1:
                    analysis['weak_bearish'].append(desc)
                elif signal_value <= -2:
                    analysis['strong_bearish'].append(desc)
        
        return analysis

class StockProbabilityCalculator:
    """Calculate probability of profit for stock positions using proper statistical methods"""
    
    @staticmethod
    def calculate_historical_statistics(price_data, window=252):
        """Calculate historical returns, volatility, and drift"""
        try:
            returns = price_data.pct_change().dropna()
            
            # Calculate annualized statistics
            daily_returns = returns.tail(window) if len(returns) > window else returns
            mean_return = daily_returns.mean() * 252  # Annualized drift
            volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
            
            return {
                'mean_return': mean_return,
                'volatility': volatility,
                'daily_volatility': daily_returns.std(),
                'daily_mean': daily_returns.mean()
            }
        except Exception as e:
            return {
                'mean_return': 0.08,  # Default 8% annual return
                'volatility': 0.20,   # Default 20% volatility
                'daily_volatility': 0.20 / np.sqrt(252),
                'daily_mean': 0.08 / 252
            }
    
    @staticmethod
    def monte_carlo_simulation(current_price, target_price, stop_loss, days_to_target, 
                              mean_return, volatility, num_simulations=10000):
        """Monte Carlo simulation for stock price probability"""
        try:
            # Convert to daily parameters
            dt = 1/252  # Daily time step
            daily_drift = mean_return * dt
            daily_vol = volatility * np.sqrt(dt)
            
            successful_trades = 0
            
            for _ in range(num_simulations):
                price = current_price
                hit_target = False
                hit_stop = False
                
                for day in range(days_to_target):
                    # Generate random price movement
                    random_shock = np.random.normal(0, 1)
                    price_change = daily_drift + daily_vol * random_shock
                    price = price * (1 + price_change)
                    
                    # Check if target or stop loss is hit
                    if target_price > current_price:  # Long position
                        if price >= target_price:
                            hit_target = True
                            break
                        elif price <= stop_loss:
                            hit_stop = True
                            break
                    else:  # Short position
                        if price <= target_price:
                            hit_target = True
                            break
                        elif price >= stop_loss:
                            hit_stop = True
                            break
                
                if hit_target:
                    successful_trades += 1
            
            return successful_trades / num_simulations
            
        except Exception as e:
            return 0.5
    
    @staticmethod
    def geometric_brownian_motion_probability(current_price, target_price, stop_loss, 
                                            days_to_target, mean_return, volatility):
        """Calculate probability using Geometric Brownian Motion"""
        try:
            # Convert to time in years
            T = days_to_target / 252
            
            if T <= 0:
                return 0.5
            
            # For long positions (target > current)
            if target_price > current_price:
                # Calculate probability of reaching target before stop loss
                # Using barrier option pricing theory
                
                # Drift-adjusted parameters
                mu = mean_return - 0.5 * volatility**2
                
                # Distance to target and stop loss (in log terms)
                log_target = np.log(target_price / current_price)
                log_stop = np.log(stop_loss / current_price)
                
                # Probability calculations using normal distribution
                d1_target = (log_target - mu * T) / (volatility * np.sqrt(T))
                d1_stop = (log_stop - mu * T) / (volatility * np.sqrt(T))
                
                # Probability of reaching target
                prob_target = 1 - norm.cdf(d1_target)
                
                # Probability of hitting stop loss
                prob_stop = norm.cdf(d1_stop)
                
                # Combined probability (simplified approach)
                prob_profit = prob_target / (prob_target + prob_stop) if (prob_target + prob_stop) > 0 else 0.5
                
            else:  # Short positions (target < current)
                # For short positions, logic is reversed
                mu = mean_return - 0.5 * volatility**2
                
                log_target = np.log(target_price / current_price)
                log_stop = np.log(stop_loss / current_price)
                
                d1_target = (log_target - mu * T) / (volatility * np.sqrt(T))
                d1_stop = (log_stop - mu * T) / (volatility * np.sqrt(T))
                
                prob_target = norm.cdf(d1_target)
                prob_stop = 1 - norm.cdf(d1_stop)
                
                prob_profit = prob_target / (prob_target + prob_stop) if (prob_target + prob_stop) > 0 else 0.5
            
            return max(0.05, min(0.95, prob_profit))
            
        except Exception as e:
            return 0.5

def get_real_news_sentiment(symbol):
    """Get real news sentiment with updated data structure handling"""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        
        if not news:
            st.warning(f"No news data available for {symbol}")
            return []
        
        news_items = []
        for article in news[:15]:
            try:
                # Handle new data structure - extract from 'content' field
                content = article.get('content', {})
                
                # Try multiple possible field names for title
                title = (content.get('title') or 
                        content.get('headline') or 
                        content.get('summary') or
                        article.get('title') or
                        article.get('headline') or
                        'No title available')
                
                # Skip if no meaningful title
                if not title or title == 'No title available':
                    continue
                
                # Extract other fields from content or main article
                publisher = (content.get('provider', {}).get('displayName') or
                           content.get('publisher') or
                           content.get('source') or
                           article.get('publisher') or
                           'Unknown')
                
                # Extract timestamp
                timestamp = (content.get('pubDate') or
                           content.get('publishTime') or
                           content.get('providerPublishTime') or
                           article.get('providerPublishTime') or
                           article.get('timestamp') or 0)
                
                # Extract URL
                url = (content.get('clickThroughUrl') or
                      content.get('url') or
                      content.get('link') or
                      article.get('link') or
                      article.get('url') or '')
                
                # Sentiment analysis
                try:
                    from textblob import TextBlob
                    blob = TextBlob(title)
                    polarity = blob.sentiment.polarity
                    subjectivity = blob.sentiment.subjectivity
                    
                    if polarity > 0.3:
                        sentiment = "Very Positive"
                    elif polarity > 0.1:
                        sentiment = "Positive"
                    elif polarity < -0.3:
                        sentiment = "Very Negative"
                    elif polarity < -0.1:
                        sentiment = "Negative"
                    else:
                        sentiment = "Neutral"
                        
                except ImportError:
                    st.error("TextBlob not installed. Run: pip install textblob")
                    sentiment = "Unknown"
                    polarity = 0
                    subjectivity = 0
                except Exception as e:
                    sentiment = "Neutral"
                    polarity = 0
                    subjectivity = 0.5
                
                # Convert timestamp to readable date
                try:
                    if timestamp and isinstance(timestamp, (int, float)) and timestamp > 0:
                        date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
                    else:
                        date = datetime.now().strftime('%Y-%m-%d %H:%M')
                except:
                    date = datetime.now().strftime('%Y-%m-%d %H:%M')
                
                news_items.append({
                    "title": title,
                    "sentiment": sentiment,
                    "polarity_score": round(polarity, 3),
                    "subjectivity_score": round(subjectivity, 3),
                    "date": date,
                    "publisher": publisher,
                    "url": url
                })
                
            except Exception as e:
                # Skip problematic articles
                continue
        
        # If no news items found, create a fallback based on recent performance
        if not news_items:
            try:
                # Get recent price data for fallback news
                hist_data = ticker.history(period="5d")
                if len(hist_data) >= 2:
                    current_price = hist_data['Close'].iloc[-1]
                    prev_price = hist_data['Close'].iloc[-2]
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                    
                    if change_pct > 3:
                        title = f"{symbol} shows strong performance with {change_pct:.1f}% gain"
                        sentiment = "Positive"
                        polarity = 0.4
                    elif change_pct < -3:
                        title = f"{symbol} experiences {abs(change_pct):.1f}% decline in recent trading"
                        sentiment = "Negative"
                        polarity = -0.4
                    else:
                        title = f"{symbol} trading with mixed signals, {change_pct:+.1f}% change"
                        sentiment = "Neutral"
                        polarity = 0.1 if change_pct > 0 else -0.1
                    
                    news_items.append({
                        "title": title,
                        "sentiment": sentiment,
                        "polarity_score": polarity,
                        "subjectivity_score": 0.6,
                        "date": datetime.now().strftime('%Y-%m-%d %H:%M'),
                        "publisher": "Market Analysis",
                        "url": ""
                    })
            except:
                pass
        
        return news_items
        
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

class StockAnalyzer:
    def __init__(self, symbol, period="1y"):
        self.symbol = symbol.upper()
        self.period = period
        self.data = None
        self.info = None
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance with fallback for 1mo period"""
        try:
            ticker = yf.Ticker(self.symbol)
            
            # Try original period first
            self.data = ticker.history(period=self.period)
            
            # Fallback for 1mo period issue
            if self.data.empty and self.period == "1mo":
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                self.data = ticker.history(start=start_date, end=end_date)
            
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
        
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        volume = self.data['Volume']
        
        # EMA Ribbon indicators
        indicators['EMA_10'] = TechnicalIndicators.ema(close, 10)
        indicators['EMA_21'] = TechnicalIndicators.ema(close, 21)
        indicators['EMA_50'] = TechnicalIndicators.ema(close, 50)
        indicators['EMA_200'] = TechnicalIndicators.ema(close, 200)
        
        # Keep other indicators
        indicators['SMA_20'] = TechnicalIndicators.sma(close, 20)
        indicators['EMA_12'] = TechnicalIndicators.ema(close, 12)
        indicators['EMA_26'] = TechnicalIndicators.ema(close, 26)
        
        indicators['RSI'] = TechnicalIndicators.rsi(close)
        
        macd, signal, histogram = TechnicalIndicators.macd(close)
        indicators['MACD'] = macd
        indicators['MACD_signal'] = signal
        indicators['MACD_histogram'] = histogram
        
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(close)
        indicators['BB_upper'] = bb_upper
        indicators['BB_middle'] = bb_middle
        indicators['BB_lower'] = bb_lower
        
        stoch_k, stoch_d = TechnicalIndicators.stochastic(high, low, close)
        indicators['STOCH_K'] = stoch_k
        indicators['STOCH_D'] = stoch_d
        
        indicators['ATR'] = TechnicalIndicators.atr(high, low, close)
        
        adx, plus_di, minus_di = TechnicalIndicators.adx(high, low, close)
        indicators['ADX'] = adx
        indicators['PLUS_DI'] = plus_di
        indicators['MINUS_DI'] = minus_di
        
        support_levels, resistance_levels = TechnicalIndicators.detect_support_resistance(self.data)
        indicators['support_levels'] = support_levels
        indicators['resistance_levels'] = resistance_levels
        
        pivot_data = TechnicalIndicators.calculate_pivot_points(self.data)
        if pivot_data:
            indicators.update(pivot_data)
        
        return indicators
    
    def generate_technical_signals(self, indicators):
        """Generate buy/sell signals based on technical indicators"""
        signals = {}
        current_price = self.data['Close'].iloc[-1]
        
        # EMA Ribbon signals
        ema_ribbon_signal = TechnicalIndicators.ema_ribbon_signal(self.data['Close'])
        signals['EMA_ribbon_signal'] = ema_ribbon_signal
        
        # RSI signals
        rsi = indicators['RSI'].iloc[-1]
        if rsi < 30:
            signals['RSI_signal'] = 1
        elif rsi > 70:
            signals['RSI_signal'] = -1
        else:
            signals['RSI_signal'] = 0
        
        # Enhanced MACD signals
        macd_signals = TechnicalIndicators.enhanced_macd_signals(
            self.data, 
            indicators['MACD'], 
            indicators['MACD_signal'], 
            indicators['MACD_histogram']
        )
        
        if macd_signals['overall_macd_signal'] >= 2:
            signals['MACD_signal'] = 1
        elif macd_signals['overall_macd_signal'] <= -2:
            signals['MACD_signal'] = -1
        else:
            signals['MACD_signal'] = 0
        
        signals['macd_details'] = macd_signals
        
        # Bollinger Bands signals
        bb_upper = indicators['BB_upper'].iloc[-1]
        bb_lower = indicators['BB_lower'].iloc[-1]
        
        if current_price <= bb_lower:
            signals['BB_signal'] = 1
        elif current_price >= bb_upper:
            signals['BB_signal'] = -1
        else:
            signals['BB_signal'] = 0
        
        # Stochastic signals
        stoch_k = indicators['STOCH_K'].iloc[-1]
        stoch_d = indicators['STOCH_D'].iloc[-1]
        
        if stoch_k < 20 and stoch_d < 20:
            signals['STOCH_signal'] = 1
        elif stoch_k > 80 and stoch_d > 80:
            signals['STOCH_signal'] = -1
        else:
            signals['STOCH_signal'] = 0
        
        # Support and Resistance Signals
        support_levels = indicators.get('support_levels', [])
        resistance_levels = indicators.get('resistance_levels', [])
        
        near_support, support_distance = TechnicalIndicators.check_level_proximity(
            current_price, support_levels, threshold=0.02
        )
        near_resistance, resistance_distance = TechnicalIndicators.check_level_proximity(
            current_price, resistance_levels, threshold=0.02
        )
        
        signals['SR_signal'] = 0
        if near_support:
            signals['SR_signal'] = 1
            signals['near_support'] = near_support
            signals['support_distance'] = support_distance
        elif near_resistance:
            signals['SR_signal'] = -1
            signals['near_resistance'] = near_resistance
            signals['resistance_distance'] = resistance_distance
        
        # Pivot point signals
        pivot_point = indicators.get('pivot_point', current_price)
        if current_price > pivot_point:
            signals['pivot_signal'] = 1
        else:
            signals['pivot_signal'] = -1
        
        # ADX trend strength signal
        adx_value = indicators['ADX'].iloc[-1]
        plus_di = indicators['PLUS_DI'].iloc[-1]
        minus_di = indicators['MINUS_DI'].iloc[-1]
        
        if adx_value > 25:
            if plus_di > minus_di:
                signals['ADX_signal'] = 1
            else:
                signals['ADX_signal'] = -1
        else:
            signals['ADX_signal'] = 0
        
        return signals

def create_price_chart(data, indicators):
    """Create comprehensive price chart with technical indicators including EMA ribbon"""
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=('Price, EMA Ribbon & Support/Resistance', 'MACD', 'RSI', 'ADX', 'Volume'),
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
    
    # EMA Ribbon
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['EMA_10'],
            mode='lines',
            name='EMA 10',
            line=dict(color='lime', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['EMA_21'],
            mode='lines',
            name='EMA 21',
            line=dict(color='orange', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['EMA_50'],
            mode='lines',
            name='EMA 50',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['EMA_200'],
            mode='lines',
            name='EMA 200',
            line=dict(color='red', width=3)
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
    
    # Support and Resistance Levels
    support_levels = indicators.get('support_levels', [])
    resistance_levels = indicators.get('resistance_levels', [])
    
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
    
    # Pivot points
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
    
    fig.update_layout(
        title="Advanced Technical Analysis Dashboard with EMA Ribbon",
        xaxis_rangeslider_visible=False,
        height=1000,
        showlegend=True,
        template="plotly_dark"
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="ADX", row=4, col=1)
    fig.update_yaxes(title_text="Volume", row=5, col=1)
    
    return fig

def main():
    st.title("📈 Advanced Stock Analysis Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📊 Analysis Settings")
    
    symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL", help="Enter a valid stock ticker symbol")
    
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
    
    # Editable Position Settings
    st.sidebar.subheader("✏️ Editable Position Settings")
    position_size = st.sidebar.number_input("Position Size (shares)", value=100, min_value=1)
    target_price = st.sidebar.number_input("Target Price ($)", value=0.0, min_value=0.0, step=0.01)
    stop_loss = st.sidebar.number_input("Stop Loss ($)", value=0.0, min_value=0.0, step=0.01)
    days_to_target = st.sidebar.slider("Days to Target", min_value=5, max_value=90, value=30, step=5)
    
    # Probability calculation method
    prob_method = st.sidebar.selectbox(
        "Probability Calculation Method",
        ["Monte Carlo Simulation", "Geometric Brownian Motion", "Combined"],
        index=2
    )
    
    analysis_type = st.sidebar.multiselect(
        "Select Analysis Type",
        ["Technical Analysis", "Probability Analysis", "Fundamental Analysis", "News Sentiment"],
        default=["Technical Analysis", "Probability Analysis"]
    )
    
    if st.sidebar.button("🔍 Analyze Stock", type="primary"):
        if symbol:
            with st.spinner(f"Analyzing {symbol}..."):
                analyzer = StockAnalyzer(symbol, period)
                
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
                    
                    # Set default target and stop loss if not provided
                    if target_price <= 0:
                        target_price = round(current_price * 1.05, 2)
                    if stop_loss <= 0:
                        stop_loss = round(current_price * 0.95, 2)
                    
                    # Technical Analysis
                    if "Technical Analysis" in analysis_type:
                        st.markdown("## 🔧 Technical Analysis")
                        
                        indicators = analyzer.calculate_technical_indicators()
                        
                        if indicators:
                            tech_signals = analyzer.generate_technical_signals(indicators)
                            
                            fig = create_price_chart(analyzer.data, indicators)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("📊 Technical Indicators")
                                
                                macd_details = tech_signals.get('macd_details', {})
                                divergence_info = macd_details.get('divergence_details', {})
                                
                                macd_description = f"MACD: {indicators['MACD'].iloc[-1]:.3f}"
                                if divergence_info.get('bullish_divergence'):
                                    macd_description += " (Bullish Divergence)"
                                elif divergence_info.get('bearish_divergence'):
                                    macd_description += " (Bearish Divergence)"
                                elif divergence_info.get('hidden_bullish'):
                                    macd_description += " (Hidden Bullish)"
                                elif divergence_info.get('hidden_bearish'):
                                    macd_description += " (Hidden Bearish)"
                                
                                # EMA Ribbon signal description
                                ema_ribbon_signal = tech_signals.get('EMA_ribbon_signal', 0)
                                if ema_ribbon_signal > 0:
                                    ema_description = "EMA10 > EMA21 > EMA50 > EMA200 (Bullish Alignment)"
                                elif ema_ribbon_signal < 0:
                                    ema_description = "EMA10 < EMA21 < EMA50 < EMA200 (Bearish Alignment)"
                                else:
                                    ema_description = "Mixed EMA alignment (Neutral)"
                                
                                tech_df = pd.DataFrame({
                                    'Indicator': ['EMA Ribbon', 'MACD', 'RSI', 'Bollinger Bands', 'Stochastic', 'Support/Resistance', 'Pivot Point', 'ADX'],
                                    'Signal': [
                                        '🟢 Buy' if tech_signals.get('EMA_ribbon_signal', 0) > 0 else '🔴 Sell' if tech_signals.get('EMA_ribbon_signal', 0) < 0 else '🟡 Neutral',
                                        '🟢 Buy' if tech_signals.get('MACD_signal', 0) > 0 else '🔴 Sell' if tech_signals.get('MACD_signal', 0) < 0 else '🟡 Neutral',
                                        '🟢 Buy' if tech_signals.get('RSI_signal', 0) > 0 else '🔴 Sell' if tech_signals.get('RSI_signal', 0) < 0 else '🟡 Neutral',
                                        '🟢 Buy' if tech_signals.get('BB_signal', 0) > 0 else '🔴 Sell' if tech_signals.get('BB_signal', 0) < 0 else '🟡 Neutral',
                                        '🟢 Buy' if tech_signals.get('STOCH_signal', 0) > 0 else '🔴 Sell' if tech_signals.get('STOCH_signal', 0) < 0 else '🟡 Neutral',
                                        '🟢 Buy' if tech_signals.get('SR_signal', 0) > 0 else '🔴 Sell' if tech_signals.get('SR_signal', 0) < 0 else '🟡 Neutral',
                                        '🟢 Bullish' if tech_signals.get('pivot_signal', 0) > 0 else '🔴 Bearish' if tech_signals.get('pivot_signal', 0) < 0 else '🟡 Neutral',
                                        '🟢 Strong Trend' if tech_signals.get('ADX_signal', 0) > 0 else '🔴 Strong Trend' if tech_signals.get('ADX_signal', 0) < 0 else '🟡 No Trend'
                                    ],
                                    'Value': [
                                        ema_description,
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
                                
                                # Use corrected technical signal calculation
                                tech_calc = TechnicalProbabilityCalculator()
                                net_score, bullish_score, bearish_score = tech_calc.calculate_signal_strength_score(tech_signals)[:3]
                                
                                if net_score >= 0.6:
                                    signal_color = "green"
                                    signal_text = "🟢 STRONG BUY"
                                elif net_score >= 0.2:
                                    signal_color = "lightgreen"
                                    signal_text = "🟢 BUY"
                                elif net_score <= -0.6:
                                    signal_color = "red"
                                    signal_text = "🔴 STRONG SELL"
                                elif net_score <= -0.2:
                                    signal_color = "lightcoral"
                                    signal_text = "🔴 SELL"
                                else:
                                    signal_color = "yellow"
                                    signal_text = "🟡 NEUTRAL"
                                
                                st.markdown(f"<h2 style='color: {signal_color}'>{signal_text}</h2>", unsafe_allow_html=True)
                                st.markdown(f"**Signal Strength:** {net_score:.2f} (-1 to +1)")
                                
                                # Signal analysis breakdown
                                signal_analysis = tech_calc.get_signal_analysis(tech_signals)
                                
                                st.subheader("📝 Signal Analysis")
                                
                                if signal_analysis['strong_bullish']:
                                    st.markdown("**🟢 Strong Bullish Signals:**")
                                    for signal in signal_analysis['strong_bullish']:
                                        st.markdown(f"- {signal}")
                                
                                if signal_analysis['weak_bullish']:
                                    st.markdown("**🟢 Weak Bullish Signals:**")
                                    for signal in signal_analysis['weak_bullish']:
                                        st.markdown(f"- {signal}")
                                
                                if signal_analysis['strong_bearish']:
                                    st.markdown("**🔴 Strong Bearish Signals:**")
                                    for signal in signal_analysis['strong_bearish']:
                                        st.markdown(f"- {signal}")
                                
                                if signal_analysis['weak_bearish']:
                                    st.markdown("**🔴 Weak Bearish Signals:**")
                                    for signal in signal_analysis['weak_bearish']:
                                        st.markdown(f"- {signal}")
                                
                                if signal_analysis['neutral']:
                                    st.markdown("**🟡 Neutral Signals:**")
                                    for signal in signal_analysis['neutral']:
                                        st.markdown(f"- {signal}")
                    
                    # UPDATED: Probability Analysis with corrected technical probability
                    if "Probability Analysis" in analysis_type and indicators:
                        st.markdown("## 🎯 Probability Analysis")
                        
                        # Display user position details
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Position Size", f"{position_size} shares")
                        
                        with col2:
                            st.metric("Target Price", f"${target_price:.2f}")
                        
                        with col3:
                            st.metric("Stop Loss", f"${stop_loss:.2f}")
                        
                        with col4:
                            st.metric("Days to Target", f"{days_to_target} days")
                        
                        # Calculate probabilities using proper methods
                        prob_calc = StockProbabilityCalculator()
                        tech_calc = TechnicalProbabilityCalculator()
                        
                        # Get historical statistics
                        hist_stats = prob_calc.calculate_historical_statistics(analyzer.data['Close'])
                        
                        # Determine position type
                        if target_price > current_price:
                            position_type = "long"
                        else:
                            position_type = "short"
                        
                        # Corrected technical probability calculation
                        technical_prob, net_score, bullish_score, bearish_score = tech_calc.calculate_technical_probability(
                            tech_signals, position_type
                        )
                        
                        # Stock probability calculations
                        if prob_method == "Monte Carlo Simulation":
                            stock_prob = prob_calc.monte_carlo_simulation(
                                current_price, target_price, stop_loss, days_to_target,
                                hist_stats['mean_return'], hist_stats['volatility']
                            )
                        elif prob_method == "Geometric Brownian Motion":
                            stock_prob = prob_calc.geometric_brownian_motion_probability(
                                current_price, target_price, stop_loss, days_to_target,
                                hist_stats['mean_return'], hist_stats['volatility']
                            )
                        else:  # Combined
                            monte_carlo_prob = prob_calc.monte_carlo_simulation(
                                current_price, target_price, stop_loss, days_to_target,
                                hist_stats['mean_return'], hist_stats['volatility']
                            )
                            gbm_prob = prob_calc.geometric_brownian_motion_probability(
                                current_price, target_price, stop_loss, days_to_target,
                                hist_stats['mean_return'], hist_stats['volatility']
                            )
                            stock_prob = (monte_carlo_prob + gbm_prob) / 2
                        
                        # Combined probability (weighted average)
                        combined_prob = (technical_prob * 0.4 + stock_prob * 0.6)
                        
                        st.markdown("---")
                        st.subheader("🎯 Probability of Profit Analysis")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            prob_color = "green" if combined_prob >= 0.7 else "orange" if combined_prob >= 0.5 else "red"
                            st.markdown(f"**Combined PoP:** <span style='color: {prob_color}; font-size: 24px; font-weight: bold'>{combined_prob:.1%}</span>", unsafe_allow_html=True)
                        
                        with col2:
                            st.metric("Technical PoP", f"{technical_prob:.1%}")
                        
                        with col3:
                            st.metric("Statistical PoP", f"{stock_prob:.1%}")
                        
                        with col4:
                            st.metric("Method Used", prob_method)
                        
                        # Technical Analysis Breakdown
                        st.markdown("### 📊 Technical Analysis Breakdown")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Net Signal Score", f"{net_score:.2f}")
                        
                        with col2:
                            st.metric("Bullish Strength", f"{bullish_score:.1f}")
                        
                        with col3:
                            st.metric("Bearish Strength", f"{bearish_score:.1f}")
                        
                        with col4:
                            st.metric("Position Type", position_type.title())
                        
                        # Historical Statistics Display
                        st.markdown("### 📈 Historical Statistics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Annual Return", f"{hist_stats['mean_return']:.1%}")
                        
                        with col2:
                            st.metric("Annual Volatility", f"{hist_stats['volatility']:.1%}")
                        
                        with col3:
                            st.metric("Daily Volatility", f"{hist_stats['daily_volatility']:.2%}")
                        
                        with col4:
                            st.metric("Daily Return", f"{hist_stats['daily_mean']:.3%}")
                        
                        # Probability breakdown
                        st.markdown("### 📊 Probability Breakdown")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Technical Analysis:**")
                            st.write(f"- Signal alignment with position: {net_score:.2f}")
                            st.write(f"- Weighted by signal reliability")
                            st.write(f"- Accounts for conflicting signals")
                            st.write(f"- Technical PoP: {technical_prob:.1%}")
                        
                        with col2:
                            st.markdown("**Statistical Analysis:**")
                            st.write(f"- Method: {prob_method}")
                            st.write(f"- Historical return: {hist_stats['mean_return']:.1%}")
                            st.write(f"- Volatility: {hist_stats['volatility']:.1%}")
                            st.write(f"- Time horizon: {days_to_target} days")
                            st.write(f"- Statistical PoP: {stock_prob:.1%}")
                        
                        # Risk-Reward Analysis
                        st.markdown("### ⚖️ Risk-Reward Analysis")
                        
                        potential_profit = (target_price - current_price) * position_size
                        potential_loss = (current_price - stop_loss) * position_size
                        actual_rr_ratio = (target_price - current_price) / (current_price - stop_loss) if (current_price - stop_loss) > 0 else 0
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            profit_color = "green" if potential_profit > 0 else "red"
                            st.markdown(f"**Potential Profit:** <span style='color: {profit_color}'>${potential_profit:.2f}</span>", unsafe_allow_html=True)
                        
                        with col2:
                            loss_color = "red" if potential_loss > 0 else "green"
                            st.markdown(f"**Potential Loss:** <span style='color: {loss_color}'>${potential_loss:.2f}</span>", unsafe_allow_html=True)
                        
                        with col3:
                            rr_color = "green" if actual_rr_ratio >= 2 else "orange" if actual_rr_ratio >= 1 else "red"
                            st.markdown(f"**Risk:Reward Ratio:** <span style='color: {rr_color}'>1:{actual_rr_ratio:.2f}</span>", unsafe_allow_html=True)
                        
                        # Expected Value
                        expected_value = (combined_prob * potential_profit) - ((1 - combined_prob) * potential_loss)
                        ev_color = "green" if expected_value > 0 else "red"
                        st.markdown(f"**Expected Value:** <span style='color: {ev_color}; font-size: 20px'>${expected_value:.2f}</span>", unsafe_allow_html=True)
                        
                        # Position Validation
                        st.markdown("### ✅ Position Validation")
                        
                        if target_price > current_price and stop_loss < current_price:
                            position_type_display = "Long Position (Bullish)"
                            position_color = "green"
                            position_valid = True
                        elif target_price < current_price and stop_loss > current_price:
                            position_type_display = "Short Position (Bearish)"
                            position_color = "red"
                            position_valid = True
                        else:
                            position_type_display = "Invalid Position Setup"
                            position_color = "red"
                            position_valid = False
                        
                        st.markdown(f"**Position Type:** <span style='color: {position_color}'>{position_type_display}</span>", unsafe_allow_html=True)
                        
                        if not position_valid:
                            st.error("⚠️ **Invalid Position Setup**: For long positions, target should be above current price and stop loss below. For short positions, target should be below current price and stop loss above.")
                        
                        # Recommendation
                        st.markdown("### 📝 Trading Recommendation")
                        
                        if not position_valid:
                            recommendation = "🔴 **INVALID SETUP** - Please correct your target and stop loss levels"
                            rec_color = "red"
                        elif combined_prob >= 0.75 and actual_rr_ratio >= 2:
                            rec_color = "green"
                            recommendation = "🟢 **EXCELLENT TRADE** - High probability with good risk/reward"
                        elif combined_prob >= 0.65 and actual_rr_ratio >= 1.5:
                            rec_color = "lightgreen"
                            recommendation = "🟢 **GOOD TRADE** - Favorable probability and risk/reward"
                        elif combined_prob >= 0.55:
                            rec_color = "orange"
                            recommendation = "🟡 **MODERATE TRADE** - Consider position sizing"
                        else:
                            rec_color = "red"
                            recommendation = "🔴 **HIGH RISK TRADE** - Low probability of success"
                        
                        st.markdown(f"<div style='padding: 10px; border-left: 4px solid {rec_color}; background-color: rgba(128,128,128,0.1)'>{recommendation}</div>", unsafe_allow_html=True)
                    
                    # Fundamental Analysis
                    if "Fundamental Analysis" in analysis_type and analyzer.info:
                        st.markdown("## 📊 Fundamental Analysis")
                        
                        info = analyzer.info
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.subheader("💰 Valuation Metrics")
                            pe_ratio = info.get('trailingPE', 'N/A')
                            forward_pe = info.get('forwardPE', 'N/A')
                            peg_ratio = info.get('pegRatio', 'N/A')
                            price_to_book = info.get('priceToBook', 'N/A')
                            
                            st.write(f"**P/E Ratio:** {pe_ratio}")
                            st.write(f"**Forward P/E:** {forward_pe}")
                            st.write(f"**PEG Ratio:** {peg_ratio}")
                            st.write(f"**Price-to-Book:** {price_to_book}")
                        
                        with col2:
                            st.subheader("💸 Financial Health")
                            market_cap = info.get('marketCap', 'N/A')
                            debt_to_equity = info.get('debtToEquity', 'N/A')
                            current_ratio = info.get('currentRatio', 'N/A')
                            roe = info.get('returnOnEquity', 'N/A')
                            
                            if isinstance(market_cap, (int, float)):
                                market_cap = f"${market_cap/1e9:.2f}B"
                            
                            st.write(f"**Market Cap:** {market_cap}")
                            st.write(f"**Debt-to-Equity:** {debt_to_equity}")
                            st.write(f"**Current Ratio:** {current_ratio}")
                            st.write(f"**ROE:** {roe}")
                        
                        with col3:
                            st.subheader("📈 Growth & Profitability")
                            revenue_growth = info.get('revenueGrowth', 'N/A')
                            earnings_growth = info.get('earningsGrowth', 'N/A')
                            profit_margin = info.get('profitMargins', 'N/A')
                            operating_margin = info.get('operatingMargins', 'N/A')
                            
                            if isinstance(revenue_growth, (int, float)):
                                revenue_growth = f"{revenue_growth*100:.2f}%"
                            if isinstance(earnings_growth, (int, float)):
                                earnings_growth = f"{earnings_growth*100:.2f}%"
                            if isinstance(profit_margin, (int, float)):
                                profit_margin = f"{profit_margin*100:.2f}%"
                            if isinstance(operating_margin, (int, float)):
                                operating_margin = f"{operating_margin*100:.2f}%"
                            
                            st.write(f"**Revenue Growth:** {revenue_growth}")
                            st.write(f"**Earnings Growth:** {earnings_growth}")
                            st.write(f"**Profit Margin:** {profit_margin}")
                            st.write(f"**Operating Margin:** {operating_margin}")
                    
                    # News Sentiment Analysis
                    if "News Sentiment" in analysis_type:
                        st.markdown("## 📰 News Sentiment Analysis")
                        
                        news_data = get_real_news_sentiment(symbol)
                        
                        if news_data:
                            # Overall sentiment summary
                            sentiments = [item['polarity_score'] for item in news_data]
                            avg_sentiment = np.mean(sentiments)
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if avg_sentiment > 0.1:
                                    sentiment_color = "green"
                                    sentiment_text = "🟢 Positive"
                                elif avg_sentiment < -0.1:
                                    sentiment_color = "red"
                                    sentiment_text = "🔴 Negative"
                                else:
                                    sentiment_color = "orange"
                                    sentiment_text = "🟡 Neutral"
                                
                                st.markdown(f"**Overall Sentiment:** <span style='color: {sentiment_color}'>{sentiment_text}</span>", unsafe_allow_html=True)
                            
                            with col2:
                                st.metric("Average Sentiment Score", f"{avg_sentiment:.3f}")
                            
                            with col3:
                                st.metric("News Articles Analyzed", len(news_data))
                            
                            # News articles
                            st.subheader("📄 Recent News Articles")
                            
                            for i, article in enumerate(news_data[:10]):
                                with st.expander(f"{article['title'][:100]}..." if len(article['title']) > 100 else article['title']):
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        sentiment_color = "green" if article['sentiment'] in ["Positive", "Very Positive"] else "red" if article['sentiment'] in ["Negative", "Very Negative"] else "orange"
                                        st.markdown(f"**Sentiment:** <span style='color: {sentiment_color}'>{article['sentiment']}</span>", unsafe_allow_html=True)
                                    
                                    with col2:
                                        st.write(f"**Score:** {article['polarity_score']}")
                                    
                                    with col3:
                                        st.write(f"**Date:** {article['date']}")
                                    
                                    st.write(f"**Publisher:** {article['publisher']}")
                                    
                                    if article['url']:
                                        st.markdown(f"[Read full article]({article['url']})")
                        else:
                            st.warning("No news data available for sentiment analysis.")
        else:
            st.warning("Please enter a stock symbol.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Disclaimer:** This analysis is for educational purposes only and should not be considered as financial advice. Always conduct your own research and consult with financial professionals before making investment decisions.")

if __name__ == '__main__':
    main()

