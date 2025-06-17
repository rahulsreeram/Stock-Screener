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

class ProbabilityCalculator:
    """Calculate probability of profit for trading positions"""
    
    @staticmethod
    def calculate_multi_indicator_probability(signals, individual_probabilities=None):
        """Calculate probability of profit using multiple indicator approach"""
        if individual_probabilities is None:
            individual_probabilities = {
                'MA_signal': 0.35,
                'MACD_signal': 0.35,
                'RSI_signal': 0.35,
                'BB_signal': 0.35,
                'STOCH_signal': 0.35,
                'SR_signal': 0.40,
                'pivot_signal': 0.35,
                'ADX_signal': 0.30
            }
        
        active_signals = []
        signal_probs = []
        
        for signal_name, signal_value in signals.items():
            if signal_name in individual_probabilities and signal_value != 0:
                active_signals.append(signal_name)
                signal_probs.append(individual_probabilities[signal_name])
        
        if not signal_probs:
            return 0.5
        
        prob_all_wrong = 1.0
        for prob in signal_probs:
            prob_all_wrong *= (1 - prob)
        
        prob_at_least_one_correct = 1 - prob_all_wrong
        
        return prob_at_least_one_correct
    
    @staticmethod
    def calculate_stock_probability_of_profit(current_price, target_price, stop_loss, 
                                            volatility, days_to_target=30):
        """Calculate probability of profit for stock position using Black-Scholes approach"""
        try:
            time_to_target = days_to_target / 365.0
            risk_free_rate = 0.05
            
            if volatility <= 0 or time_to_target <= 0:
                return 0.5
            
            d1_target = (np.log(current_price / target_price) + 
                        (risk_free_rate + 0.5 * volatility**2) * time_to_target) / \
                       (volatility * np.sqrt(time_to_target))
            
            d2_target = d1_target - volatility * np.sqrt(time_to_target)
            
            prob_target = norm.cdf(d2_target)
            
            d1_stop = (np.log(current_price / stop_loss) + 
                      (risk_free_rate + 0.5 * volatility**2) * time_to_target) / \
                     (volatility * np.sqrt(time_to_target))
            
            d2_stop = d1_stop - volatility * np.sqrt(time_to_target)
            
            prob_stop = 1 - norm.cdf(d2_stop)
            
            prob_profit = prob_target * (1 - prob_stop)
            
            return max(0.1, min(0.9, prob_profit))
            
        except Exception as e:
            return 0.5
    
    @staticmethod
    def calculate_historical_volatility(price_data, window=30):
        """Calculate historical volatility from price data"""
        try:
            returns = price_data.pct_change().dropna()
            volatility = returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)
            return volatility if not np.isnan(volatility) else 0.2
        except:
            return 0.2

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
        
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        volume = self.data['Volume']
        
        indicators['SMA_20'] = TechnicalIndicators.sma(close, 20)
        indicators['SMA_50'] = TechnicalIndicators.sma(close, 50)
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
        
        # Moving Average signals
        sma_20 = indicators['SMA_20'].iloc[-1]
        sma_50 = indicators['SMA_50'].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            signals['MA_signal'] = 1
        elif current_price < sma_20 < sma_50:
            signals['MA_signal'] = -1
        else:
            signals['MA_signal'] = 0
        
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
        title="Advanced Technical Analysis Dashboard",
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

def calculate_position_sizing(current_price, atr, portfolio_value=10000, risk_percent=0.01):
    """Calculate recommended position size based on risk management"""
    risk_amount = portfolio_value * risk_percent
    stop_loss_distance = atr * 2
    stop_loss_price = current_price - stop_loss_distance
    
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
    
    # Portfolio settings
    st.sidebar.subheader("💰 Portfolio Settings")
    portfolio_value = st.sidebar.number_input("Portfolio Value ($)", value=10000, min_value=1000, step=1000)
    risk_percent = st.sidebar.slider("Risk per Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1) / 100
    
    # Probability settings
    st.sidebar.subheader("🎯 Probability Settings")
    days_to_target = st.sidebar.slider("Days to Target", min_value=5, max_value=90, value=30, step=5)
    risk_reward_ratio = st.sidebar.slider("Risk:Reward Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
    
    analysis_type = st.sidebar.multiselect(
        "Select Analysis Type",
        ["Technical Analysis", "Position Sizing", "Probability Analysis", "Fundamental Analysis", "News Sentiment"],
        default=["Technical Analysis", "Position Sizing", "Probability Analysis"]
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
                    
                    # Position Sizing and Probability Analysis
                    if ("Position Sizing" in analysis_type or "Probability Analysis" in analysis_type) and indicators:
                        st.markdown("## 💰 Recommended Position & Probability Analysis")
                        
                        atr_value = indicators['ATR'].iloc[-1]
                        position_data = calculate_position_sizing(current_price, atr_value, portfolio_value, risk_percent)
                        
                        # Calculate target price based on risk-reward ratio
                        target_price = current_price + (position_data['stop_loss_distance'] * risk_reward_ratio)
                        
                        # Calculate probabilities
                        prob_calc = ProbabilityCalculator()
                        
                        # Multi-indicator probability
                        multi_indicator_prob = prob_calc.calculate_multi_indicator_probability(tech_signals)
                        
                        # Historical volatility
                        historical_vol = prob_calc.calculate_historical_volatility(analyzer.data['Close'])
                        
                        # Stock probability of profit
                        stock_prob = prob_calc.calculate_stock_probability_of_profit(
                            current_price, target_price, position_data['stop_loss_price'], 
                            historical_vol, days_to_target
                        )
                        
                        # Combined probability (weighted average)
                        combined_prob = (multi_indicator_prob * 0.6 + stock_prob * 0.4)
                        
                        # Display position sizing
                        if "Position Sizing" in analysis_type:
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
                                st.metric("Target Price", f"${target_price:.2f}")
                        
                        # Display probability analysis
                        if "Probability Analysis" in analysis_type:
                            st.markdown("---")
                            st.subheader("🎯 Probability of Profit Analysis")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                prob_color = "green" if combined_prob >= 0.7 else "orange" if combined_prob >= 0.5 else "red"
                                st.markdown(f"**Combined PoP:** <span style='color: {prob_color}; font-size: 24px; font-weight: bold'>{combined_prob:.1%}</span>", unsafe_allow_html=True)
                            
                            with col2:
                                st.metric("Multi-Indicator PoP", f"{multi_indicator_prob:.1%}")
                            
                            with col3:
                                st.metric("Statistical PoP", f"{stock_prob:.1%}")
                            
                            with col4:
                                st.metric("Historical Volatility", f"{historical_vol:.1%}")
                            
                            # Probability breakdown
                            st.markdown("### 📊 Probability Breakdown")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Multi-Indicator Analysis:**")
                                active_signals = sum(1 for signal in ['MA_signal', 'MACD_signal', 'RSI_signal', 'BB_signal', 'STOCH_signal', 'SR_signal', 'pivot_signal', 'ADX_signal'] if tech_signals.get(signal, 0) != 0)
                                st.write(f"- Active signals: {active_signals}/8")
                                st.write(f"- Individual signal probability: ~35%")
                                st.write(f"- Combined probability: {multi_indicator_prob:.1%}")
                            
                            with col2:
                                st.markdown("**Statistical Analysis:**")
                                st.write(f"- Current price: ${current_price:.2f}")
                                st.write(f"- Target price: ${target_price:.2f}")
                                st.write(f"- Stop loss: ${position_data['stop_loss_price']:.2f}")
                                st.write(f"- Days to target: {days_to_target}")
                                st.write(f"- Statistical PoP: {stock_prob:.1%}")
                            
                            # Risk-Reward Analysis
                            st.markdown("### ⚖️ Risk-Reward Analysis")
                            
                            potential_profit = target_price - current_price
                            potential_loss = current_price - position_data['stop_loss_price']
                            actual_rr_ratio = potential_profit / potential_loss if potential_loss > 0 else 0
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Potential Profit", f"${potential_profit:.2f}")
                            
                            with col2:
                                st.metric("Potential Loss", f"${potential_loss:.2f}")
                            
                            with col3:
                                st.metric("Risk:Reward Ratio", f"1:{actual_rr_ratio:.1f}")
                            
                            # Expected Value Calculation
                            expected_value = (combined_prob * potential_profit) - ((1 - combined_prob) * potential_loss)
                            expected_value_per_share = expected_value
                            total_expected_value = expected_value_per_share * position_data['shares']
                            
                            st.markdown("### 💡 Expected Value Analysis")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                ev_color = "green" if expected_value > 0 else "red"
                                st.markdown(f"**Expected Value per Share:** <span style='color: {ev_color}'>${expected_value_per_share:.2f}</span>", unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"**Total Expected Value:** <span style='color: {ev_color}'>${total_expected_value:.2f}</span>", unsafe_allow_html=True)
                            
                            # Recommendation based on probability
                            st.markdown("### 🎯 Probability-Based Recommendation")
                            
                            if combined_prob >= 0.75:
                                rec_color = "green"
                                recommendation = "🟢 **HIGH PROBABILITY TRADE** - Strong signals with favorable odds"
                            elif combined_prob >= 0.60:
                                rec_color = "lightgreen"
                                recommendation = "🟢 **GOOD PROBABILITY TRADE** - Decent odds with manageable risk"
                            elif combined_prob >= 0.45:
                                rec_color = "orange"
                                recommendation = "🟡 **MODERATE PROBABILITY TRADE** - Consider reducing position size"
                            else:
                                rec_color = "red"
                                recommendation = "🔴 **LOW PROBABILITY TRADE** - High risk, consider avoiding"
                            
                            st.markdown(f"<div style='padding: 10px; border-left: 4px solid {rec_color}; background-color: rgba(128,128,128,0.1)'>{recommendation}</div>", unsafe_allow_html=True)
                    
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
                    
                    # Real News Sentiment Analysis (Updated Section)
                    if "News Sentiment" in analysis_type:
                        st.markdown("## 📰 Real News Sentiment Analysis")
                        
                        with st.spinner("Fetching real news data..."):
                            news_items = get_real_news_sentiment(symbol)
                        
                        if news_items:
                            # Overall sentiment summary
                            sentiment_counts = {}
                            total_polarity = 0
                            
                            for item in news_items:
                                sentiment = item['sentiment']
                                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
                                total_polarity += item['polarity_score']
                            
                            avg_polarity = total_polarity / len(news_items) if news_items else 0
                            
                            # Display sentiment summary
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                overall_sentiment = "Positive" if avg_polarity > 0.1 else "Negative" if avg_polarity < -0.1 else "Neutral"
                                sentiment_color = "green" if overall_sentiment == "Positive" else "red" if overall_sentiment == "Negative" else "orange"
                                st.markdown(f"**Overall Sentiment:** <span style='color: {sentiment_color}'>{overall_sentiment}</span>", unsafe_allow_html=True)
                            
                            with col2:
                                st.metric("Average Polarity", f"{avg_polarity:.3f}")
                            
                            with col3:
                                st.metric("Total Articles", len(news_items))
                            
                            with col4:
                                positive_pct = (sentiment_counts.get('Positive', 0) + sentiment_counts.get('Very Positive', 0)) / len(news_items) * 100
                                st.metric("Positive %", f"{positive_pct:.1f}%")
                            
                            # Sentiment breakdown chart
                            st.markdown("### 📊 Sentiment Distribution")
                            
                            sentiment_df = pd.DataFrame(list(sentiment_counts.items()), columns=['Sentiment', 'Count'])
                            
                            fig_sentiment = go.Figure(data=[
                                go.Bar(
                                    x=sentiment_df['Sentiment'],
                                    y=sentiment_df['Count'],
                                    marker_color=['darkgreen' if 'Positive' in s else 'darkred' if 'Negative' in s else 'gray' for s in sentiment_df['Sentiment']]
                                )
                            ])
                            
                            fig_sentiment.update_layout(
                                title="News Sentiment Distribution",
                                xaxis_title="Sentiment",
                                yaxis_title="Number of Articles",
                                template="plotly_dark"
                            )
                            
                            st.plotly_chart(fig_sentiment, use_container_width=True)
                            
                            # Individual news items
                            st.markdown("### 📰 Recent News Articles")
                            
                            for i, item in enumerate(news_items[:10]):  # Show top 10
                                with st.expander(f"📰 {item['title'][:80]}{'...' if len(item['title']) > 80 else ''} - {item['date']}"):
                                    col1, col2 = st.columns([3, 1])
                                    
                                    with col1:
                                        st.write(f"**Full Title:** {item['title']}")
                                        st.write(f"**Publisher:** {item['publisher']}")
                                        st.write(f"**Date:** {item['date']}")
                                        if item.get('url'):
                                            st.write(f"**Link:** [Read Full Article]({item['url']})")
                                    
                                    with col2:
                                        sentiment_color = "green" if "Positive" in item['sentiment'] else "red" if "Negative" in item['sentiment'] else "orange"
                                        st.markdown(f"**Sentiment:** <span style='color: {sentiment_color}'>{item['sentiment']}</span>", unsafe_allow_html=True)
                                        st.write(f"**Polarity:** {item['polarity_score']}")
                                        st.write(f"**Subjectivity:** {item['subjectivity_score']}")
                            
                            # News sentiment impact on trading decision
                            st.markdown("### 🎯 News Impact on Trading Decision")
                            
                            if avg_polarity > 0.2:
                                news_impact = "🟢 **POSITIVE NEWS CATALYST** - Recent news strongly supports bullish sentiment"
                                impact_color = "green"
                            elif avg_polarity > 0.05:
                                news_impact = "🟢 **MILDLY POSITIVE NEWS** - Recent news slightly supports bullish sentiment"
                                impact_color = "lightgreen"
                            elif avg_polarity < -0.2:
                                news_impact = "🔴 **NEGATIVE NEWS CATALYST** - Recent news strongly supports bearish sentiment"
                                impact_color = "red"
                            elif avg_polarity < -0.05:
                                news_impact = "🔴 **MILDLY NEGATIVE NEWS** - Recent news slightly supports bearish sentiment"
                                impact_color = "lightcoral"
                            else:
                                news_impact = "🟡 **NEUTRAL NEWS** - Recent news shows mixed or neutral sentiment"
                                impact_color = "orange"
                            
                            st.markdown(f"<div style='padding: 10px; border-left: 4px solid {impact_color}; background-color: rgba(128,128,128,0.1)'>{news_impact}</div>", unsafe_allow_html=True)
                        
                        else:
                            st.warning(f"No recent news found for {symbol}. This could be due to:")
                            st.write("- Limited news coverage for this stock")
                            st.write("- API limitations or connectivity issues")
                            st.write("- Stock symbol not recognized by news sources")
                
                else:
                    st.error("Failed to fetch stock data. Please check the symbol and try again.")
        else:
            st.warning("Please enter a stock symbol.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Disclaimer:** This analysis is for educational purposes only and should not be considered as financial advice.")
    st.markdown("**News Data:** Real-time news sentiment analysis powered by Yahoo Finance and TextBlob.")

if __name__ == "__main__":
    main()

