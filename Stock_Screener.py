import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import talib
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

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .signal-buy {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .signal-sell {
        background: linear-gradient(135deg, #f44336, #da190b);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .signal-hold {
        background: linear-gradient(135deg, #ff9800, #f57c00);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

class StockAnalyzer:
    def __init__(self, symbol, period="1y"):
        self.symbol = symbol
        self.period = period
        self.stock = yf.Ticker(symbol)
        self.data = None
        self.info = None
        
    def fetch_data(self):
        """Fetch stock data and company information"""[1]
        try:
            self.data = self.stock.history(period=self.period)
            self.info = self.stock.info
            return True
        except Exception as e:
            st.error(f"Error fetching data for {self.symbol}: {str(e)}")
            return False
    
    def calculate_technical_indicators(self):
        """Calculate various technical indicators"""[7]
        if self.data is None or len(self.data) < 50:
            return None
            
        # Price data
        high = self.data['High'].values
        low = self.data['Low'].values
        close = self.data['Close'].values
        volume = self.data['Volume'].values
        
        indicators = {}
        
        # Moving Averages
        indicators['SMA_20'] = talib.SMA(close, timeperiod=20)
        indicators['SMA_50'] = talib.SMA(close, timeperiod=50)
        indicators['SMA_200'] = talib.SMA(close, timeperiod=200)
        indicators['EMA_12'] = talib.EMA(close, timeperiod=12)
        indicators['EMA_26'] = talib.EMA(close, timeperiod=26)
        
        # MACD
        indicators['MACD'], indicators['MACD_signal'], indicators['MACD_hist'] = talib.MACD(close)
        
        # RSI
        indicators['RSI'] = talib.RSI(close, timeperiod=14)
        
        # Bollinger Bands
        indicators['BB_upper'], indicators['BB_middle'], indicators['BB_lower'] = talib.BBANDS(close)
        
        # Stochastic
        indicators['STOCH_K'], indicators['STOCH_D'] = talib.STOCH(high, low, close)
        
        # Volume indicators
        indicators['OBV'] = talib.OBV(close, volume)
        
        # ATR
        indicators['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        
        # Williams %R
        indicators['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)
        
        return indicators
    
    def get_fundamental_data(self):
        """Extract fundamental analysis data"""[10][11]
        if self.info is None:
            return None
            
        fundamentals = {}
        
        # Financial metrics
        fundamentals['PE_ratio'] = self.info.get('trailingPE', 0)
        fundamentals['PEG_ratio'] = self.info.get('pegRatio', 0)
        fundamentals['Price_to_book'] = self.info.get('priceToBook', 0)
        fundamentals['Debt_to_equity'] = self.info.get('debtToEquity', 0)
        fundamentals['ROE'] = self.info.get('returnOnEquity', 0)
        fundamentals['ROA'] = self.info.get('returnOnAssets', 0)
        fundamentals['Profit_margin'] = self.info.get('profitMargins', 0)
        fundamentals['Revenue_growth'] = self.info.get('revenueGrowth', 0)
        fundamentals['Earnings_growth'] = self.info.get('earningsGrowth', 0)
        fundamentals['Current_ratio'] = self.info.get('currentRatio', 0)
        fundamentals['Quick_ratio'] = self.info.get('quickRatio', 0)
        fundamentals['Dividend_yield'] = self.info.get('dividendYield', 0)
        fundamentals['Market_cap'] = self.info.get('marketCap', 0)
        fundamentals['Enterprise_value'] = self.info.get('enterpriseValue', 0)
        fundamentals['Beta'] = self.info.get('beta', 1)
        
        return fundamentals
    
    def generate_technical_signals(self, indicators):
        """Generate buy/sell signals based on technical indicators"""[2][7]
        signals = {}
        current_price = self.data['Close'].iloc[-1]
        
        # Moving Average Signals
        sma_20 = indicators['SMA_20'][-1] if not np.isnan(indicators['SMA_20'][-1]) else 0
        sma_50 = indicators['SMA_50'][-1] if not np.isnan(indicators['SMA_50'][-1]) else 0
        sma_200 = indicators['SMA_200'][-1] if not np.isnan(indicators['SMA_200'][-1]) else 0
        
        signals['MA_signal'] = 0
        if current_price > sma_20 > sma_50 > sma_200:
            signals['MA_signal'] = 2  # Strong buy
        elif current_price > sma_20 > sma_50:
            signals['MA_signal'] = 1  # Buy
        elif current_price < sma_20 < sma_50 < sma_200:
            signals['MA_signal'] = -2  # Strong sell
        elif current_price < sma_20 < sma_50:
            signals['MA_signal'] = -1  # Sell
        
        # MACD Signal
        macd = indicators['MACD'][-1] if not np.isnan(indicators['MACD'][-1]) else 0
        macd_signal = indicators['MACD_signal'][-1] if not np.isnan(indicators['MACD_signal'][-1]) else 0
        
        if macd > macd_signal and macd > 0:
            signals['MACD_signal'] = 1
        elif macd < macd_signal and macd < 0:
            signals['MACD_signal'] = -1
        else:
            signals['MACD_signal'] = 0
        
        # RSI Signal
        rsi = indicators['RSI'][-1] if not np.isnan(indicators['RSI'][-1]) else 50
        if rsi < 30:
            signals['RSI_signal'] = 1  # Oversold - Buy
        elif rsi > 70:
            signals['RSI_signal'] = -1  # Overbought - Sell
        else:
            signals['RSI_signal'] = 0
        
        # Bollinger Bands Signal
        bb_upper = indicators['BB_upper'][-1] if not np.isnan(indicators['BB_upper'][-1]) else current_price
        bb_lower = indicators['BB_lower'][-1] if not np.isnan(indicators['BB_lower'][-1]) else current_price
        
        if current_price <= bb_lower:
            signals['BB_signal'] = 1  # Buy
        elif current_price >= bb_upper:
            signals['BB_signal'] = -1  # Sell
        else:
            signals['BB_signal'] = 0
        
        # Stochastic Signal
        stoch_k = indicators['STOCH_K'][-1] if not np.isnan(indicators['STOCH_K'][-1]) else 50
        if stoch_k < 20:
            signals['STOCH_signal'] = 1
        elif stoch_k > 80:
            signals['STOCH_signal'] = -1
        else:
            signals['STOCH_signal'] = 0
        
        return signals
    
    def generate_fundamental_signals(self, fundamentals):
        """Generate buy/sell signals based on fundamental analysis"""[10][11]
        if fundamentals is None:
            return {}
            
        signals = {}
        score = 0
        
        # PE Ratio analysis
        pe_ratio = fundamentals.get('PE_ratio', 0)
        if 0 < pe_ratio < 15:
            score += 2  # Undervalued
        elif 15 <= pe_ratio <= 25:
            score += 1  # Fair value
        elif pe_ratio > 30:
            score -= 1  # Overvalued
        
        # PEG Ratio analysis
        peg_ratio = fundamentals.get('PEG_ratio', 0)
        if 0 < peg_ratio < 1:
            score += 2  # Undervalued growth
        elif 1 <= peg_ratio <= 1.5:
            score += 1  # Fair growth value
        
        # ROE analysis
        roe = fundamentals.get('ROE', 0)
        if roe > 0.15:  # 15%
            score += 2
        elif roe > 0.10:  # 10%
            score += 1
        
        # Debt to Equity analysis
        debt_to_equity = fundamentals.get('Debt_to_equity', 0)
        if debt_to_equity < 0.3:
            score += 1
        elif debt_to_equity > 1:
            score -= 1
        
        # Revenue and Earnings Growth
        revenue_growth = fundamentals.get('Revenue_growth', 0)
        earnings_growth = fundamentals.get('Earnings_growth', 0)
        
        if revenue_growth > 0.1:  # 10% growth
            score += 1
        if earnings_growth > 0.1:  # 10% growth
            score += 1
        
        # Current Ratio
        current_ratio = fundamentals.get('Current_ratio', 0)
        if current_ratio > 1.5:
            score += 1
        elif current_ratio < 1:
            score -= 1
        
        signals['fundamental_score'] = score
        
        if score >= 5:
            signals['fundamental_signal'] = 2  # Strong buy
        elif score >= 3:
            signals['fundamental_signal'] = 1  # Buy
        elif score <= -2:
            signals['fundamental_signal'] = -1  # Sell
        else:
            signals['fundamental_signal'] = 0  # Hold
        
        return signals
    
    def calculate_position_size(self, portfolio_value, risk_percentage, atr_value, current_price):
        """Calculate position size based on risk management"""[1]
        risk_amount = portfolio_value * (risk_percentage / 100)
        
        # Use ATR for stop loss calculation
        stop_loss_distance = atr_value * 2  # 2x ATR for stop loss
        
        if stop_loss_distance > 0:
            shares = int(risk_amount / stop_loss_distance)
            position_value = shares * current_price
            
            # Don't risk more than specified percentage
            if position_value > portfolio_value * 0.1:  # Max 10% position size
                shares = int((portfolio_value * 0.1) / current_price)
                
            return max(shares, 0)
        
        return 0

def create_price_chart(data, indicators):
    """Create comprehensive price chart with technical indicators"""[4]
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Moving Averages', 'MACD', 'RSI', 'Volume'),
        row_width=[0.4, 0.2, 0.2, 0.2]
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
        go.Scatter(x=data.index, y=indicators['SMA_20'], name='SMA 20', line=dict(color='orange')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=indicators['SMA_50'], name='SMA 50', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=indicators['SMA_200'], name='SMA 200', line=dict(color='red')),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=data.index, y=indicators['BB_upper'], name='BB Upper', 
                  line=dict(color='gray', dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=indicators['BB_lower'], name='BB Lower', 
                  line=dict(color='gray', dash='dash')),
        row=1, col=1
    )
    
    # MACD
    fig.add_trace(
        go.Scatter(x=data.index, y=indicators['MACD'], name='MACD', line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=indicators['MACD_signal'], name='Signal', line=dict(color='red')),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=data.index, y=indicators['MACD_hist'], name='Histogram'),
        row=2, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=data.index, y=indicators['RSI'], name='RSI', line=dict(color='purple')),
        row=3, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # Volume
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'),
        row=4, col=1
    )
    
    fig.update_layout(
        title="Technical Analysis Dashboard",
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True,
        template="plotly_dark"
    )
    
    return fig

def main():
    st.markdown('<h1 class="main-header">🚀 Advanced Stock Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Analysis Parameters")
        
        # Stock selection
        symbol = st.text_input("Enter Stock Symbol", value="AAPL", help="Enter stock ticker (e.g., AAPL, GOOGL, TSLA)")
        
        # Time period
        period = st.selectbox(
            "Select Time Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3
        )
        
        # Portfolio settings
        st.markdown("### 💰 Portfolio Settings")
        portfolio_value = st.number_input("Portfolio Value ($)", min_value=1000, value=10000, step=1000)
        risk_percentage = st.slider("Risk per Trade (%)", min_value=1, max_value=5, value=2)
        
        # Analysis button
        analyze_button = st.button("🔍 Analyze Stock", type="primary")
    
    if analyze_button and symbol:
        # Initialize analyzer
        analyzer = StockAnalyzer(symbol.upper(), period)
        
        with st.spinner(f"Analyzing {symbol.upper()}..."):
            if analyzer.fetch_data():
                # Calculate indicators
                indicators = analyzer.calculate_technical_indicators()
                fundamentals = analyzer.get_fundamental_data()
                
                if indicators is not None:
                    # Generate signals
                    tech_signals = analyzer.generate_technical_signals(indicators)
                    fund_signals = analyzer.generate_fundamental_signals(fundamentals)
                    
                    # Current price and ATR
                    current_price = analyzer.data['Close'].iloc[-1]
                    atr_value = indicators['ATR'][-1] if not np.isnan(indicators['ATR'][-1]) else current_price * 0.02
                    
                    # Calculate position size
                    position_size = analyzer.calculate_position_size(
                        portfolio_value, risk_percentage, atr_value, current_price
                    )
                    
                    # Display results
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        # Company info
                        if analyzer.info:
                            st.markdown(f"### 🏢 {analyzer.info.get('longName', symbol.upper())}")
                            st.markdown(f"**Sector:** {analyzer.info.get('sector', 'N/A')} | **Industry:** {analyzer.info.get('industry', 'N/A')}")
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f'<div class="metric-card"><h3>${current_price:.2f}</h3><p>Current Price</p></div>', unsafe_allow_html=True)
                    
                    with col2:
                        change = ((current_price - analyzer.data['Close'].iloc[-2]) / analyzer.data['Close'].iloc[-2]) * 100
                        color = "green" if change > 0 else "red"
                        st.markdown(f'<div class="metric-card"><h3 style="color: {color};">{change:+.2f}%</h3><p>Daily Change</p></div>', unsafe_allow_html=True)
                    
                    with col3:
                        volume = analyzer.data['Volume'].iloc[-1]
                        st.markdown(f'<div class="metric-card"><h3>{volume:,.0f}</h3><p>Volume</p></div>', unsafe_allow_html=True)
                    
                    with col4:
                        market_cap = fundamentals.get('Market_cap', 0) if fundamentals else 0
                        if market_cap > 1e9:
                            market_cap_str = f"${market_cap/1e9:.1f}B"
                        elif market_cap > 1e6:
                            market_cap_str = f"${market_cap/1e6:.1f}M"
                        else:
                            market_cap_str = "N/A"
                        st.markdown(f'<div class="metric-card"><h3>{market_cap_str}</h3><p>Market Cap</p></div>', unsafe_allow_html=True)
                    
                    # Overall signal calculation
                    tech_score = sum([
                        tech_signals.get('MA_signal', 0),
                        tech_signals.get('MACD_signal', 0),
                        tech_signals.get('RSI_signal', 0),
                        tech_signals.get('BB_signal', 0),
                        tech_signals.get('STOCH_signal', 0)
                    ])
                    
                    fund_score = fund_signals.get('fundamental_signal', 0)
                    overall_score = tech_score + fund_score * 2  # Weight fundamental analysis more
                    
                    # Signal interpretation
                    if overall_score >= 4:
                        signal = "STRONG BUY"
                        signal_class = "signal-buy"
                        action = "BUY"
                    elif overall_score >= 2:
                        signal = "BUY"
                        signal_class = "signal-buy"
                        action = "BUY"
                    elif overall_score <= -4:
                        signal = "STRONG SELL"
                        signal_class = "signal-sell"
                        action = "SELL"
                    elif overall_score <= -2:
                        signal = "SELL"
                        signal_class = "signal-sell"
                        action = "SELL"
                    else:
                        signal = "HOLD"
                        signal_class = "signal-hold"
                        action = "HOLD"
                    
                    # Display signal
                    st.markdown(f'<div class="{signal_class}">📈 SIGNAL: {signal}</div>', unsafe_allow_html=True)
                    
                    # Position sizing
                    if action in ["BUY", "STRONG BUY"] and position_size > 0:
                        position_value = position_size * current_price
                        st.markdown(f"""
                        ### 💼 **Recommended Position:**
                        - **Action:** {action}
                        - **Quantity:** {position_size:,} shares
                        - **Position Value:** ${position_value:,.2f}
                        - **Risk Amount:** ${portfolio_value * (risk_percentage/100):,.2f} ({risk_percentage}% of portfolio)
                        - **Stop Loss:** ${current_price - (atr_value * 2):.2f} (2x ATR)
                        """)
                    elif action in ["SELL", "STRONG SELL"]:
                        st.markdown(f"""
                        ### 💼 **Recommended Action:**
                        - **Action:** {action}
                        - **Recommendation:** Consider selling existing positions
                        - **Stop Loss:** ${current_price + (atr_value * 2):.2f} (for short positions)
                        """)
                    else:
                        st.markdown(f"""
                        ### 💼 **Recommended Action:**
                        - **Action:** {action}
                        - **Recommendation:** Maintain current positions, monitor for changes
                        """)
                    
                    # Technical Analysis Details
                    st.markdown("## 📊 Technical Analysis Breakdown")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📈 Technical Indicators")
                        
                        # Create indicator signals dataframe
                        tech_df = pd.DataFrame({
                            'Indicator': ['Moving Averages', 'MACD', 'RSI', 'Bollinger Bands', 'Stochastic'],
                            'Signal': [
                                '🟢 Buy' if tech_signals.get('MA_signal', 0) > 0 else '🔴 Sell' if tech_signals.get('MA_signal', 0) < 0 else '🟡 Neutral',
                                '🟢 Buy' if tech_signals.get('MACD_signal', 0) > 0 else '🔴 Sell' if tech_signals.get('MACD_signal', 0) < 0 else '🟡 Neutral',
                                '🟢 Buy' if tech_signals.get('RSI_signal', 0) > 0 else '🔴 Sell' if tech_signals.get('RSI_signal', 0) < 0 else '🟡 Neutral',
                                '🟢 Buy' if tech_signals.get('BB_signal', 0) > 0 else '🔴 Sell' if tech_signals.get('BB_signal', 0) < 0 else '🟡 Neutral',
                                '🟢 Buy' if tech_signals.get('STOCH_signal', 0) > 0 else '🔴 Sell' if tech_signals.get('STOCH_signal', 0) < 0 else '🟡 Neutral'
                            ],
                            'Value': [
                                f"SMA20: ${indicators['SMA_20'][-1]:.2f}" if not np.isnan(indicators['SMA_20'][-1]) else "N/A",
                                f"MACD: {indicators['MACD'][-1]:.3f}" if not np.isnan(indicators['MACD'][-1]) else "N/A",
                                f"RSI: {indicators['RSI'][-1]:.1f}" if not np.isnan(indicators['RSI'][-1]) else "N/A",
                                f"Price vs BB: {'Upper' if current_price >= indicators['BB_upper'][-1] else 'Lower' if current_price <= indicators['BB_lower'][-1] else 'Middle'}",
                                f"Stoch %K: {indicators['STOCH_K'][-1]:.1f}" if not np.isnan(indicators['STOCH_K'][-1]) else "N/A"
                            ]
                        })
                        
                        st.dataframe(tech_df, use_container_width=True, hide_index=True)
                    
                    with col2:
                        if fundamentals:
                            st.markdown("### 🏛️ Fundamental Metrics")
                            
                            fund_df = pd.DataFrame({
                                'Metric': ['P/E Ratio', 'PEG Ratio', 'ROE', 'Debt/Equity', 'Profit Margin', 'Revenue Growth'],
                                'Value': [
                                    f"{fundamentals.get('PE_ratio', 0):.2f}" if fundamentals.get('PE_ratio', 0) > 0 else "N/A",
                                    f"{fundamentals.get('PEG_ratio', 0):.2f}" if fundamentals.get('PEG_ratio', 0) > 0 else "N/A",
                                    f"{fundamentals.get('ROE', 0)*100:.1f}%" if fundamentals.get('ROE', 0) > 0 else "N/A",
                                    f"{fundamentals.get('Debt_to_equity', 0):.2f}" if fundamentals.get('Debt_to_equity', 0) > 0 else "N/A",
                                    f"{fundamentals.get('Profit_margin', 0)*100:.1f}%" if fundamentals.get('Profit_margin', 0) > 0 else "N/A",
                                    f"{fundamentals.get('Revenue_growth', 0)*100:.1f}%" if fundamentals.get('Revenue_growth', 0) != 0 else "N/A"
                                ],
                                'Rating': [
                                    '🟢 Good' if 0 < fundamentals.get('PE_ratio', 0) < 20 else '🟡 Fair' if fundamentals.get('PE_ratio', 0) < 30 else '🔴 High',
                                    '🟢 Good' if 0 < fundamentals.get('PEG_ratio', 0) < 1 else '🟡 Fair' if fundamentals.get('PEG_ratio', 0) < 1.5 else '🔴 High',
                                    '🟢 Good' if fundamentals.get('ROE', 0) > 0.15 else '🟡 Fair' if fundamentals.get('ROE', 0) > 0.1 else '🔴 Low',
                                    '🟢 Good' if fundamentals.get('Debt_to_equity', 0) < 0.3 else '🟡 Fair' if fundamentals.get('Debt_to_equity', 0) < 1 else '🔴 High',
                                    '🟢 Good' if fundamentals.get('Profit_margin', 0) > 0.1 else '🟡 Fair' if fundamentals.get('Profit_margin', 0) > 0.05 else '🔴 Low',
                                    '🟢 Good' if fundamentals.get('Revenue_growth', 0) > 0.1 else '🟡 Fair' if fundamentals.get('Revenue_growth', 0) > 0 else '🔴 Negative'
                                ]
                            })
                            
                            st.dataframe(fund_df, use_container_width=True, hide_index=True)
                    
                    # Charts
                    st.markdown("## 📊 Technical Analysis Charts")
                    
                    # Create and display the main chart
                    chart = create_price_chart(analyzer.data, indicators)
                    st.plotly_chart(chart, use_container_width=True)
                    
                    # Additional analysis
                    st.markdown("## 🎯 Signal Reasoning")
                    
                    reasoning = []
                    
                    # Technical reasoning
                    if tech_signals.get('MA_signal', 0) > 0:
                        reasoning.append("✅ **Moving Averages**: Price is above key moving averages, indicating upward momentum")
                    elif tech_signals.get('MA_signal', 0) < 0:
                        reasoning.append("❌ **Moving Averages**: Price is below key moving averages, indicating downward pressure")
                    
                    if tech_signals.get('MACD_signal', 0) > 0:
                        reasoning.append("✅ **MACD**: MACD line above signal line with positive momentum")
                    elif tech_signals.get('MACD_signal', 0) < 0:
                        reasoning.append("❌ **MACD**: MACD line below signal line with negative momentum")
                    
                    if tech_signals.get('RSI_signal', 0) > 0:
                        reasoning.append("✅ **RSI**: Stock is oversold (RSI < 30), potential buying opportunity")
                    elif tech_signals.get('RSI_signal', 0) < 0:
                        reasoning.append("❌ **RSI**: Stock is overbought (RSI > 70), potential selling opportunity")
                    
                    # Fundamental reasoning
                    if fund_signals.get('fundamental_signal', 0) > 0:
                        reasoning.append("✅ **Fundamentals**: Strong financial metrics support buying")
                    elif fund_signals.get('fundamental_signal', 0) < 0:
                        reasoning.append("❌ **Fundamentals**: Weak financial metrics suggest caution")
                    
                    for reason in reasoning:
                        st.markdown(reason)
                    
                    if not reasoning:
                        st.markdown("🟡 **Mixed Signals**: Technical and fundamental indicators are showing conflicting or neutral signals. Consider waiting for clearer direction.")
                    
                    # Risk disclaimer
                    st.markdown("---")
                    st.markdown("""
                    **⚠️ Risk Disclaimer:** This analysis is for educational purposes only and should not be considered as financial advice. 
                    Always conduct your own research and consider consulting with a financial advisor before making investment decisions. 
                    Past performance does not guarantee future results.
                    """)
                
                else:
                    st.error("Unable to calculate technical indicators. Please try a different stock or time period.")
            else:
                st.error("Unable to fetch stock data. Please check the symbol and try again.")

if __name__ == "__main__":
    main()
