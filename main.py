import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import Dict, Optional, Tuple
import os
import io
import requests
from PIL import Image


# Telegram API Configuration
API_TOKEN = "7594281395:AAHZq1bz9Ym5-9bjjtMs_kcwS5CzUjznMEE"
CHAT_ID = "7311904934"

# Output directory
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def sendmessage(message):
    """Send text message via Telegram Bot API."""
    apiToken = API_TOKEN
    chatID = CHAT_ID
    apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'
    try:
        response = requests.post(
            apiURL, json={'chat_id': chatID, 'text': message})
        print(response.text)
    except Exception as e:
        print(f"Error sending message: {e}")

def sendphoto(image):
    """Send photo via Telegram Bot API."""
    apiToken = API_TOKEN
    chatID = CHAT_ID
    apiURL = f'https://api.telegram.org/bot{apiToken}/sendPhoto'
    files = {'photo': open(image, 'rb')}
    try:
        response = requests.post(apiURL, files=files, data={'chat_id': chatID})
        print(response.text)
    except Exception as e:
        print(f"Error sending photo: {e}")
    finally:
        files['photo'].close()


def get_trading_days(df: pd.DataFrame, n_days: int = 5) -> pd.DataFrame | None:
    """
    Filter dataframe to last N trading days.
    Returns None if fewer than n_days trading days are available.
    """
    df = df.copy()
    df['_date'] = pd.to_datetime(df['timestamp']).dt.date
    unique_dates = sorted(df['_date'].unique())

    if len(unique_dates) < n_days:
        return None

    last_n_dates = unique_dates[-n_days:]
    filtered = df[df['_date'].isin(last_n_dates)].copy()
    filtered.drop(columns=['_date'], inplace=True)
    return filtered


class BitcoinFuturesAnalyzer:
    """Bitcoin futures analysis and visualization tool."""
    
    # Color scheme
    COLORS = {
        'spot': '#A9A9A9',           # DarkGray
        'futures': '#FF69B4',        # HotPink
        'volume': '#FFC0CB',         # Pink
        'difference': '#FF69B4',     # HotPink
        'apy': '#FF69B4'            # HotPink
    }
    
    def __init__(self, futures_symbol: str, expiry_date: datetime,
                 label: str, lookback: int = 762):
        self.futures_symbol = futures_symbol
        self.expiry_date = expiry_date
        self.label = label
        self.lookback = lookback
        
    def load_and_process_data(self) -> pd.DataFrame:
        """Load and process spot and futures data."""
        # Load data
        spot_data = self._load_asset_data('BINANCE:BTCUSDT', ['time', 'close'])
        futures_data = self._load_asset_data(self.futures_symbol, ['time', 'close', 'volume'])
        
        # Merge and rename columns
        data = spot_data.merge(futures_data, on='time', how='inner')
        data = data.rename(columns={
            'time': 'timestamp',
            'close_x': 'spot_price',
            'close_y': 'futures_price',
            'volume': 'futures_volume'
        })
        
        # Calculate derived metrics
        data = self._calculate_metrics(data, self.expiry_date)
        
        return data.dropna().reset_index(drop=True)
    
    def _load_asset_data(self, symbol: str, columns: list) -> pd.DataFrame:
        """Load asset price data."""
        from price_loaders.tradingview import load_asset_price
        return load_asset_price(symbol, self.lookback, "1", None)[columns]
    
    def _calculate_metrics(self, data: pd.DataFrame, expiry_date: datetime) -> pd.DataFrame:
        """Calculate futures basis, days to expiry, and APY."""
        # Convert timestamps
        data['timestamp'] = pd.to_datetime(data['timestamp']).dt.tz_localize(None)
        
        # Calculate futures basis (as percentage)
        data['futures_basis_pct'] = ((data['futures_price'] / data['spot_price']) - 1) * 100
        
        # Calculate days to expiry
        data['days_to_expiry'] = (expiry_date - data['timestamp']).dt.days
        
        # Calculate APY (avoid division by zero)
        mask = data['days_to_expiry'] > 0
        data.loc[mask, 'apy'] = (data.loc[mask, 'futures_basis_pct'] * 365 / data.loc[mask, 'days_to_expiry'])
        
        return data
    
    def create_visualization(self, data: pd.DataFrame) -> plt.Figure:
        """Create comprehensive futures analysis visualization."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        # Configure plot style
        plt.style.use('seaborn-v0_8-whitegrid')  # Modern clean style
        
        # Plot 1: Price comparison with volume
        self._plot_prices_and_volume(axes[0], data)
        
        # Plot 2: Futures basis
        self._plot_futures_basis(axes[1], data)
        
        # Plot 3: APY
        self._plot_apy(axes[2], data)
        
        # Format x-axis for all subplots
        self._format_x_axis(axes)
        
        # Final layout adjustments
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(top=0.95, bottom=0.15)  # More space for rotated labels
        
        return fig

    def create_intraday_visualization(self, data: pd.DataFrame,
                                       n_trading_days: int = 5) -> plt.Figure | None:
        """
        Create intraday visualization for the last N trading days.
        Returns None if fewer than n_trading_days are available.
        """
        df_filtered = get_trading_days(data, n_trading_days)
        if df_filtered is None:
            print(f"  [Intraday] Skipped: fewer than {n_trading_days} trading days "
                  f"for {self.label}")
            return None

        df_filtered = df_filtered.sort_values('timestamp').reset_index(drop=True)

        fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)

        # Configure plot style
        plt.style.use('seaborn-v0_8-whitegrid')

        # Plot 1: Price comparison with volume
        self._plot_prices_and_volume(axes[0], df_filtered)
        axes[0].set_title(
            f'Bitcoin: Spot vs Futures – Last {n_trading_days} Days ({self.label})',
            fontsize=14, fontweight='bold', pad=20
        )

        # Plot 2: Futures basis
        self._plot_futures_basis(axes[1], df_filtered)

        # Plot 3: APY
        self._plot_apy(axes[2], df_filtered)

        # Add day boundary vertical lines
        dates = df_filtered['timestamp'].dt.date
        for i in range(1, len(df_filtered)):
            if dates.iloc[i] != dates.iloc[i - 1]:
                for ax in axes:
                    ax.axvline(x=df_filtered['timestamp'].iloc[i], color='gray',
                               linestyle='-', linewidth=0.8, alpha=0.5)

        # Format x-axis for all subplots
        self._format_x_axis(axes)

        # Final layout adjustments
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(top=0.95, bottom=0.15)

        return fig
    
    def _plot_prices_and_volume(self, ax: plt.Axes, data: pd.DataFrame) -> None:
        """Plot spot and futures prices with volume overlay."""
        # Price lines
        ax.plot(data['timestamp'], data['spot_price'], 
                label='Spot Price', color=self.COLORS['spot'], linewidth=2)
        ax.plot(data['timestamp'], data['futures_price'], 
                label='Futures Price', color=self.COLORS['futures'], linewidth=2)
        
        # Volume on secondary axis
        ax2 = ax.twinx()
        ax2.plot(data['timestamp'], data['futures_volume'], 
                 label='Futures Volume', color=self.COLORS['volume'], 
                 linewidth=0.8, linestyle='--', alpha=0.7)
        
        # Formatting
        ax.set_ylabel('Price (USDT)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Volume', fontsize=11, fontweight='bold')
        ax.set_title('Bitcoin: Spot vs Futures Prices', fontsize=14, fontweight='bold', pad=20)
        
        # Legends
        ax.legend(loc='upper left', fontsize=10)
        ax2.legend(loc='upper right', fontsize=10)
        
        # Grid
        ax.grid(True, linestyle='--', alpha=0.3)
    
    def _plot_futures_basis(self, ax: plt.Axes, data: pd.DataFrame) -> None:
        """Plot futures basis (price difference percentage)."""
        ax.plot(data['timestamp'], data['futures_basis_pct'], 
                color=self.COLORS['difference'], linewidth=1.5)
        
        # Formatting
        ax.set_ylabel('Basis (%)', fontsize=11, fontweight='bold')
        ax.set_title('Futures Basis (Premium/Discount)', fontsize=14, fontweight='bold', pad=20)
        
        # Custom y-axis ticks - wider spacing for better readability
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        
        # Determine appropriate tick spacing with wider intervals
        if y_range < 0.2:
            tick_step = 0.02    # 0.02% steps for very small ranges
        elif y_range < 0.5:
            tick_step = 0.05    # 0.05% steps for small ranges
        elif y_range < 1.0:
            tick_step = 0.1     # 0.1% steps for medium ranges
        elif y_range < 2.0:
            tick_step = 0.2     # 0.2% steps for larger ranges
        else:
            tick_step = 0.5     # 0.5% steps for very large ranges
            
        tick_start = np.floor(y_min / tick_step) * tick_step
        tick_end = np.ceil(y_max / tick_step) * tick_step
        tick_range = np.arange(tick_start, tick_end + tick_step, tick_step)
        
        # Limit the number of ticks to avoid overcrowding (max 10 ticks)
        if len(tick_range) > 10:
            tick_step = tick_step * 2  # Double the step size
            tick_start = np.floor(y_min / tick_step) * tick_step
            tick_end = np.ceil(y_max / tick_step) * tick_step
            tick_range = np.arange(tick_start, tick_end + tick_step, tick_step)
        
        ax.set_yticks(tick_range)
        ax.set_ylim(y_min, y_max)  # Maintain original range
        
        # Format y-axis labels - show appropriate decimal places based on tick step
        if tick_step >= 0.1:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        
        ax.grid(True, linestyle='--', alpha=0.3)
    
    def _plot_apy(self, ax: plt.Axes, data: pd.DataFrame) -> None:
        """Plot Annual Percentage Yield."""
        # Filter out extreme values for better visualization
        clean_data = data[np.abs(data['apy']) < 100]  # Remove outliers
        
        ax.plot(clean_data['timestamp'], clean_data['apy'], 
                color=self.COLORS['apy'], linewidth=1.5)
        
        # Formatting
        ax.set_ylabel('APY (%)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax.set_title('Annualized Yield from Futures Basis', fontsize=14, fontweight='bold', pad=20)
        
        # Custom y-axis with proper formatting
        y_min, y_max = ax.get_ylim()
        
        # Determine appropriate tick spacing
        y_range = y_max - y_min
        if y_range < 2:
            tick_step = 0.2
        elif y_range < 5:
            tick_step = 0.5
        elif y_range < 10:
            tick_step = 1.0
        else:
            tick_step = 2.0
            
        tick_start = np.floor(y_min / tick_step) * tick_step
        tick_end = np.ceil(y_max / tick_step) * tick_step
        tick_range = np.arange(tick_start, tick_end + tick_step, tick_step)
        ax.set_yticks(tick_range)
        ax.set_ylim(y_min, y_max)  # Maintain original range
        
        # Format y-axis labels
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        
        ax.grid(True, linestyle='--', alpha=0.3)
    
    def _format_x_axis(self, axes: list) -> None:
        """Format x-axis for all subplots."""
        for ax in axes:
            # Date formatting - every 1 hour with full timestamp format
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
            
            # Minor ticks every 30 minutes for finer granularity
            ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
            
            # Rotate labels for better readability with longer format
            for label in ax.get_xticklabels():
                label.set_rotation(90)  # Vertical rotation for long timestamps
                label.set_horizontalalignment('center')
                label.set_fontsize(8)  # Smaller font for longer timestamps
            
            # Grid - major grid every hour, minor grid every 30 minutes
            ax.xaxis.grid(True, which='major', alpha=0.5, linewidth=0.8)
            ax.xaxis.grid(True, which='minor', alpha=0.2, linewidth=0.4)
    
    def save_chart_as_image(self, fig: plt.Figure, filename: str = 'Binance_chart.png', 
                           dpi: int = 300) -> str:
        """Save matplotlib figure as high-quality PNG image."""
        try:
            filepath = os.path.join(OUTPUT_DIR, filename)
            # Save the figure with high DPI for better quality
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', 
                       format='png', pad_inches=0.2)
            
            print(f"Chart saved as {filepath}")
            return os.path.abspath(filepath)
            
        except Exception as e:
            print(f"Error saving chart: {e}")
            return None
    
    def generate_summary_stats(self, data: pd.DataFrame) -> Dict[str, float]:
        """Generate summary statistics."""
        clean_apy = data['apy'].dropna()
        clean_apy = clean_apy[np.abs(clean_apy) < 100]  # Remove outliers
        
        return {
            'avg_basis_pct': data['futures_basis_pct'].mean(),
            'current_basis_pct': data['futures_basis_pct'].iloc[-1],
            'avg_apy': clean_apy.mean(),
            'current_apy': clean_apy.iloc[-1] if len(clean_apy) > 0 else np.nan,
            'days_remaining': data['days_to_expiry'].iloc[-1]
        }


# =============================================================================
# CONTRACT CONFIGURATIONS
# =============================================================================
CONTRACT_CONFIGS = [
    {
        'futures_symbol': 'BINANCE:BTCUSDH2026',
        'expiry_date': datetime(2026, 3, 27, 8, 0, 0),
        'label': 'H2026_Mar',
    },
    {
        'futures_symbol': 'BINANCE:BTCUSDM2026',
        'expiry_date': datetime(2026, 6, 26, 8, 0, 0),
        'label': 'M2026_Jun',
    },
]


def send_telegram_notification(img_path: str, intraday_img_path: str | None,
                                stats: Dict[str, float], label: str):
    """Send telegram notification with chart image and statistics."""
    try:
        # Create comprehensive message with statistics
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        msg = f"""🚀 Binance Cash and Carry Arbitrage Alert – {label}
        
📊 Analysis Summary ({current_time}):
- Current Basis: {stats['current_basis_pct']:.3f}%
- Average Basis: {stats['avg_basis_pct']:.3f}%
- Current APY: {stats['current_apy']:.1f}%
- Average APY: {stats['avg_apy']:.1f}%
- Days to Expiry: {stats['days_remaining']}

💰 Futures vs Spot Analysis
Chart attached below 📈"""
        
        # Send message first
        sendmessage(msg)
        
        # Send chart images
        if img_path and os.path.exists(img_path):
            sendphoto(img_path)
            print("✅ Main chart sent!")
        else:
            print("❌ Image file not found for telegram notification")

        if intraday_img_path and os.path.exists(intraday_img_path):
            sendphoto(intraday_img_path)
            print("✅ Intraday chart sent!")
            
    except Exception as e:
        print(f"❌ Error sending telegram notification: {e}")


def run_complete_analysis():
    """Run complete analysis for all contracts and send telegram notifications."""
    try:
        print("🔄 Starting Bitcoin Futures Analysis...")

        for config in CONTRACT_CONFIGS:
            label = config['label']
            try:
                print(f"\n{'='*60}")
                print(f"Analyzing {label} ({config['futures_symbol']})")
                print(f"{'='*60}")

                # Initialize analyzer
                analyzer = BitcoinFuturesAnalyzer(
                    futures_symbol=config['futures_symbol'],
                    expiry_date=config['expiry_date'],
                    label=label,
                    lookback=762
                )

                # Load and process data
                print("Loading and processing data...")
                data = analyzer.load_and_process_data()
                print(f"Processed {len(data)} data points")

                # Generate main visualization
                print("Creating visualization...")
                fig = analyzer.create_visualization(data)

                # Generate summary statistics
                stats = analyzer.generate_summary_stats(data)
                print("\nSummary Statistics:")
                print(f"Average Basis: {stats['avg_basis_pct']:.3f}%")
                print(f"Current Basis: {stats['current_basis_pct']:.3f}%")
                print(f"Average APY: {stats['avg_apy']:.1f}%")
                print(f"Current APY: {stats['current_apy']:.1f}%")
                print(f"Days to Expiry: {stats['days_remaining']}")

                # Save main chart
                img_path = analyzer.save_chart_as_image(fig, f'Binance_{label}_chart.png')
                plt.close(fig)

                # Generate intraday visualization (last 5 trading days)
                print("Creating intraday visualization...")
                intraday_fig = analyzer.create_intraday_visualization(data, n_trading_days=5)
                intraday_img_path = None
                if intraday_fig is not None:
                    intraday_img_path = analyzer.save_chart_as_image(
                        intraday_fig, f'Binance_{label}_intraday_chart.png'
                    )
                    plt.close(intraday_fig)

                # Send telegram notification
                print("📱 Sending Telegram notification...")
                send_telegram_notification(img_path, intraday_img_path, stats, label)

                print(f"✅ {label} analysis complete!")

            except Exception as e:
                error_msg = f"❌ {label} analysis failed: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                try:
                    sendmessage(f"🚨 Bitcoin Futures Analysis Error – {label}\n\n{error_msg}")
                except:
                    print("Failed to send error notification to Telegram")

        print(f"\n{'='*60}")
        print("✅ All analysis complete!")
        print(f"{'='*60}")
        
    except Exception as e:
        error_msg = f"❌ Analysis failed: {str(e)}"
        print(error_msg)
        
        # Send error notification to telegram
        try:
            sendmessage(f"🚨 Bitcoin Futures Analysis Error\n\n{error_msg}")
        except:
            print("Failed to send error notification to Telegram")
        
        raise


if __name__ == "__main__":
    run_complete_analysis()
