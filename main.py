"""
Binance Cash and Carry Arbitrage Analysis
Analyzes BTC spot vs futures basis and APY across multiple contracts
Sends reports to Telegram
"""

import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import warnings
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
API_TOKEN = os.getenv("TELEGRAM_API_TOKEN", "7594281395:AAHZq1bz9Ym5-9bjjtMs_kcwS5CzUjznMEE")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "7311904934")

# Timeframe and lookback (same as original)
TIMEFRAME = '1'
LOOKBACK = 762

# Output directory
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# TELEGRAM FUNCTIONS
# =============================================================================
def send_message(message: str) -> bool:
    """Send a text message to Telegram"""
    if not API_TOKEN or not CHAT_ID:
        print("Warning: Telegram credentials not configured")
        return False

    api_url = f'https://api.telegram.org/bot{API_TOKEN}/sendMessage'
    try:
        response = requests.post(api_url, json={'chat_id': CHAT_ID, 'text': message})
        print(f"Message sent: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error sending message: {e}")
        return False


def send_photo(image_path: str) -> bool:
    """Send a photo to Telegram"""
    if not API_TOKEN or not CHAT_ID:
        print("Warning: Telegram credentials not configured")
        return False

    api_url = f'https://api.telegram.org/bot{API_TOKEN}/sendPhoto'
    try:
        with open(image_path, 'rb') as photo:
            response = requests.post(api_url, files={'photo': photo}, data={'chat_id': CHAT_ID})
        print(f"Photo sent: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error sending photo: {e}")
        return False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
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


# =============================================================================
# DATA LOADING
# =============================================================================
def load_asset_price(symbol: str, lookback: int, timeframe: str, extra=None) -> pd.DataFrame:
    """Load price data for a given symbol"""
    from price_loaders.tradingview import load_asset_price
    return load_asset_price(symbol, lookback, timeframe, extra)


# =============================================================================
# ANALYZER
# =============================================================================
class BitcoinFuturesAnalyzer:
    """Analyzes Bitcoin cash-and-carry arbitrage opportunities"""

    # Color scheme
    COLORS = {
        'spot': '#A9A9A9',
        'futures': '#FF69B4',
        'volume': '#FFC0CB',
        'basis': '#FF69B4',
        'apy': '#FF69B4',
    }

    def __init__(self, spot_symbol: str, futures_symbol: str,
                 expiry_date: str, label: str,
                 lookback: int = LOOKBACK):
        self.spot_symbol = spot_symbol
        self.futures_symbol = futures_symbol
        self.expiry_date = dt.datetime.fromisoformat(expiry_date)
        self.label = label
        self.lookback = lookback

        # Will be populated by load_and_calculate()
        self.df = None

    def _calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate futures basis, days to expiry, and APY"""
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

        # Futures basis (percentage)
        df['futures_basis_pct'] = ((df['futures_price'] / df['spot_price']) - 1) * 100

        # Days to expiry
        df['days_to_expiry'] = (self.expiry_date - df['timestamp']).dt.days

        # APY (avoid division by zero)
        mask = df['days_to_expiry'] > 0
        df['apy'] = np.nan
        df.loc[mask, 'apy'] = (
            df.loc[mask, 'futures_basis_pct'] * 365 / df.loc[mask, 'days_to_expiry']
        )

        return df

    def load_and_calculate(self) -> pd.DataFrame:
        """Load spot and futures data, merge and calculate metrics"""
        print(f"  Loading {TIMEFRAME}-min data (lookback={self.lookback})...")

        spot = load_asset_price(self.spot_symbol, self.lookback, TIMEFRAME, None)[['time', 'close']]
        futures = load_asset_price(self.futures_symbol, self.lookback, TIMEFRAME, None)[['time', 'close', 'volume']]

        df = spot.merge(futures, on='time', how='inner')
        df = df.rename(columns={
            'time': 'timestamp',
            'close_x': 'spot_price',
            'close_y': 'futures_price',
            'volume': 'futures_volume'
        })

        df = self._calculate_metrics(df)
        self.df = df.dropna(subset=['spot_price', 'futures_price']).reset_index(drop=True)

        print(f"  Data loaded: {len(self.df)} rows")
        return self.df

    def generate_summary_stats(self) -> Dict[str, float]:
        """Generate summary statistics"""
        if self.df is None or len(self.df) == 0:
            return {}

        clean_apy = self.df['apy'].dropna()
        clean_apy = clean_apy[np.abs(clean_apy) < 100]

        return {
            'avg_basis_pct': self.df['futures_basis_pct'].mean(),
            'current_basis_pct': self.df['futures_basis_pct'].iloc[-1],
            'avg_apy': clean_apy.mean() if len(clean_apy) > 0 else np.nan,
            'current_apy': clean_apy.iloc[-1] if len(clean_apy) > 0 else np.nan,
            'days_remaining': int(self.df['days_to_expiry'].iloc[-1]),
        }

    # =========================================================================
    # CHART DRAWING
    # =========================================================================
    def _draw_chart(self, df: pd.DataFrame, title: str, img_path: str,
                    show_day_boundaries: bool = False,
                    figsize: tuple = (14, 12)) -> str:
        """Shared chart drawing: 3 subplots (price+vol, basis, APY)"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        fig, (ax_price, ax_basis, ax_apy) = plt.subplots(
            3, 1, figsize=figsize, sharex=True
        )
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # --- Subplot 1: Spot vs Futures Prices + Volume ---
        ax_price.plot(df['timestamp'], df['spot_price'],
                      label='Spot Price', color=self.COLORS['spot'], linewidth=2)
        ax_price.plot(df['timestamp'], df['futures_price'],
                      label='Futures Price', color=self.COLORS['futures'], linewidth=2)

        ax_vol = ax_price.twinx()
        ax_vol.plot(df['timestamp'], df['futures_volume'],
                    label='Futures Volume', color=self.COLORS['volume'],
                    linewidth=0.8, linestyle='--', alpha=0.7)

        ax_price.set_ylabel('Price (USDT)', fontsize=11, fontweight='bold')
        ax_vol.set_ylabel('Volume', fontsize=11, fontweight='bold')
        ax_price.set_title('Bitcoin: Spot vs Futures Prices', fontsize=14, fontweight='bold', pad=20)
        ax_price.legend(loc='upper left', fontsize=10)
        ax_vol.legend(loc='upper right', fontsize=10)
        ax_price.grid(True, linestyle='--', alpha=0.3)

        # --- Subplot 2: Futures Basis % ---
        ax_basis.plot(df['timestamp'], df['futures_basis_pct'],
                      color=self.COLORS['basis'], linewidth=1.5)
        ax_basis.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        ax_basis.set_ylabel('Basis (%)', fontsize=11, fontweight='bold')
        ax_basis.set_title('Futures Basis (Premium/Discount)', fontsize=14, fontweight='bold', pad=20)
        ax_basis.grid(True, linestyle='--', alpha=0.3)
        self._format_basis_yticks(ax_basis)

        # --- Subplot 3: APY ---
        clean = df[np.abs(df['apy']) < 100].copy()
        ax_apy.plot(clean['timestamp'], clean['apy'],
                    color=self.COLORS['apy'], linewidth=1.5)
        ax_apy.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        ax_apy.set_ylabel('APY (%)', fontsize=11, fontweight='bold')
        ax_apy.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax_apy.set_title('Annualized Yield from Futures Basis', fontsize=14, fontweight='bold', pad=20)
        ax_apy.grid(True, linestyle='--', alpha=0.3)
        self._format_apy_yticks(ax_apy)

        # --- Day boundary lines (for intraday chart) ---
        if show_day_boundaries:
            dates = df['timestamp'].dt.date
            for i in range(1, len(df)):
                if dates.iloc[i] != dates.iloc[i - 1]:
                    for ax in [ax_price, ax_basis, ax_apy]:
                        ax.axvline(x=df['timestamp'].iloc[i], color='gray',
                                   linestyle='-', linewidth=0.8, alpha=0.5)

        # --- X-axis formatting ---
        self._format_x_axis([ax_price, ax_basis, ax_apy])

        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(top=0.95, bottom=0.15)
        fig.savefig(img_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)

        return img_path

    @staticmethod
    def _format_basis_yticks(ax: plt.Axes) -> None:
        """Smart y-axis tick spacing for basis subplot"""
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        if y_range < 0.2:
            step = 0.02
        elif y_range < 0.5:
            step = 0.05
        elif y_range < 1.0:
            step = 0.1
        elif y_range < 2.0:
            step = 0.2
        else:
            step = 0.5

        ticks = np.arange(
            np.floor(y_min / step) * step,
            np.ceil(y_max / step) * step + step,
            step
        )
        if len(ticks) > 10:
            step *= 2
            ticks = np.arange(
                np.floor(y_min / step) * step,
                np.ceil(y_max / step) * step + step,
                step
            )
        ax.set_yticks(ticks)
        ax.set_ylim(y_min, y_max)
        fmt = f'{{x:.1f}}' if step >= 0.1 else f'{{x:.3f}}'
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: fmt.format(x=x)))

    @staticmethod
    def _format_apy_yticks(ax: plt.Axes) -> None:
        """Smart y-axis tick spacing for APY subplot"""
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        if y_range < 2:
            step = 0.2
        elif y_range < 5:
            step = 0.5
        elif y_range < 10:
            step = 1.0
        else:
            step = 2.0

        ticks = np.arange(
            np.floor(y_min / step) * step,
            np.ceil(y_max / step) * step + step,
            step
        )
        ax.set_yticks(ticks)
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))

    @staticmethod
    def _format_x_axis(axes: list) -> None:
        """Format x-axis for all subplots (same as original)"""
        for ax in axes:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
            ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))

            for label in ax.get_xticklabels():
                label.set_rotation(90)
                label.set_horizontalalignment('center')
                label.set_fontsize(8)

            ax.xaxis.grid(True, which='major', alpha=0.5, linewidth=0.8)
            ax.xaxis.grid(True, which='minor', alpha=0.2, linewidth=0.4)

    # =========================================================================
    # PUBLIC CHART METHODS
    # =========================================================================
    def create_chart(self) -> str | None:
        """Create the main chart (all data)"""
        if self.df is None or len(self.df) == 0:
            print(f"  [Chart] Skipped: no data for {self.label}")
            return None

        title = f'Bitcoin: Spot vs Futures Prices ({self.label})'
        img_path = os.path.join(OUTPUT_DIR, f'BTC_{self.label}_chart.png')

        return self._draw_chart(self.df, title, img_path,
                                show_day_boundaries=False, figsize=(14, 12))

    def create_intraday_chart(self, n_trading_days: int = 5) -> str | None:
        """
        Create intraday chart for the last N trading days.
        Returns None if fewer than n_trading_days are available.
        """
        if self.df is None or len(self.df) == 0:
            print(f"  [Intraday] Skipped: no data for {self.label}")
            return None

        df_filtered = get_trading_days(self.df, n_trading_days)
        if df_filtered is None:
            print(f"  [Intraday] Skipped: fewer than {n_trading_days} trading days "
                  f"for {self.label}")
            return None

        df_filtered = df_filtered.sort_values('timestamp').reset_index(drop=True)

        title = (f'Bitcoin: Spot vs Futures – Last {n_trading_days} Days '
                 f'({self.label}, 1-min)')
        img_path = os.path.join(OUTPUT_DIR, f'BTC_{self.label}_intraday_chart.png')

        return self._draw_chart(df_filtered, title, img_path,
                                show_day_boundaries=True, figsize=(18, 12))

    # =========================================================================
    # RUN ANALYSIS
    # =========================================================================
    def run_analysis(self, send_telegram: bool = True) -> dict:
        """Run complete analysis and optionally send to Telegram"""
        print(f"\n{'='*60}")
        print(f"Analyzing {self.label} ({self.futures_symbol})")
        print(f"Expiry: {self.expiry_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}")

        # Load data
        self.load_and_calculate()

        # Summary stats
        stats = self.generate_summary_stats()
        if stats:
            print(f"  Current Basis: {stats['current_basis_pct']:.3f}%")
            print(f"  Average Basis: {stats['avg_basis_pct']:.3f}%")
            print(f"  Current APY:   {stats['current_apy']:.1f}%")
            print(f"  Average APY:   {stats['avg_apy']:.1f}%")
            print(f"  Days to Expiry: {stats['days_remaining']}")

        # Create charts
        chart_path = self.create_chart()
        if chart_path:
            print(f"  Chart saved: {chart_path}")

        intraday_path = self.create_intraday_chart(n_trading_days=5)
        if intraday_path:
            print(f"  Intraday chart saved: {intraday_path}")

        # Send to Telegram
        if send_telegram and stats:
            current_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = (
                f"🚀 BTC Cash & Carry – {self.label}\n"
                f"📊 {current_time}\n\n"
                f"• Current Basis: {stats['current_basis_pct']:.3f}%\n"
                f"• Average Basis: {stats['avg_basis_pct']:.3f}%\n"
                f"• Current APY: {stats['current_apy']:.1f}%\n"
                f"• Average APY: {stats['avg_apy']:.1f}%\n"
                f"• Days to Expiry: {stats['days_remaining']}"
            )
            send_message(msg)
            if chart_path:
                send_photo(chart_path)
            if intraday_path:
                send_photo(intraday_path)

        return {
            'label': self.label,
            'stats': stats,
            'chart_path': chart_path,
            'intraday_chart_path': intraday_path,
        }


# =============================================================================
# CONTRACT CONFIGURATIONS
# =============================================================================
CONTRACT_CONFIGS = [
    {
        'spot_symbol': 'BINANCE:BTCUSDT',
        'futures_symbol': 'BINANCE:BTCUSDH2026',
        'expiry_date': '2026-03-27T08:00:00',
        'label': 'H2026_Mar',
    },
    {
        'spot_symbol': 'BINANCE:BTCUSDT',
        'futures_symbol': 'BINANCE:BTCUSDM2026',
        'expiry_date': '2026-06-26T08:00:00',
        'label': 'M2026_Jun',
    },
]


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main execution function"""
    print("=" * 60)
    print("Binance BTC Cash & Carry Arbitrage Analysis")
    print(f"Started at: {dt.datetime.now()}")
    print("=" * 60)

    results = []

    for config in CONTRACT_CONFIGS:
        try:
            analyzer = BitcoinFuturesAnalyzer(
                spot_symbol=config['spot_symbol'],
                futures_symbol=config['futures_symbol'],
                expiry_date=config['expiry_date'],
                label=config['label'],
            )
            result = analyzer.run_analysis(send_telegram=True)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {config['label']}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print(f"Finished at: {dt.datetime.now()}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
