
"""
Trading Calendar Utilities for Brazilian Stock Market (B3)
=========================================================

Advanced trading calendar utilities for filtering market hours,
handling Brazilian holidays, and creating continuous time series
for professional trading visualizations.

Features:
- B3 market hours filtering (10h-17h, weekdays only)
- Brazilian holiday handling using pandas_market_calendars
- Sequential index creation for continuous visualization
- Market session boundaries and annotations
- Performance optimized with caching

Author: Enhanced Quantitative Trading System
Date: August 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
import functools

# Try to import market calendars, fallback to manual implementation
try:
    import pandas_market_calendars as mcal
    MARKET_CALENDARS_AVAILABLE = True
except ImportError:
    MARKET_CALENDARS_AVAILABLE = False
    logging.warning("pandas_market_calendars not available, using fallback holiday handling")

# Brazilian holidays fallback
try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False
    logging.warning("holidays library not available, basic holiday handling only")

logger = logging.getLogger(__name__)

class BrazilianTradingCalendar:
    """
    Brazilian Trading Calendar utilities for B3 stock exchange.
    Handles market hours, holidays, and continuous time series creation.
    """
    
    def __init__(self):
        self.market_open = time(10, 0)  # 10:00 AM BRT
        self.market_close = time(16, 55)  # 4:55 PM BRT (main session)
        self.timezone = 'America/Sao_Paulo'  # Brazilian timezone
        
        # Initialize market calendar
        self._init_market_calendar()
        
        # Cache for performance
        self._cache = {}
    
    def _init_market_calendar(self):
        """Initialize the market calendar with B3 rules"""
        try:
            if MARKET_CALENDARS_AVAILABLE:
                # Use pandas_market_calendars for B3 (BMF/BVMF)
                self.calendar = mcal.get_calendar('BVMF')  # B3 calendar
                logger.info("Successfully loaded B3 calendar from pandas_market_calendars")
            else:
                self.calendar = None
                logger.warning("Using fallback calendar implementation")
        except Exception as e:
            logger.warning(f"Error loading market calendar: {e}, using fallback")
            self.calendar = None
    
    def get_brazilian_holidays(self, start_date: Union[str, pd.Timestamp], 
                              end_date: Union[str, pd.Timestamp]) -> List[pd.Timestamp]:
        """Get Brazilian holidays for the specified date range"""
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        cache_key = f"holidays_{start_date.date()}_{end_date.date()}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        holidays_list = []
        
        if HOLIDAYS_AVAILABLE:
            # Use holidays library for Brazilian holidays
            br_holidays = holidays.Brazil(years=range(start_date.year, end_date.year + 1))
            holidays_list = [pd.to_datetime(date) for date in br_holidays.keys() 
                           if start_date <= pd.to_datetime(date) <= end_date]
        else:
            # Fallback: common Brazilian holidays (manual list)
            for year in range(start_date.year, end_date.year + 1):
                holidays_list.extend([
                    pd.Timestamp(f'{year}-01-01'),  # New Year
                    pd.Timestamp(f'{year}-04-21'),  # Tiradentes
                    pd.Timestamp(f'{year}-05-01'),  # Labor Day
                    pd.Timestamp(f'{year}-09-07'),  # Independence Day
                    pd.Timestamp(f'{year}-10-12'),  # Our Lady of Aparecida
                    pd.Timestamp(f'{year}-11-02'),  # All Souls Day
                    pd.Timestamp(f'{year}-11-15'),  # Proclamation of the Republic
                    pd.Timestamp(f'{year}-12-25'),  # Christmas
                ])
        
        # Filter to date range
        holidays_list = [h for h in holidays_list if start_date <= h <= end_date]
        self._cache[cache_key] = holidays_list
        
        return holidays_list
    
    def is_trading_day(self, date: Union[str, pd.Timestamp]) -> bool:
        """Check if a given date is a trading day"""
        date = pd.to_datetime(date)
        
        # Check if weekend
        if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if holiday
        holidays = self.get_brazilian_holidays(date - timedelta(days=1), date + timedelta(days=1))
        if any(h.date() == date.date() for h in holidays):
            return False
        
        return True
    
    def filter_market_hours(self, df: pd.DataFrame, datetime_col: Optional[str] = None) -> pd.DataFrame:
        """
        Filter DataFrame to include only market hours (10h-17h, weekdays, no holidays)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with datetime index or datetime column
        datetime_col : str, optional
            Name of datetime column if not using index
            
        Returns:
        --------
        pd.DataFrame
            Filtered DataFrame with only market hours
        """
        if df.empty:
            return df
        
        # Work with copy
        filtered_df = df.copy()
        
        # Get datetime series
        if datetime_col:
            dt_series = pd.to_datetime(filtered_df[datetime_col])
        else:
            # Convert DatetimeIndex to Series so we can use .dt accessor
            dt_series = filtered_df.index.to_series()
        
        # Create mask for market hours
        mask = pd.Series(True, index=filtered_df.index)
        
        # Filter by time (10:00 to 16:55)
        time_mask = (dt_series.dt.time >= self.market_open) & (dt_series.dt.time <= self.market_close)
        mask &= time_mask
        
        # Filter by weekday (Monday=0 to Friday=4)
        weekday_mask = dt_series.dt.weekday < 5
        mask &= weekday_mask
        
        # Filter by holidays
        if len(dt_series) > 0:
            start_date = dt_series.min()
            end_date = dt_series.max()
            holidays = self.get_brazilian_holidays(start_date, end_date)
            
            if holidays:
                holiday_dates = [h.date() for h in holidays]
                holiday_mask = ~dt_series.dt.date.isin(holiday_dates)
                mask &= holiday_mask
        
        # Apply filter
        filtered_df = filtered_df[mask]
        
        logger.info(f"Filtered data: {len(df)} -> {len(filtered_df)} rows "
                   f"({len(filtered_df)/len(df)*100:.1f}% remaining)")
        
        return filtered_df
    
    def add_sequential_index(self, df: pd.DataFrame, datetime_col: Optional[str] = None) -> pd.DataFrame:
        """
        Add sequential index column for continuous visualization without gaps
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with datetime index or column
        datetime_col : str, optional
            Name of datetime column if not using index
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added 'seq_index' column
        """
        result_df = df.copy()
        
        # Add sequential index (0, 1, 2, ...)
        result_df['seq_index'] = range(len(result_df))
        
        # Store original datetime for tooltips
        if datetime_col:
            result_df['original_datetime'] = pd.to_datetime(result_df[datetime_col])
        else:
            result_df['original_datetime'] = pd.to_datetime(result_df.index)
        
        return result_df
    
    def get_session_boundaries(self, df: pd.DataFrame, datetime_col: Optional[str] = None) -> Dict[str, List]:
        """
        Get session boundaries for visual markers
        
        Returns:
        --------
        Dict with 'session_starts', 'session_ends', 'day_separators'
        """
        if df.empty:
            return {'session_starts': [], 'session_ends': [], 'day_separators': []}
        
        # Get datetime series
        if datetime_col:
            dt_series = pd.to_datetime(df[datetime_col])
        else:
            dt_series = pd.to_datetime(df.index)
        
        session_starts = []
        session_ends = []
        day_separators = []
        
        if len(dt_series) > 0:
            current_date = None
            for i, dt in enumerate(dt_series):
                if current_date is None or dt.date() != current_date:
                    # New day - mark session start and day separator
                    if i > 0:  # Not the first day
                        day_separators.append(i)
                    session_starts.append(i)
                    current_date = dt.date()
                
                # Check if this is end of session (close to market close or last of day)
                if i == len(dt_series) - 1 or dt_series.iloc[i + 1].date() != current_date:
                    session_ends.append(i)
        
        return {
            'session_starts': session_starts,
            'session_ends': session_ends,
            'day_separators': day_separators[1:] if day_separators else []  # Skip first separator
        }
    
    def get_trading_days_range(self, start_date: Union[str, pd.Timestamp], 
                              end_date: Union[str, pd.Timestamp]) -> pd.DatetimeIndex:
        """
        Get range of trading days (excluding weekends and holidays)
        """
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        if self.calendar:
            # Use market calendar if available
            try:
                schedule = self.calendar.schedule(start_date=start_date, end_date=end_date)
                return schedule.index
            except Exception as e:
                logger.warning(f"Error using market calendar: {e}, falling back to manual method")
        
        # Fallback: manual calculation
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_days = []
        
        for date in all_dates:
            if self.is_trading_day(date):
                trading_days.append(date)
        
        return pd.DatetimeIndex(trading_days)
    
    def create_market_hours_index(self, start_date: Union[str, pd.Timestamp],
                                 end_date: Union[str, pd.Timestamp],
                                 freq: str = '1H') -> pd.DatetimeIndex:
        """
        Create complete market hours index for the given period
        
        Parameters:
        -----------
        start_date : str or pd.Timestamp
            Start date
        end_date : str or pd.Timestamp  
            End date
        freq : str
            Frequency ('1H', '30T', etc.)
            
        Returns:
        --------
        pd.DatetimeIndex
            Complete index of market hours
        """
        trading_days = self.get_trading_days_range(start_date, end_date)
        
        market_hours = []
        for date in trading_days:
            # Create hourly index for this day (10:00 to 16:55)
            day_start = pd.Timestamp.combine(date.date(), self.market_open)
            day_end = pd.Timestamp.combine(date.date(), self.market_close)
            
            day_hours = pd.date_range(start=day_start, end=day_end, freq=freq)
            market_hours.extend(day_hours)
        
        return pd.DatetimeIndex(market_hours)

# Global instance for convenience
brazilian_calendar = BrazilianTradingCalendar()

# Convenience functions
def filter_market_hours(df: pd.DataFrame, datetime_col: Optional[str] = None) -> pd.DataFrame:
    """Convenience function to filter market hours"""
    return brazilian_calendar.filter_market_hours(df, datetime_col)

def add_sequential_index(df: pd.DataFrame, datetime_col: Optional[str] = None) -> pd.DataFrame:
    """Convenience function to add sequential index"""
    return brazilian_calendar.add_sequential_index(df, datetime_col)

def get_session_boundaries(df: pd.DataFrame, datetime_col: Optional[str] = None) -> Dict[str, List]:
    """Convenience function to get session boundaries"""
    return brazilian_calendar.get_session_boundaries(df, datetime_col)

def is_trading_day(date: Union[str, pd.Timestamp]) -> bool:
    """Convenience function to check if date is trading day"""
    return brazilian_calendar.is_trading_day(date)

@functools.lru_cache(maxsize=128)
def get_rangebreaks(start_date: str, end_date: str) -> List[Dict]:
    """
    Get Plotly rangebreaks for the specified date range
    Cached for performance
    
    Returns:
    --------
    List of rangebreak dictionaries for Plotly
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Get all dates in range
    all_dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
    
    # Find non-trading days
    non_trading_dates = []
    for date in all_dates:
        if not brazilian_calendar.is_trading_day(date):
            non_trading_dates.append(date.strftime('%Y-%m-%d'))
    
    # Create rangebreaks
    rangebreaks = []
    
    # Add weekend breaks (standard)
    rangebreaks.append(dict(bounds=["sat", "mon"]))
    
    # Add specific holiday breaks
    if non_trading_dates:
        rangebreaks.append(dict(values=non_trading_dates))
    
    # Add non-market hours breaks (before 10:00 and after 17:00)
    rangebreaks.extend([
        dict(bounds=[0, 10], pattern="hour"),  # Before 10:00
        dict(bounds=[17, 24], pattern="hour")  # After 17:00
    ])
    
    return rangebreaks

def install_required_packages():
    """Install required packages if not available"""
    packages_to_install = []
    
    if not MARKET_CALENDARS_AVAILABLE:
        packages_to_install.append('pandas-market-calendars')
    
    if not HOLIDAYS_AVAILABLE:
        packages_to_install.append('holidays')
    
    if packages_to_install:
        import subprocess
        import sys
        
        for package in packages_to_install:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                logger.info(f"Successfully installed {package}")
            except subprocess.CalledProcessError:
                logger.error(f"Failed to install {package}")

# Install packages on import if needed
if __name__ == "__main__":
    install_required_packages()
