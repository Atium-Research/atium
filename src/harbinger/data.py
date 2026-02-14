"""
Data Adapter base class.

Users implement this to connect their data to harbinger.
"""

from abc import ABC, abstractmethod

import numpy as np


class DataAdapter(ABC):
    """
    Abstract base class for data adapters.
    
    Users implement this to provide their data to the backtester.
    The backtester calls these methods during simulation.
    """
    
    # =========================================================================
    # Required Methods
    # =========================================================================
    
    @abstractmethod
    def get_dates(self) -> list:
        """
        Return list of all available dates in chronological order.
        
        Returns:
            List of date objects
        """
        pass
    
    @abstractmethod
    def get_universe(self, date) -> list[str]:
        """
        Return list of tickers in the universe for this date.
        
        Args:
            date: The date to query
            
        Returns:
            List of ticker strings
        """
        pass
    
    @abstractmethod
    def get_alphas(self, date) -> dict[str, float]:
        """
        Return expected returns / alpha scores for this date.
        
        Args:
            date: The date to query
            
        Returns:
            Dict mapping ticker -> alpha score
        """
        pass
    
    @abstractmethod
    def get_covariance(self, date) -> tuple[np.ndarray, list[str]]:
        """
        Return covariance matrix for this date.
        
        Args:
            date: The date to query
            
        Returns:
            Tuple of (cov_matrix, tickers) where cov_matrix is n x n
            and tickers defines the ordering
        """
        pass
    
    @abstractmethod
    def get_prices(self, date) -> dict[str, float]:
        """
        Return prices for this date.
        
        Args:
            date: The date to query
            
        Returns:
            Dict mapping ticker -> price
        """
        pass
    
    @abstractmethod
    def get_returns(self, date) -> dict[str, float]:
        """
        Return realized returns for this date (for PnL calculation).
        
        These are the returns from date to the next trading date.
        
        Args:
            date: The date to query
            
        Returns:
            Dict mapping ticker -> return
        """
        pass
    
    # =========================================================================
    # Optional Methods (required by some objectives/constraints)
    # =========================================================================
    
    def get_benchmark_weights(self, date) -> dict[str, float]:
        """
        Return benchmark weights for this date.
        
        Required by: TargetActiveRisk, MinimizeActiveVariance
        
        Args:
            date: The date to query
            
        Returns:
            Dict mapping ticker -> benchmark weight
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_benchmark_weights(). "
            "This is required for TargetActiveRisk and MinimizeActiveVariance objectives."
        )
    
    def get_sector_mapping(self, date) -> dict[str, str]:
        """
        Return sector mapping for this date.
        
        Required by: MaxSectorWeight, SectorNeutral
        
        Args:
            date: The date to query
            
        Returns:
            Dict mapping ticker -> sector name
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_sector_mapping(). "
            "This is required for MaxSectorWeight and SectorNeutral constraints."
        )
    
    def get_factor_exposures(self, date) -> dict[str, dict[str, float]]:
        """
        Return factor exposures for this date.
        
        Required by: MaxFactorExposure, FactorNeutral
        
        Args:
            date: The date to query
            
        Returns:
            Dict mapping ticker -> {factor_name: exposure}
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_factor_exposures(). "
            "This is required for MaxFactorExposure and FactorNeutral constraints."
        )
