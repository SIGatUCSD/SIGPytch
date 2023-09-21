from dataclasses import dataclass
from datetime import datetime

@dataclass
class P:
    """
    Class contains named constants for different period lengths 
    """
    TDAYS_PER_MONTH = 21
    TDAYS_PER_YEAR = 252
    
@dataclass
class DATE:
    """
    Class containing datetime information for calculating metrics
    """
    YEAR = datetime.now().year
    START = f"{YEAR}-01-01"
    
@dataclass
class INTERVALS:
    """
    Class containing common periods
    """
    WEEKLY = 5
    MONTHLY = 21
    YEARLY = 252