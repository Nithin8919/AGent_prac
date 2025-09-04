"""
Analytics Agent - Data Analysis and Insights
==========================================

This agent provides advanced analytics, aggregations, and insights
from the vehicle registration data.

Features:
- Statistical analysis
- Trend analysis
- Growth calculations
- Market share analysis
- Comparative analytics
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from llamaindex_connector import VehicleRegistrationConnector

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsResult:
    """Result from analytics operations."""
    metric_name: str
    value: Any
    percentage: Optional[float] = None
    trend: Optional[str] = None
    comparison: Optional[Dict[str, Any]] = None
    insights: Optional[List[str]] = None


class AnalyticsAgent:
    """Agent for advanced data analytics and insights."""
    
    def __init__(self, db_path: str = "../db/vehicles.db", openai_api_key: Optional[str] = None):
        """
        Initialize the analytics agent.
        
        Args:
            db_path: Path to SQLite database
            openai_api_key: OpenAI API key for LlamaIndex
        """
        self.connector = VehicleRegistrationConnector(db_path, openai_api_key)
        self.initialized = False
        
        logger.info("Analytics Agent initialized")
    
    def initialize(self):
        """Initialize the LlamaIndex connector."""
        try:
            self.connector.initialize()
            self.initialized = True
            logger.info("Analytics Agent connector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize analytics agent: {str(e)}")
            raise
    
    def calculate_market_share(self, entity_type: str = "manufacturer") -> List[AnalyticsResult]:
        """
        Calculate market share for manufacturers or other entities.
        
        Args:
            entity_type: Type of entity to analyze (manufacturer, fuel_type, etc.)
            
        Returns:
            List of AnalyticsResult objects with market share data
        """
        if not self.initialized:
            raise ValueError("Analytics agent not initialized")
        
        # Query for market share data
        if entity_type == "manufacturer":
            question = "What is the market share of each vehicle manufacturer by registration count?"
        elif entity_type == "fuel_type":
            question = "What is the market share of each fuel type by usage?"
        elif entity_type == "vehicle_type":
            question = "What is the market share of each vehicle type by registration count?"
        else:
            question = f"What is the distribution of {entity_type}?"
        
        result = self.connector.query(question)
        
        if not result.get('success', False):
            return []
        
        # Parse the result and create AnalyticsResult objects
        return self._parse_market_share_result(result, entity_type)
    
    def calculate_growth_rate(self, entity: str, year1: int, year2: int) -> AnalyticsResult:
        """
        Calculate growth rate between two years.
        
        Args:
            entity: Entity to analyze (e.g., "total", "honda", "petrol")
            year1: Starting year
            year2: Ending year
            
        Returns:
            AnalyticsResult with growth rate information
        """
        if not self.initialized:
            raise ValueError("Analytics agent not initialized")
        
        # Query for data in both years
        question1 = f"How many {entity} vehicles were registered in {year1}?"
        question2 = f"How many {entity} vehicles were registered in {year2}?"
        
        result1 = self.connector.query(question1)
        result2 = self.connector.query(question2)
        
        if not (result1.get('success', False) and result2.get('success', False)):
            return AnalyticsResult(
                metric_name="growth_rate",
                value=None,
                insights=["Unable to calculate growth rate due to query errors"]
            )
        
        # Extract numbers from results
        count1 = self._extract_number_from_result(result1)
        count2 = self._extract_number_from_result(result2)
        
        if count1 is None or count2 is None:
            return AnalyticsResult(
                metric_name="growth_rate",
                value=None,
                insights=["Unable to extract numeric values from results"]
            )
        
        # Calculate growth rate
        if count1 == 0:
            growth_rate = float('inf') if count2 > 0 else 0
        else:
            growth_rate = ((count2 - count1) / count1) * 100
        
        # Determine trend
        if growth_rate > 5:
            trend = "strong_growth"
        elif growth_rate > 0:
            trend = "moderate_growth"
        elif growth_rate > -5:
            trend = "stable"
        else:
            trend = "decline"
        
        insights = [
            f"Growth rate from {year1} to {year2}: {growth_rate:.1f}%",
            f"Change in registrations: {count2 - count1:,} vehicles",
            f"Trend: {trend.replace('_', ' ').title()}"
        ]
        
        return AnalyticsResult(
            metric_name="growth_rate",
            value=growth_rate,
            percentage=growth_rate,
            trend=trend,
            comparison={
                f"{year1}": count1,
                f"{year2}": count2
            },
            insights=insights
        )
    
    def analyze_trends(self, entity: str = "total", years: List[int] = None) -> AnalyticsResult:
        """
        Analyze trends over multiple years.
        
        Args:
            entity: Entity to analyze
            years: List of years to analyze (default: 2019-2025)
            
        Returns:
            AnalyticsResult with trend analysis
        """
        if years is None:
            years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
        
        if not self.initialized:
            raise ValueError("Analytics agent not initialized")
        
        yearly_data = {}
        
        # Get data for each year
        for year in years:
            question = f"How many {entity} vehicles were registered in {year}?"
            result = self.connector.query(question)
            
            if result.get('success', False):
                count = self._extract_number_from_result(result)
                if count is not None:
                    yearly_data[year] = count
        
        if not yearly_data:
            return AnalyticsResult(
                metric_name="trend_analysis",
                value=None,
                insights=["Unable to retrieve trend data"]
            )
        
        # Calculate trend insights
        insights = []
        years_list = sorted(yearly_data.keys())
        
        if len(years_list) >= 2:
            first_year = years_list[0]
            last_year = years_list[-1]
            first_count = yearly_data[first_year]
            last_count = yearly_data[last_year]
            
            total_growth = ((last_count - first_count) / first_count) * 100 if first_count > 0 else 0
            insights.append(f"Overall growth ({first_year}-{last_year}): {total_growth:.1f}%")
            
            # Find peak year
            peak_year = max(yearly_data.keys(), key=lambda y: yearly_data[y])
            peak_count = yearly_data[peak_year]
            insights.append(f"Peak year: {peak_year} with {peak_count:,} registrations")
            
            # Find year with highest growth
            if len(years_list) >= 3:
                max_growth = 0
                max_growth_year = None
                for i in range(1, len(years_list)):
                    year = years_list[i]
                    prev_year = years_list[i-1]
                    growth = ((yearly_data[year] - yearly_data[prev_year]) / yearly_data[prev_year]) * 100
                    if growth > max_growth:
                        max_growth = growth
                        max_growth_year = year
                
                if max_growth_year:
                    insights.append(f"Highest year-over-year growth: {max_growth_year} ({max_growth:.1f}%)")
        
        return AnalyticsResult(
            metric_name="trend_analysis",
            value=yearly_data,
            insights=insights
        )
    
    def compare_entities(self, entity_type: str, entities: List[str]) -> AnalyticsResult:
        """
        Compare multiple entities (e.g., manufacturers, fuel types).
        
        Args:
            entity_type: Type of entities to compare
            entities: List of entity names to compare
            
        Returns:
            AnalyticsResult with comparison data
        """
        if not self.initialized:
            raise ValueError("Analytics agent not initialized")
        
        comparison_data = {}
        
        for entity in entities:
            question = f"How many {entity} {entity_type} vehicles are registered?"
            result = self.connector.query(question)
            
            if result.get('success', False):
                count = self._extract_number_from_result(result)
                if count is not None:
                    comparison_data[entity] = count
        
        if not comparison_data:
            return AnalyticsResult(
                metric_name="entity_comparison",
                value=None,
                insights=["Unable to retrieve comparison data"]
            )
        
        # Calculate insights
        total = sum(comparison_data.values())
        insights = []
        
        for entity, count in comparison_data.items():
            percentage = (count / total) * 100 if total > 0 else 0
            insights.append(f"{entity.title()}: {count:,} ({percentage:.1f}%)")
        
        # Find leader
        leader = max(comparison_data.keys(), key=lambda e: comparison_data[e])
        leader_count = comparison_data[leader]
        leader_percentage = (leader_count / total) * 100 if total > 0 else 0
        
        insights.append(f"Market leader: {leader.title()} with {leader_percentage:.1f}% share")
        
        return AnalyticsResult(
            metric_name="entity_comparison",
            value=comparison_data,
            percentage=leader_percentage,
            insights=insights
        )
    
    def _parse_market_share_result(self, result: Dict[str, Any], entity_type: str) -> List[AnalyticsResult]:
        """Parse market share query result into AnalyticsResult objects."""
        # This is a simplified parser - in a real implementation, you'd parse the SQL result
        # For now, we'll return a basic result
        return [
            AnalyticsResult(
                metric_name="market_share",
                value=result.get('answer', ''),
                insights=["Market share data retrieved successfully"]
            )
        ]
    
    def _extract_number_from_result(self, result: Dict[str, Any]) -> Optional[int]:
        """Extract numeric value from query result."""
        answer = result.get('answer', '')
        
        # Try to find numbers in the answer
        import re
        numbers = re.findall(r'\d{1,3}(?:,\d{3})*', answer)
        
        if numbers:
            # Return the first number found, removing commas
            return int(numbers[0].replace(',', ''))
        
        return None
    
    def generate_insights_report(self) -> List[str]:
        """Generate a comprehensive insights report."""
        if not self.initialized:
            return ["Analytics agent not initialized"]
        
        insights = []
        
        try:
            # Market share insights
            market_share = self.calculate_market_share("manufacturer")
            if market_share:
                insights.append("📊 Market Share Analysis: Top manufacturers by registration count")
            
            # Growth insights
            growth = self.calculate_growth_rate("total", 2023, 2024)
            if growth.insights:
                insights.extend(growth.insights)
            
            # Trend insights
            trends = self.analyze_trends("total")
            if trends.insights:
                insights.extend(trends.insights)
            
        except Exception as e:
            insights.append(f"Error generating insights: {str(e)}")
        
        return insights
 
    def test_connection(self) -> bool:
        """Test if the analytics agent can connect to the database."""
        try:
            if not self.initialized:
                self.initialize()
            
            return True
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False


def main():
    """Test the analytics agent."""
    print("📈 Testing Analytics Agent")
    print("=" * 40)
    
    agent = AnalyticsAgent()
    
    # Test connection
    print("🔍 Testing database connection...")
    if agent.test_connection():
        print("✅ Database connection successful!")
        
        print("\n💡 To test analytics features, set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("   Then run the full analytics system")
        
    else:
        print("❌ Database connection failed!")


if __name__ == "__main__":
    main()