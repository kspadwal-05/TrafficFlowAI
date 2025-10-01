"""
TrafficFlow AI - Data Quality Validation
Advanced data quality checks and validation pipeline
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta
import json
import os

logger = logging.getLogger(__name__)

class DataQualityValidator:
    """Advanced data quality validation for traffic data"""
    
    def __init__(self):
        self.quality_rules = self._load_quality_rules()
        self.validation_results = {}
    
    def _load_quality_rules(self) -> Dict:
        """Load data quality rules and thresholds"""
        return {
            "volume": {
                "min_value": 0,
                "max_value": 50000,
                "required": True,
                "null_allowed": False
            },
            "average_speed": {
                "min_value": 0,
                "max_value": 200,
                "required": True,
                "null_allowed": False
            },
            "posted_speed_limit": {
                "min_value": 10,
                "max_value": 120,
                "required": True,
                "null_allowed": False
            },
            "percentile_speed_85": {
                "min_value": 0,
                "max_value": 200,
                "required": False,
                "null_allowed": True
            },
            "street_name": {
                "min_length": 2,
                "max_length": 100,
                "required": True,
                "null_allowed": False
            },
            "record_id": {
                "pattern": r"^[A-Z]-\d{3}$",  # Format: A-001, B-002, etc.
                "required": True,
                "null_allowed": False
            }
        }
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality validation"""
        logger.info(f"Starting data quality validation for {len(df)} records")
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "total_records": len(df),
            "quality_score": 0.0,
            "issues": [],
            "recommendations": [],
            "column_analysis": {},
            "data_completeness": {},
            "data_consistency": {},
            "anomalies": []
        }
        
        # 1. Data Completeness Analysis
        completeness_results = self._analyze_completeness(df)
        validation_results["data_completeness"] = completeness_results
        
        # 2. Data Consistency Analysis
        consistency_results = self._analyze_consistency(df)
        validation_results["data_consistency"] = consistency_results
        
        # 3. Column-specific validation
        column_results = self._validate_columns(df)
        validation_results["column_analysis"] = column_results
        
        # 4. Statistical anomaly detection
        anomalies = self._detect_statistical_anomalies(df)
        validation_results["anomalies"] = anomalies
        
        # 5. Business rule validation
        business_issues = self._validate_business_rules(df)
        validation_results["issues"].extend(business_issues)
        
        # 6. Calculate overall quality score
        quality_score = self._calculate_quality_score(validation_results)
        validation_results["quality_score"] = quality_score
        
        # 7. Generate recommendations
        recommendations = self._generate_recommendations(validation_results)
        validation_results["recommendations"] = recommendations
        
        logger.info(f"Data quality validation completed. Quality score: {quality_score:.3f}")
        return validation_results
    
    def _analyze_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data completeness"""
        completeness = {}
        
        for column in df.columns:
            total_values = len(df)
            non_null_values = df[column].notna().sum()
            null_count = total_values - non_null_values
            completeness_pct = (non_null_values / total_values) * 100
            
            completeness[column] = {
                "total_values": total_values,
                "non_null_values": non_null_values,
                "null_count": null_count,
                "completeness_percentage": completeness_pct,
                "is_complete": completeness_pct >= 95.0
            }
        
        return completeness
    
    def _analyze_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data consistency"""
        consistency = {}
        
        # Check for duplicate records
        duplicate_count = df.duplicated().sum()
        consistency["duplicates"] = {
            "count": duplicate_count,
            "percentage": (duplicate_count / len(df)) * 100,
            "is_acceptable": duplicate_count / len(df) < 0.05
        }
        
        # Check for data type consistency
        type_consistency = {}
        for column in df.columns:
            if df[column].dtype == 'object':
                # Check if all values can be converted to numeric
                try:
                    pd.to_numeric(df[column], errors='raise')
                    type_consistency[column] = "numeric_string"
                except:
                    type_consistency[column] = "text"
            else:
                type_consistency[column] = str(df[column].dtype)
        
        consistency["data_types"] = type_consistency
        
        return consistency
    
    def _validate_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate individual columns against quality rules"""
        column_results = {}
        
        for column, rules in self.quality_rules.items():
            if column not in df.columns:
                continue
            
            column_data = df[column]
            issues = []
            
            # Check for required fields
            if rules.get("required", False):
                null_count = column_data.isna().sum()
                if null_count > 0:
                    issues.append(f"Missing {null_count} required values")
            
            # Check value ranges
            if "min_value" in rules:
                below_min = (column_data < rules["min_value"]).sum()
                if below_min > 0:
                    issues.append(f"{below_min} values below minimum ({rules['min_value']})")
            
            if "max_value" in rules:
                above_max = (column_data > rules["max_value"]).sum()
                if above_max > 0:
                    issues.append(f"{above_max} values above maximum ({rules['max_value']})")
            
            # Check string length
            if "min_length" in rules and column_data.dtype == 'object':
                short_strings = column_data.str.len() < rules["min_length"]
                short_count = short_strings.sum()
                if short_count > 0:
                    issues.append(f"{short_count} strings below minimum length")
            
            # Check pattern matching
            if "pattern" in rules and column_data.dtype == 'object':
                import re
                pattern_matches = column_data.str.match(rules["pattern"], na=False)
                invalid_patterns = (~pattern_matches).sum()
                if invalid_patterns > 0:
                    issues.append(f"{invalid_patterns} values don't match required pattern")
            
            column_results[column] = {
                "issues": issues,
                "issue_count": len(issues),
                "is_valid": len(issues) == 0
            }
        
        return column_results
    
    def _detect_statistical_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect statistical anomalies in the data"""
        anomalies = []
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in df.columns and df[column].notna().sum() > 0:
                data = df[column].dropna()
                
                # Z-score anomaly detection
                z_scores = np.abs((data - data.mean()) / data.std())
                outliers = z_scores > 3
                
                if outliers.sum() > 0:
                    anomalies.append({
                        "column": column,
                        "type": "statistical_outlier",
                        "count": outliers.sum(),
                        "percentage": (outliers.sum() / len(data)) * 100,
                        "severity": "high" if outliers.sum() > len(data) * 0.05 else "medium"
                    })
                
                # Check for unusual distributions
                if data.std() == 0:
                    anomalies.append({
                        "column": column,
                        "type": "no_variation",
                        "count": len(data),
                        "percentage": 100.0,
                        "severity": "high"
                    })
        
        return anomalies
    
    def _validate_business_rules(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate business-specific rules"""
        issues = []
        
        # Rule 1: Average speed should not exceed posted speed limit by more than 20%
        if "averageSpeed" in df.columns and "postedSpeedLimit" in df.columns:
            speed_violations = df[
                (df["averageSpeed"] > df["postedSpeedLimit"] * 1.2) &
                (df["averageSpeed"].notna()) &
                (df["postedSpeedLimit"].notna())
            ]
            
            if len(speed_violations) > 0:
                issues.append({
                    "rule": "speed_limit_violation",
                    "description": "Average speed exceeds posted limit by >20%",
                    "count": len(speed_violations),
                    "severity": "medium"
                })
        
        # Rule 2: Volume should be reasonable for road type
        if "volume" in df.columns and "roadType" in df.columns:
            # Define expected volume ranges by road type
            volume_ranges = {
                "Residential": (0, 2000),
                "Collector": (500, 5000),
                "Arterial": (1000, 10000),
                "Highway": (2000, 20000)
            }
            
            for road_type, (min_vol, max_vol) in volume_ranges.items():
                road_data = df[df["roadType"] == road_type]
                if len(road_data) > 0:
                    volume_violations = road_data[
                        (road_data["volume"] < min_vol) | (road_data["volume"] > max_vol)
                    ]
                    
                    if len(volume_violations) > 0:
                        issues.append({
                            "rule": "volume_range_violation",
                            "description": f"Volume outside expected range for {road_type}",
                            "count": len(volume_violations),
                            "severity": "low"
                        })
        
        # Rule 3: 85th percentile should be higher than average speed
        if "percentileSpeed85" in df.columns and "averageSpeed" in df.columns:
            percentile_violations = df[
                (df["percentileSpeed85"] < df["averageSpeed"]) &
                (df["percentileSpeed85"].notna()) &
                (df["averageSpeed"].notna())
            ]
            
            if len(percentile_violations) > 0:
                issues.append({
                    "rule": "percentile_speed_violation",
                    "description": "85th percentile speed lower than average speed",
                    "count": len(percentile_violations),
                    "severity": "high"
                })
        
        return issues
    
    def _calculate_quality_score(self, validation_results: Dict) -> float:
        """Calculate overall data quality score"""
        score = 1.0
        
        # Deduct for completeness issues
        completeness = validation_results["data_completeness"]
        for column, metrics in completeness.items():
            if not metrics["is_complete"]:
                score -= (100 - metrics["completeness_percentage"]) / 1000
        
        # Deduct for consistency issues
        consistency = validation_results["data_consistency"]
        if not consistency["duplicates"]["is_acceptable"]:
            score -= consistency["duplicates"]["percentage"] / 100
        
        # Deduct for column validation issues
        column_analysis = validation_results["column_analysis"]
        total_issues = sum(metrics["issue_count"] for metrics in column_analysis.values())
        score -= min(0.2, total_issues / len(validation_results["total_records"]))
        
        # Deduct for anomalies
        anomalies = validation_results["anomalies"]
        for anomaly in anomalies:
            if anomaly["severity"] == "high":
                score -= 0.1
            elif anomaly["severity"] == "medium":
                score -= 0.05
        
        # Deduct for business rule violations
        for issue in validation_results["issues"]:
            if issue["severity"] == "high":
                score -= 0.1
            elif issue["severity"] == "medium":
                score -= 0.05
            else:
                score -= 0.02
        
        return max(0.0, min(1.0, score))
    
    def _generate_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate data quality improvement recommendations"""
        recommendations = []
        
        # Completeness recommendations
        completeness = validation_results["data_completeness"]
        incomplete_columns = [col for col, metrics in completeness.items() 
                            if not metrics["is_complete"]]
        
        if incomplete_columns:
            recommendations.append(
                f"Improve data completeness for columns: {', '.join(incomplete_columns)}"
            )
        
        # Consistency recommendations
        consistency = validation_results["data_consistency"]
        if not consistency["duplicates"]["is_acceptable"]:
            recommendations.append(
                f"Remove {consistency['duplicates']['count']} duplicate records"
            )
        
        # Anomaly recommendations
        anomalies = validation_results["anomalies"]
        high_severity_anomalies = [a for a in anomalies if a["severity"] == "high"]
        
        if high_severity_anomalies:
            recommendations.append(
                f"Investigate {len(high_severity_anomalies)} high-severity data anomalies"
            )
        
        # Business rule recommendations
        high_severity_issues = [i for i in validation_results["issues"] 
                              if i["severity"] == "high"]
        
        if high_severity_issues:
            recommendations.append(
                f"Address {len(high_severity_issues)} high-severity business rule violations"
            )
        
        return recommendations
    
    def save_validation_report(self, results: Dict, filepath: str):
        """Save validation report to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {filepath}")

def main():
    """Main data quality validation function"""
    import sys
    import os
    
    # Add src to path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    from etl_main_enhanced import EnhancedAccessExtractor, EnhancedConfig
    
    # Initialize validator
    validator = DataQualityValidator()
    
    # Load configuration
    cfg = EnhancedConfig.from_env()
    
    # Extract data
    extractor = EnhancedAccessExtractor(cfg)
    df = extractor.extract()
    
    # Validate data
    results = validator.validate_dataset(df)
    
    # Save report
    report_path = "/app/data/processed/quality_report.json"
    validator.save_validation_report(results, report_path)
    
    # Print summary
    print(f"Data Quality Validation Complete")
    print(f"Quality Score: {results['quality_score']:.3f}")
    print(f"Total Issues: {len(results['issues'])}")
    print(f"Anomalies: {len(results['anomalies'])}")
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    main()
