"""Analysis tools for the conservation agent.

Each tool provides specific conservation analysis capabilities and
can be called by the agent based on user query intent.
"""

import json
from typing import Any, Dict, List

from data_loader import load_species_data, fuzzy_match_species


class ConservationTools:
    """Collection of conservation analysis tools."""

    @staticmethod
    def retrieve_species_info(species_name: str) -> Dict[str, Any]:
        """
        Retrieve detailed information about a specific endangered species.

        Args:
            species_name: Common name of the species (e.g., "Javan Rhino", "panda").

        Returns:
            Dictionary containing species information including:
            - scientific_name: Scientific name
            - population: Current population count
            - status: Conservation status (e.g., "Critically Endangered")
            - location: Geographic range
            - threats: List of primary threats
            - migration: Migration pattern description
            - funding_received: Current funding amount
            - funding_needed: Required funding for recovery
            - population_trend: Population trajectory
            - recovery_probability: Estimated success probability
        """
        matched_key = fuzzy_match_species(species_name)
        data = load_species_data()

        if matched_key and matched_key in data:
            species_data = data[matched_key].copy()
            species_data["matched_name"] = matched_key
            return species_data

        # Return suggestions if no exact match
        suggestions = [k for k in data.keys() if species_name.lower() in k or k in species_name.lower()]
        return {
            "error": f"Species '{species_name}' not found in database.",
            "suggestions": suggestions[:3] if suggestions else list(data.keys())[:3]
        }

    @staticmethod
    def analyze_threats(target: str, threat_type: str = "") -> Dict[str, Any]:
        """
        Analyze primary threats affecting a species or region.

        Args:
            target: Species name or region to analyze.
            threat_type: Optional specific threat category to focus on.

        Returns:
            Dictionary containing:
            - target: The analyzed species/region
            - threats: List of identified threats
            - severity: Threat severity level (High/Medium/Low)
            - mitigation: Recommended mitigation strategies
        """
        matched_key = fuzzy_match_species(target)
        data = load_species_data()

        if matched_key and matched_key in data:
            species_data = data[matched_key]
            threats = species_data.get("threats", [])

            # Calculate severity based on population and status
            status = species_data.get("status", "")
            population = species_data.get("population", 0)

            if status == "Critically Endangered" or population < 100:
                severity = "Critical"
            elif status == "Endangered" or population < 1000:
                severity = "High"
            elif status == "Vulnerable":
                severity = "Medium"
            else:
                severity = "Low"

            # Generate mitigation recommendations
            mitigation = []
            threat_list = [t.lower() for t in threats]
            if "poaching" in " ".join(threat_list):
                mitigation.append("Increase anti-poaching patrols with drone surveillance")
                mitigation.append("Strengthen law enforcement and penalties")
            if "habitat loss" in " ".join(threat_list) or "habitat fragmentation" in " ".join(threat_list):
                mitigation.append("Establish protected habitat corridors")
                mitigation.append("Work with local communities on sustainable land use")
            if "climate change" in " ".join(threat_list):
                mitigation.append("Implement climate resilience programs")
                mitigation.append("Protect climate refugia areas")

            return {
                "target": target,
                "matched_species": matched_key,
                "threats": threats,
                "severity": severity,
                "threat_count": len(threats),
                "mitigation_recommendations": mitigation
            }

        # General threat analysis by threat type
        threat_data = {
            "habitat loss": {"severity": "High", "affected": "85%", "regions": ["Southeast Asia", "South America"]},
            "poaching": {"severity": "Critical", "affected": "45%", "regions": ["Africa", "Asia"]},
            "climate change": {"severity": "High", "affected": "35%", "regions": ["Global", "Arctic", "Marine"]},
            "pollution": {"severity": "Medium", "affected": "25%", "regions": ["Marine", "Urban areas"]},
            "invasive species": {"severity": "Medium", "affected": "20%", "regions": ["Islands", "Freshwater"]},
        }

        if threat_type.lower() in threat_data:
            return {
                "target": target,
                "threat_category": threat_type,
                **threat_data[threat_type.lower()]
            }

        return {
            "target": target,
            "threats": ["Data not available - species not in database"],
            "severity": "Unknown"
        }

    @staticmethod
    def analyze_funding(category: str) -> Dict[str, Any]:
        """
        Analyze conservation funding patterns and identify gaps.

        Args:
            category: Taxonomic group, region, or specific species.

        Returns:
            Dictionary containing:
            - category: The analyzed category
            - allocation: Percentage of global funding
            - funding_received: Actual funding amount
            - funding_needed: Required funding
            - funding_gap: Shortfall amount
            - status: Assessment of funding adequacy
        """
        category_lower = category.lower()
        data = load_species_data()

        # Taxonomic group analysis
        funding_allocation = {
            "mammals": {"allocation": "60%", "status": "Well funded relative to others"},
            "mammal": {"allocation": "60%", "status": "Well funded relative to others"},
            "birds": {"allocation": "25%", "status": "Moderately funded"},
            "bird": {"allocation": "25%", "status": "Moderately funded"},
            "reptiles": {"allocation": "8%", "status": "Underfunded"},
            "reptile": {"allocation": "8%", "status": "Underfunded"},
            "amphibians": {"allocation": "2%", "status": "Severely underfunded"},
            "amphibian": {"allocation": "2%", "status": "Severely underfunded"},
            "fish": {"allocation": "3%", "status": "Underfunded"},
            "invertebrates": {"allocation": "2%", "status": "Severely underfunded"},
            "invertebrate": {"allocation": "2%", "status": "Severely underfunded"},
        }

        if category_lower in funding_allocation:
            return {
                "category": category,
                **funding_allocation[category_lower],
                "global_gap": "$80 billion annually"
            }

        # Specific species funding analysis
        matched_key = fuzzy_match_species(category)
        if matched_key and matched_key in data:
            species_data = data[matched_key]
            received = species_data.get("funding_received", 0)
            needed = species_data.get("funding_needed", 0)
            gap = needed - received
            percentage = (received / needed * 100) if needed > 0 else 0

            status = "Well funded" if percentage >= 80 else "Adequately funded" if percentage >= 50 else "Underfunded"

            return {
                "category": category,
                "matched_species": matched_key,
                "funding_received": received,
                "funding_needed": needed,
                "funding_gap": gap,
                "funding_percentage": f"{percentage:.1f}%",
                "status": status
            }

        # Default response
        return {
            "category": category,
            "allocation": "Unknown category",
            "global_gap": "$80 billion annually",
            "insight": "Funding is unevenly distributed - mammals receive 60% while amphibians receive 2%"
        }

    @staticmethod
    def get_migration_data(species_name: str) -> Dict[str, Any]:
        """
        Retrieve migration pattern information for a species.

        Args:
            species_name: Name of the species.

        Returns:
            Dictionary containing:
            - species: The queried species
            - migration_pattern: Description of migration behavior
            - seasonal_info: Seasonal movement patterns
            - conservation_challenges: Threats during migration
        """
        matched_key = fuzzy_match_species(species_name)
        data = load_species_data()

        if matched_key and matched_key in data:
            species_data = data[matched_key]
            migration = species_data.get("migration", "Unknown")
            location = species_data.get("location", "Unknown")

            # Determine conservation challenges based on migration type
            challenges = []
            if "long-distance" in migration.lower():
                challenges.append("Cross-border coordination required")
                challenges.append("Habitat fragmentation blocking routes")
            if "seasonal" in migration.lower():
                challenges.append("Climate change affecting timing")
            if "non-migratory" in migration.lower():
                challenges.append("Limited genetic exchange between populations")

            return {
                "species": species_name,
                "matched_species": matched_key,
                "migration_pattern": migration,
                "location": location,
                "conservation_challenges": challenges
            }

        return {
            "species": species_name,
            "error": "Migration data not found for this species"
        }

    @staticmethod
    def compare_conservation_metrics(entities: List[str], metric: str) -> Dict[str, Any]:
        """
        Compare conservation metrics across multiple species or regions.

        Args:
            entities: List of species names or regions to compare.
            metric: The metric to compare (population, funding, threats, recovery_probability).

        Returns:
            Dictionary containing:
            - entities: List of compared entities
            - metric: The compared metric
            - comparison: Dictionary of entity -> value
            - ranking: Sorted list from best to worst
            - analysis: Brief interpretation of results
        """
        data = load_species_data()
        metric = metric.lower()
        results = {}
        matched_entities = []

        for entity in entities:
            matched_key = fuzzy_match_species(entity)
            if matched_key and matched_key in data:
                species_data = data[matched_key]
                matched_entities.append(matched_key)

                if metric == "population":
                    results[matched_key] = species_data.get("population", 0)
                elif metric == "funding":
                    results[matched_key] = species_data.get("funding_received", 0)
                elif metric in ["threat", "threats"]:
                    results[matched_key] = len(species_data.get("threats", []))
                elif metric in ["recovery", "recovery_probability", "recovery-probability"]:
                    results[matched_key] = species_data.get("recovery_probability", 0)
                elif metric == "status":
                    results[matched_key] = species_data.get("status", "Unknown")
                else:
                    results[matched_key] = f"Unknown metric: {metric}"

        # Sort by value (for numeric metrics)
        if results and all(isinstance(v, (int, float)) for v in results.values()):
            ranking = sorted(results.items(), key=lambda x: x[1], reverse=metric != "threats")
            best = ranking[0]
            worst = ranking[-1]
            analysis = f"{best[0].title()} has the highest {metric} ({best[1]:,}), while {worst[0].title()} has the lowest ({worst[1]:,})."
        else:
            ranking = list(results.items())
            analysis = "Comparison data displayed below."

        return {
            "entities": entities,
            "matched_entities": matched_entities,
            "metric": metric,
            "comparison": results,
            "ranking": ranking,
            "analysis": analysis
        }

    @staticmethod
    def predict_recovery_success(species_name: str, intervention: str) -> Dict[str, Any]:
        """
        Predict recovery success probability based on intervention type.

        Args:
            species_name: Name of the species.
            intervention: Type of intervention (e.g., "habitat_restoration", "anti_poaching").

        Returns:
            Dictionary containing:
            - species: The queried species
            - intervention: The proposed intervention
            - baseline_probability: Current recovery probability
            - projected_probability: Predicted probability with intervention
            - improvement: Percentage improvement
            - timeline: Estimated years to recovery
            - key_factors: Factors affecting success
        """
        matched_key = fuzzy_match_species(species_name)
        data = load_species_data()

        # Intervention effectiveness multipliers
        intervention_effectiveness = {
            "legal protection": 1.35,
            "habitat restoration": 1.30,
            "captive breeding": 1.25,
            "anti poaching": 1.20,
            "anti-poaching": 1.20,
            "community conservation": 1.15,
            "climate adaptation": 1.10,
            "funding increase": 1.15,
        }

        if matched_key and matched_key in data:
            species_data = data[matched_key]
            baseline = species_data.get("recovery_probability", 0.1)

            intervention_lower = intervention.lower()
            multiplier = 1.0

            for key, value in intervention_effectiveness.items():
                if key in intervention_lower:
                    multiplier = value
                    break

            projected = min(0.95, baseline * multiplier)  # Cap at 95%
            improvement = ((projected - baseline) / baseline * 100) if baseline > 0 else 0

            # Estimate timeline based on population and reproductive rate
            population = species_data.get("population", 0)
            if population < 100:
                timeline = "50-100 years"
            elif population < 1000:
                timeline = "25-50 years"
            elif population < 10000:
                timeline = "10-25 years"
            else:
                timeline = "5-15 years"

            # Key success factors
            key_factors = [
                "Adequate and sustained funding",
                "Political stability in range countries",
                "Community support and engagement",
                "Effective law enforcement",
            ]

            if baseline < 0.1:
                key_factors.append("Genetic rescue may be needed")

            return {
                "species": species_name,
                "matched_species": matched_key,
                "intervention": intervention,
                "baseline_probability": f"{baseline:.1%}",
                "projected_probability": f"{projected:.1%}",
                "improvement": f"{improvement:.0f}%",
                "timeline": timeline,
                "key_factors": key_factors
            }

        return {
            "species": species_name,
            "error": "Species not found in database"
        }


# Tool registry for dynamic execution
TOOL_REGISTRY = {
    "retrieve_species_info": ConservationTools.retrieve_species_info,
    "analyze_threats": ConservationTools.analyze_threats,
    "analyze_funding": ConservationTools.analyze_funding,
    "get_migration_data": ConservationTools.get_migration_data,
    "compare_conservation_metrics": ConservationTools.compare_conservation_metrics,
    "predict_recovery_success": ConservationTools.predict_recovery_success,
}
