import time
from datetime import datetime


class ViolationManager:
    def __init__(self, class_names):
        self.violations = []
        self.class_names = class_names

    def add_violation(self, log, violation_type, location, details, image_url=None):
        """
        Add a new violation to the violations log formatted for the frontend

        Args:
            track_id (int): Track ID of the vehicle
            violation_type (str): Type of violation (speed, rlv, wrong_way)
            location (str): Location of the violation
            details (str): Details about the violation
            image_url (str, optional): URL to the violation image
        """
        # Map internal violation types to frontend-friendly names
        violation_type_map = {
            "speed": "Speeding",
            "rlv": "Red Light Violation",
            "wrong_way": "Wrong Way Driving"
        }

        track_id = log["track_id"]
        vehicle_class = self.class_names[log["class_id"]]

        # Generate a unique ID for this violation
        violation_id = f"{vehicle_class}-{track_id}-{int(time.time())}"

        # Check if this violation has already been recorded for this track
        for violation in self.violations:
            if violation["id"].startswith(f"{vehicle_class}-{track_id}"):
                # Only record one violation per track per type
                return

        # Build the violation record in the format expected by the frontend
        violation = {
            "id": violation_id,
            "plate": f"{vehicle_class}-{track_id}",
            "type": violation_type_map.get(violation_type, violation_type),
            "status": "Pending",
            "date": datetime.now().isoformat(),
            "location": location,
            "evidence": image_url
        }

        # Add type-specific details based on violation type
        if violation_type == "speed":
            violation["speed"] = details
        elif violation_type == "rlv":
            violation["signalTime"] = details
        elif violation_type == "wrong_way":
            violation["laneDetails"] = details

        self.violations.insert(0, violation)
        print(f"Added {violation_type} violation for track {track_id}")

    def get_violations(self, limit=None):
        """
        Get all violations, optionally limited to a certain number

        Args:
            limit (int, optional): Maximum number of violations to return

        Returns:
            list: List of violation records
        """
        if limit:
            return self.violations[:limit]
        return self.violations

    def get_violation(self, violation_id):
        """
        Get a specific violation by ID

        Args:
            violation_id (str): ID of the violation to retrieve

        Returns:
            dict: Violation record or None if not found
        """
        for violation in self.violations:
            if violation["id"] == violation_id:
                return violation
        return None

    def is_violated_already(self, violation_id):
        for violation in self.violations:
            if violation["id"].startswith(violation_id):
                return violation["type"]
        return None
