class VehicleMovementTracker:
    def __init__(self,
                 vehicle_id,
                 vehicle_type,
                 vehicle_location,
                 vehicle_speed):
        self.vehicle_id = vehicle_id
        self.vehicle_type = vehicle_type
        self.vehicle_location = vehicle_location
        self.vehicle_speed = vehicle_speed

    def update_vehicle_location(self, new_location):
        self.vehicle_location = new_location

    def update_vehicle_speed(self, new_speed):
        self.vehicle_speed = new_speed

    def get_vehicle_location(self):
        return self.vehicle_location

    def get_vehicle_speed(self):
        return self.vehicle_speed

    def get_vehicle_id(self):
        return self.vehicle_id

    def get_vehicle_type(self):
        return
