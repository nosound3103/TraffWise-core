class Vehicle:
    def __init__(self, id, position, speed=0, direction=None):
        self.id = id
        self.position = position
        self.speed = speed
        self.direction = direction

        self.history = [position]


class Bus(Vehicle):
    def __init__(self, id, position, speed=0, direction=None):
        super().__init__(id, position, speed, direction)

    def __str__(self):
        return f"Bus {self.id} at {self.position} moving {self.speed} units in direction {self.direction}"


class Car(Vehicle):
    def __init__(self, id, position, speed=0, direction=None):
        super().__init__(id, position, speed, direction)

    def __str__(self):
        return f"Car {self.id} at {self.position} moving {self.speed} units in direction {self.direction}"


class Truck(Vehicle):
    def __init__(self, id, position, speed=0, direction=None):
        super().__init__(id, position, speed, direction)

    def __str__(self):
        return f"Truck {self.id} at {self.position} moving {self.speed} units in direction {self.direction}"


class Motorcycle(Vehicle):
    def __init__(self, id, position, speed=0, direction=None):
        super().__init__(id, position, speed, direction)

    def __str__(self):
        return f"Motorcycle {self.id} at {self.position} moving {self.speed} units in direction {self.direction}"
