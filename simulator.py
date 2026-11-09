import tkinter as tk

from world import PhysicsWorld


class SimulatorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.world = PhysicsWorld()
        self.running = False
        self.camera_x = self.world.width / 2
        self.camera_y = self.world.height / 2
        self.camera_z = self.world.depth / 2
        self.camera_drag = False
        self.camera_drag_moved = False
        self.zoom = 1.0
        self.time_scale = 1.0
        self.max_time_scale = 1000.0