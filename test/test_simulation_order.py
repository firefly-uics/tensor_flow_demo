import unittest

from ts.simulation_change import Change
from ts.simulation_high_low import HighLow
from ts.simulation_history import SimulationHistory
from ts.simulation_order import Order


class SimulationTestCase(unittest.TestCase):
    def setUp(self):
        self.simulation = Order('002396.SZ', '2018-12-01', '2018-12-30')

    def test_execute(self):
        self.simulation.execute()
