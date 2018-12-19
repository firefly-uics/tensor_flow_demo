import unittest

from ts.simulation_change import Change
from ts.simulation_high_low import HighLow
from ts.simulation_history import SimulationHistory


class SimulationTestCase(unittest.TestCase):
    def setUp(self):
        self.simulation = Change('002396.SZ', '2018-10-20', '2018-10-30')

    def test_execute(self):
        self.simulation.execute()
