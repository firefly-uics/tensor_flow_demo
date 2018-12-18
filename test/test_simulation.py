import unittest

from ts.simulation_high_low import HighLow
from ts.simulation_history import SimulationHistory


class SimulationTestCase(unittest.TestCase):
    def setUp(self):
        self.simulation = HighLow('002396.SZ', '2018-11-01', '2018-12-01')

    def test_execute(self):
        self.simulation.execute()
