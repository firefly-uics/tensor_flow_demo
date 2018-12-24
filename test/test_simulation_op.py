import unittest

from ts.simulation_change import Change
from ts.simulation_high_low import HighLow
from ts.simulation_history import SimulationHistory
from ts.simulation_op import Op
from ts.simulation_order import Order


class SimulationTestCase(unittest.TestCase):
    def setUp(self):
        self.simulation = Op('002396.SZ', '2018-11-01', '2018-11-30')

    def test_execute(self):
        self.simulation.execute()
