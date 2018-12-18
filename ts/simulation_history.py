import logging

from ts.db_utils import get_daily, get_hist_data

logging.basicConfig(level=logging.DEBUG)


class account:
    init = 10000
    total = 10000
    T0 = 0
    T1 = 0
    change_unit = 100
    last_price = 0
    change_fee = 0

    def __str__(self) -> str:
        return 'total:%s, t0:%s, t1:%s, clean total:%s, last_price:%s, change_fee:%s' % (self.total, self.T0, self.T1, self.last_price * (self.T0 + self.T1)+self.total, self.last_price, self.change_fee)


class SimulationHistory:

    def __init__(self, code, start=None, end=None):
        self._code = code
        self._start = start
        self._end = end

    def get_code(self):
        return self._code

    def execute(self):
        df = get_hist_data(self._code, self._start, self._end)

        logging.info("data count:%s", len(df))

        old_index = ''

        for index, row in df.iterrows():
            price = row['close']
            account.last_price = price

            is_change = old_index != index

            if is_change:
                old_index = index
                account.T1 = account.T1 + account.T0
                account.T0 = 0

            if self.is_buy(index, row):
                self.buy(price)

            if self.is_sell(index, row):
                self.sell(price)

            if self.is_clean(index, row):
                self.clean(price)

    def buy(self, price):
        if account.total > (account.change_unit * price):
            logging.debug("buy success")
            account.total = account.total - account.change_unit * price - self.buy_fee(account.change_unit, price)
            account.T0 = account.T0 + account.change_unit
            self.print_change()

    def sell(self, price):
        if account.T1 > account.change_unit:
            logging.info("sell success")
            account.total = account.total + account.change_unit * price - self.sell_fee(account.change_unit, price)
            account.T1 = account.T1 - account.change_unit
            self.print_change()

    def is_buy(self, index, row):
        return True

    def is_sell(self, index, row):
        return True

    def buy_fee(self, change_unit, price):
        b_fee = change_unit * price * 0.0001 + change_unit / 1000 * 1.1
        account.change_fee = account.change_fee + b_fee
        return b_fee

    def print_change(self):
        logging.info('account:%s', account())

    def sell_fee(self, change_unit, price):
        s_fee = change_unit * price * 0.0001 + change_unit * price * 0.0003 + change_unit / 1000 * 1
        account.change_fee = account.change_fee + s_fee
        return s_fee

    def is_clean(self, index, row):
        return False

    def clean(self, price):
        logging.info("清空股票账户")
        account.total = account.total + account.T1 * price - self.sell_fee(account.T1, price)
        account.T1 = 0
        self.print_change()
