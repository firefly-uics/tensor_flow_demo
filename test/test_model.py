import seaborn as sns

from ts.build_model import BuildModel
from ts.db_utils import get_daily
import matplotlib.pyplot as plt

def test_create():
    pass
    # column_names = ['open', 'high', 'close', 'change', 'vol']
    # code = '002396.SZ'
    # df = get_daily(code)
    #
    # build_model = BuildModel(code + '_low', df, column_names, 'low')
    #
    # build_model.create_model()
    #
    # column_names = ['open', 'low', 'close', 'change', 'vol']
    # code = '002396.SZ'
    # df = get_daily(code)
    #
    # build_model = BuildModel(code + '_high', df, column_names, 'high')
    #
    # build_model.create_model()


def test_create_pchange():
    column_names = ['open', 'high', 'low',  'pre_close', 'pct_chg', 'amount','vol']
    code = '002396.SZ'
    df = get_daily(code)
    #
    # dataset = df.copy()
    # print(dataset.tail())
    #
    # print(dataset.isna().sum())
    #
    # dataset = dataset.dropna()
    #
    # train_dataset = dataset.sample(frac=0.8, random_state=0)
    # test_dataset = dataset.drop(train_dataset.index)
    #
    # # df.loc[:, 'pct_chg'] = df['pct_chg'].apply(lambda x: int(x))
    # #
    # # print(df)
    #
    # sns.pairplot(train_dataset[['open', 'high', 'low', 'close', 'pre_close', 'pct_chg', 'amount','vol']], diag_kind="kde")
    #
    # plt.show()

    build_model = BuildModel(code + '_low', df, column_names, 'low')

    build_model.create_model()

    # train_stats = train_dataset.describe()

    # print(train_stats)

test_create_pchange()
