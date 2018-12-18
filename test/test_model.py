from ts.build_model import BuildModel
from ts.db_utils import get_daily


def test_create():
    column_names = ['open', 'high', 'close', 'change', 'vol']
    code = '002396.SZ'
    df = get_daily(code)

    build_model = BuildModel(code + '_low', df, column_names, 'low')

    build_model.create_model()

    column_names = ['open', 'low', 'close', 'change', 'vol']
    code = '002396.SZ'
    df = get_daily(code)

    build_model = BuildModel(code + '_high', df, column_names, 'high')

    build_model.create_model()


test_create()
