from ml.data import make_data, FEATURES

def test_synthetic_data():
    df = make_data(100)
    assert len(df) == 100
    assert set(FEATURES).issubset(df.columns)
    assert df["action"].nunique() >= 2
