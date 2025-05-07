from aiq.dataset.loader import DataLoader


if __name__ == "__main__":
    instruments = DataLoader.load_instruments(
        data_dir=None,
        market="000300.SH",
        start_time="2025-01-01",
        end_time="2025-01-03",
    )
    print(instruments)

    features = DataLoader.load_instrument_features(
        data_dir=None,
        instrument="002081.SZ",
        start_time="2025-01-21",
        end_time="2025-02-21",
    )
    print(features.shape)
