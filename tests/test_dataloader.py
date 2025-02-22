from aiq.dataset.loader import DataLoader


if __name__ == "__main__":
    df = DataLoader.load_instruments(
        data_dir=None,
        market="000300.SH",
        start_time="2025-01-01",
        end_time="2025-01-03",
    )
    print(df)
