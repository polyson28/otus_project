import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.features.pipeline import TimeSeriesFeaturePipeline
from src.io.s3 import read_csv, read_excel, write_parquet


def _read_debug_dataset() -> tuple[Path, object]:
    csv_path = PROJECT_ROOT / "Data" / "final.csv"
    xlsx_path = PROJECT_ROOT / "Project 2_2023.xlsx"

    if csv_path.exists():
        return csv_path, read_csv(csv_path)
    if xlsx_path.exists():
        return xlsx_path, read_excel(xlsx_path)

    raise FileNotFoundError("No local debug dataset found in Data/final.csv or Project 2_2023.xlsx.")


def main() -> None:
    config = load_config()
    feature_config = config["feature_engineering"]
    dataset_path, df = _read_debug_dataset()

    pipeline = TimeSeriesFeaturePipeline(
        date_col=config["date_col"],
        target_col=config["target_col"],
        lags=feature_config["lags"],
        rolling_windows=feature_config["rolling_windows"],
        use_calendar_features=feature_config["use_calendar_features"],
        use_tax_features=feature_config["use_tax_features"],
        use_macro_features=feature_config["use_macro_features"],
    )

    X = pipeline.fit_transform(df)

    artifacts_dir = PROJECT_ROOT / "artifacts" / "debug"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    features_path = artifacts_dir / "features.parquet"
    pipeline_path = artifacts_dir / "feature_pipeline.pkl"

    write_parquet(X, features_path)
    pipeline.save(str(pipeline_path))

    print(f"Built features from {dataset_path}")
    print(f"Features shape: {X.shape}")
    print(f"Saved features to {features_path}")
    print(f"Saved feature pipeline to {pipeline_path}")


if __name__ == "__main__":
    main()
