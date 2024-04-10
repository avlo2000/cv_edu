from pathlib import Path

import kaggle
import zipfile
import pandas as pd
import ydata_profiling


def fetch_data():
    api = kaggle.api
    print(api.get_config_value("username"))
    api.authenticate()
    api.competition_download_files('planttraits2024', quiet=False)

    path_to_data = Path('planttraits2024')
    if not path_to_data.exists():
        with zipfile.ZipFile('planttraits2024.zip', 'r') as zip_ref:
            zip_ref.extractall(path_to_data)


def prepare_train_report():
    df = pd.read_csv('planttraits2024/train.csv')
    report = ydata_profiling.ProfileReport(df)
    report.to_file("train_report_0.html")
    with open('train_report.html', 'w') as f:
        f.write(report.html)


if __name__ == '__main__':
    fetch_data()
    prepare_train_report()

