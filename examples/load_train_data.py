import pandas as pd


def load_train_data(data_path: str) -> pd.DataFrame:
    # read from 3 rd row
    df = pd.read_excel(data_path, skiprows=2)
    df.columns = df.columns.str.lower()
    df = df.replace("绝对值", "absolute")
    df = df.replace("相对值", "relative")
    # rename column name "序号"
    df = df.rename(columns={"类别": "type", "序号": "id"})
    return df


def get_by_id(df: pd.DataFrame, id: str) -> pd.DataFrame | None:
    row = df[df["id"] == id]
    if row.empty:
        return None
    return row.iloc[0]


if __name__ == "__main__":
    df = load_train_data(
        "data/色差仪/爱色丽MA5QC色差仪/上汽江宁工厂-爱色丽MA-5-QC.xlsx"
    )
    print(df.head())
