import numpy as np
import pandas as pd


def construct_inputs_by_teams(df: pd.DataFrame, match_df: pd.DataFrame) -> pd.DataFrame:
    """ Constructs a model input using the home team, away team and the odds from a fixture. """

    copy_df = df.dropna(ignore_index=True)

    def fill_team_stats(team_col: str, team: str):
        c = team_col[0]
        # Preserve the user-selected team name.  Only copy/estimate the
        # pre-match statistical columns with the same home/away prefix.
        stat_cols = [
            col for col in copy_df.columns
            if col.startswith(c) and col != team_col
        ]
        team_rows = copy_df[copy_df[team_col] == team]
        if not team_rows.empty:
            match_df[stat_cols] = team_rows.iloc[0][stat_cols].values
            return

        # Promoted/new teams may not yet have a row in the selected top-flight
        # history.  Use robust recent-league medians as a neutral baseline
        # instead of crashing or pretending the club is another fuzzy match.
        recent = copy_df.head(200)
        for column in stat_cols:
            numeric = pd.to_numeric(recent[column], errors='coerce')
            median = numeric.median() if numeric.notna().any() else np.nan
            match_df.at[0, column] = 0 if pd.isna(median) else median

    # 1. Validate the inputs.
    if match_df.shape[0] != 1:
        raise ValueError(f'match_df should contain a single match only, got {match_df.shape[0]} matches.')

    required_cols = ['Home', 'Away', '1', 'X', '2']
    if any([c not in match_df.columns for c in required_cols]):
        raise ValueError(
            f'Missing columns found. Provided columns are: {match_df.columns.tolist()}, '
            f'required columns are: {required_cols}'
        )

    if not copy_df['Date'].is_monotonic_decreasing:
        raise ValueError('Dates of historical dataframe "df" are not sorted in descending order.')

    # 2. Adding df columns to match_df columns.
    match_df = match_df.reindex(columns=copy_df.columns, fill_value=np.nan)
    match_df.at[0, 'Season'] = copy_df.at[0, 'Season']
    match_df['Season'] = match_df['Season'].astype(int)
    match_df.at[0, 'Week'] = copy_df.at[0, 'Week'] + 1
    match_df['Week'] = match_df['Week'].astype(int)

    # 3. Filling stats columns for home, away teams.
    fill_team_stats(team_col='Home', team=match_df.at[0, 'Home'])
    fill_team_stats(team_col='Away', team=match_df.at[0, 'Away'])
    with pd.option_context('future.no_silent_downcasting', True):
        match_df = match_df.fillna(value=0)
    return match_df.infer_objects(copy=False)


def construct_inputs_by_fixture(df: pd.DataFrame, fixture_df: pd.DataFrame) -> pd.DataFrame:
    """ Constructs multiple model inputs using the home team, away team and the odds from a fixture. """

    rows = [
        construct_inputs_by_teams(df=df, match_df=pd.DataFrame(data=[dict(zip(fixture_df.columns, t))]))
        for t in fixture_df.itertuples(index=False)
    ]
    return pd.concat(rows, axis=0, ignore_index=True)
