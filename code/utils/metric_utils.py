from sklearn.metrics import log_loss


def get_score(solution, submission, id_column_name="id"):
    if id_column_name not in solution.columns or id_column_name not in submission.columns:
        raise ValueError(f"Column '{id_column_name}' does not exist in one of the DataFrames")

    solution_ids = set(solution[id_column_name].values.tolist())
    submission_ids = set(submission[id_column_name].values.tolist())

    # Assert submission is made for all solution ids
    assert solution_ids == submission_ids

    df = solution.merge(submission, on=id_column_name)

    required_columns = ["winner_model_a", "winner_model_b", "winner_tie", "label"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing in the merged DataFrame")

    y_pred = df[["winner_model_a", "winner_model_b", "winner_tie"]].values
    if y_pred.max() > 1 or y_pred.min() < 0:
        raise ValueError("Submitted values were not valid probabilities")

    y_true = df["label"].values
    assert set(y_true) <= {0, 1, 2}

    # score = log_loss(y_true, y_pred, eps='auto')
    score = log_loss(y_true, y_pred)

    return {"lb": round(score, 4)}
