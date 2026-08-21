from chart_prm.verifier import (
    build_candidate,
    evaluate_question_group,
    process_score,
    select_by_majority_vote,
    select_by_process_score,
    summarize,
)


def test_process_score_is_mean_pass_rate():
    assert process_score([{"score": 1}, {"score": 1}, {"score": 0}]) == 2 / 3
    assert process_score([{"score": 1}]) == 1.0
    assert process_score([]) is None
    assert process_score([{"score": None}]) is None


def test_build_candidate_requires_score_and_answer():
    meta = {"rollout_index": 0, "ground_truth": "42", "model_final_answer": "42"}
    candidate = build_candidate(meta, [{"score": 1}, {"score": 1}])
    assert candidate == {
        "rollout_index": 0,
        "final_answer": "42",
        "process_score": 1.0,
        "correct": True,
    }

    assert build_candidate(meta, []) is None  # no judge score
    assert (
        build_candidate({**meta, "model_final_answer": "  "}, [{"score": 1}]) is None
    )  # blank final answer


def test_build_candidate_uses_whole_token_matching():
    meta = {"rollout_index": 0, "ground_truth": "4", "model_final_answer": "The answer is 94."}
    candidate = build_candidate(meta, [{"score": 1}])
    assert candidate["correct"] is False


def test_select_by_process_score_prefers_higher_score_then_lower_index():
    candidates = [
        {"rollout_index": 2, "process_score": 0.5, "final_answer": "a", "correct": False},
        {"rollout_index": 0, "process_score": 1.0, "final_answer": "b", "correct": True},
        {"rollout_index": 1, "process_score": 1.0, "final_answer": "c", "correct": False},
    ]
    picked = select_by_process_score(candidates)
    assert picked["rollout_index"] == 0  # tie on 1.0 broken by lowest index


def test_select_by_majority_vote_breaks_ties_deterministically():
    candidates = [
        {"rollout_index": 3, "process_score": 0.0, "final_answer": "Reverse", "correct": True},
        {"rollout_index": 1, "process_score": 0.0, "final_answer": "Convolve", "correct": False},
        {"rollout_index": 2, "process_score": 0.0, "final_answer": "reverse", "correct": True},
    ]
    picked = select_by_majority_vote(candidates)
    assert picked["final_answer"].lower() == "reverse"
    assert picked["rollout_index"] == 2  # earliest member of the winning (2-vote) group


def test_evaluate_question_group_requires_at_least_two_candidates():
    single = [{"rollout_index": 0, "process_score": 1.0, "final_answer": "a", "correct": True}]
    try:
        evaluate_question_group(single)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_evaluate_question_group_scores_all_strategies():
    candidates = [
        {"rollout_index": 0, "process_score": 1.0, "final_answer": "42", "correct": True},
        {"rollout_index": 1, "process_score": 0.0, "final_answer": "17", "correct": False},
        {"rollout_index": 2, "process_score": 0.0, "final_answer": "17", "correct": False},
    ]
    result = evaluate_question_group(candidates)
    assert result["n_candidates"] == 3
    assert result["random_expected_correct"] == 1 / 3
    assert result["prm_correct"] == 1.0  # PRM picks rollout 0 (score 1.0)
    assert result["majority_correct"] == 0.0  # majority answer "17" is wrong
    assert result["oracle_correct"] == 1.0  # a correct rollout exists


def test_summarize_averages_across_questions_and_conditions_on_oracle():
    per_question = [
        {
            "n_candidates": 2,
            "random_expected_correct": 0.5,
            "prm_correct": 1.0,
            "majority_correct": 1.0,
            "oracle_correct": 1.0,
        },
        {
            "n_candidates": 2,
            "random_expected_correct": 0.0,
            "prm_correct": 0.0,
            "majority_correct": 0.0,
            "oracle_correct": 0.0,
        },
    ]
    summary = summarize(per_question)
    assert summary["n_questions"] == 2
    assert summary["random_baseline_accuracy"] == 0.25
    assert summary["prm_best_of_n_accuracy"] == 0.5
    assert summary["oracle_accuracy"] == 0.5
    # only question 1 has oracle_correct == 1, and PRM got it right there
    assert summary["prm_accuracy_when_oracle_positive"] == 1.0
