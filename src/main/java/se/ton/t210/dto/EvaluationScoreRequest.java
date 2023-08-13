package se.ton.t210.dto;

import lombok.Getter;

@Getter
public class EvaluationScoreRequest {

    private final Long evaluationItemId;
    private final int score;

    public EvaluationScoreRequest(Long evaluationItemId, int score) {
        this.evaluationItemId = evaluationItemId;
        this.score = score;
    }
}
