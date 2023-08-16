package se.ton.t210.dto;

import lombok.Getter;

@Getter
public class CalculateEvaluationScoreResponse {

    private final int score;

    public CalculateEvaluationScoreResponse(int score) {
        this.score = score;
    }
}
