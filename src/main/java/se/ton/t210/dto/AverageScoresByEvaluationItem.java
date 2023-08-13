package se.ton.t210.dto;

import lombok.Getter;

@Getter
public class AverageScoresByEvaluationItem {

    private final Long evaluationItemId;
    private final double avgScore;

    public AverageScoresByEvaluationItem(Long evaluationItemId, double avgScore) {
        this.evaluationItemId = evaluationItemId;
        this.avgScore = avgScore;
    }
}
