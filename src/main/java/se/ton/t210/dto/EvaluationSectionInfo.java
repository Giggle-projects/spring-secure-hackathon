package se.ton.t210.dto;

import lombok.Getter;

@Getter
public class EvaluationSectionInfo {

    private final Long evaluationId;
    private final String evaluationName;
    private final float maxScore;
    private final float minScore;
    private final int evaluationScore;

    public EvaluationSectionInfo(Long evaluationId, String evaluationName, float maxScore, float minScore, int evaluationScore) {
        this.evaluationId = evaluationId;
        this.evaluationName = evaluationName;
        this.maxScore = maxScore;
        this.minScore = minScore;
        this.evaluationScore = evaluationScore;
    }
}
