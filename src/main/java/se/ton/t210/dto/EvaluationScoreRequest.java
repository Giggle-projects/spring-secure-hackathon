package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.EvaluationItemScore;

@Getter
public class EvaluationScoreRequest {

    private final Long evaluationItemId;
    private final Float score;

    public EvaluationScoreRequest(Long evaluationItemId, Float score) {
        this.evaluationItemId = evaluationItemId;
        this.score = score;
    }

    public EvaluationItemScore toEntity(Long memberId) {
        return new EvaluationItemScore(memberId, evaluationItemId, score);
    }
}
