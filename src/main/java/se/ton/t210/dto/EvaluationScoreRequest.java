package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.EvaluationItemScore;

import javax.validation.constraints.NotBlank;

@Getter
public class EvaluationScoreRequest {

    @NotBlank
    private final Long evaluationItemId;

    @NotBlank
    private final Float score;

    public EvaluationScoreRequest(Long evaluationItemId, Float score) {
        this.evaluationItemId = evaluationItemId;
        this.score = score;
    }

    public EvaluationItemScore toEntity(Long memberId) {
        return new EvaluationItemScore(memberId, evaluationItemId, score);
    }
}
