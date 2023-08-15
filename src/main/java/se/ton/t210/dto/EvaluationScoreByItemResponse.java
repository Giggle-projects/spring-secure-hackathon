package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.EvaluationItem;
import se.ton.t210.domain.EvaluationItemScore;

@Getter
public class EvaluationScoreByItemResponse {

    private final Long evaluationItemId;
    private final String evaluationItemName;
    private final int evaluationScore;

    public EvaluationScoreByItemResponse(Long evaluationItemId, String evaluationItemName, int evaluationScore) {
        this.evaluationItemId = evaluationItemId;
        this.evaluationItemName = evaluationItemName;
        this.evaluationScore = evaluationScore;
    }
}
