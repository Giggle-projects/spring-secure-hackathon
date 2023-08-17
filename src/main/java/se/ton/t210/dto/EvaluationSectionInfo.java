package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.EvaluationItem;
import se.ton.t210.domain.EvaluationScoreSection;

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

    public static EvaluationSectionInfo of(EvaluationItem evaluationItem, float prevItemBaseScore, EvaluationScoreSection section) {
        return new EvaluationSectionInfo(
            evaluationItem.getId(),
            evaluationItem.getName(),
            prevItemBaseScore,
            section.getSectionBaseScore(),
            section.getEvaluationScore()
        );
    }
}
