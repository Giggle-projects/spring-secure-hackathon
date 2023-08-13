package se.ton.t210.domain;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import lombok.Getter;

@Getter
@Entity
public class EvaluationScoreSection {

    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Id
    private Long id;
    private Long evaluationItemId;
    private int sectionBaseScore;
    private int evaluationScore;

    public EvaluationScoreSection() {
    }

    public EvaluationScoreSection(Long id, Long evaluationItemId, int sectionBaseScore, int evaluationScore) {
        this.id = id;
        this.evaluationItemId = evaluationItemId;
        this.sectionBaseScore = sectionBaseScore;
        this.evaluationScore = evaluationScore;
    }

    public EvaluationScoreSection(Long evaluationItemId, int sectionBaseScore, int evaluationScore) {
        this(null, evaluationItemId, sectionBaseScore, evaluationScore);
    }
}
