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

    private int score;

    public EvaluationScoreSection() {
    }

    public EvaluationScoreSection(Long id, Long evaluationItemId, int sectionBaseScore, int score) {
        this.id = id;
        this.evaluationItemId = evaluationItemId;
        this.sectionBaseScore = sectionBaseScore;
        this.score = score;
    }

    public EvaluationScoreSection(Long evaluationItemId, int sectionBaseScore, int score) {
        this(null, evaluationItemId, sectionBaseScore, score);
    }
}
