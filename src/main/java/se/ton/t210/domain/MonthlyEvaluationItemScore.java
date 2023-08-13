package se.ton.t210.domain;

import lombok.Getter;

import javax.persistence.*;
import java.time.LocalDate;
import se.ton.t210.domain.converter.ScoreRecordYearAndMonthConverter;

@Getter
@Entity
public class MonthlyEvaluationItemScore {

    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Id
    private Long id;
    private Long memberId;
    private Long evaluationItemId;
    private int score;

    @Convert(converter = ScoreRecordYearAndMonthConverter.class)
    private final LocalDate yearMonth = LocalDate.now();

    public MonthlyEvaluationItemScore() {
    }

    public MonthlyEvaluationItemScore(Long id, Long memberId, Long evaluationItemId, int score) {
        this.id = id;
        this.memberId = memberId;
        this.evaluationItemId = evaluationItemId;
        this.score = score;
    }

    public MonthlyEvaluationItemScore(Long memberId, Long evaluationItemId, int score) {
        this(null, memberId, evaluationItemId, score);
    }

    public static MonthlyEvaluationItemScore of(Member member, EvaluationItem evaluationItem, int score) {
        return new MonthlyEvaluationItemScore(
            member.getId(),
            evaluationItem.getId(),
            score
        );
    }
}
