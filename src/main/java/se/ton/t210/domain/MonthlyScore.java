package se.ton.t210.domain;

import lombok.Getter;
import se.ton.t210.domain.converter.ScoreRecordYearAndMonthConverter;
import se.ton.t210.domain.type.ApplicationType;

import javax.persistence.*;
import java.time.LocalDate;

@Getter
@Entity
public class MonthlyScore {

    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Id
    private Long id;
    private ApplicationType applicationType;
    private Long memberId;
    private int score;

    @Convert(converter = ScoreRecordYearAndMonthConverter.class)
    private final LocalDate yearMonth = LocalDate.now();

    public MonthlyScore() {
    }

    public MonthlyScore(Long id, ApplicationType applicationType, Long memberId, int score) {
        this.id = id;
        this.applicationType = applicationType;
        this.memberId = memberId;
        this.score = score;
    }

    public MonthlyScore(ApplicationType applicationType, Long memberId, int score) {
        this(null, applicationType, memberId, score);
    }

    public static MonthlyScore of(Member member, int score) {
        return new MonthlyScore(member.getApplicationType(), member.getId(), score);
    }

    public void update(int evaluationScoreSum) {
        this.score = evaluationScoreSum;
    }
}
