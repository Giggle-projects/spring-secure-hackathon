package se.ton.t210.domain;

import lombok.Getter;

import javax.persistence.*;
import java.time.LocalDate;
import se.ton.t210.domain.converter.ScoreRecordYearAndMonthConverter;

@Getter
@Entity
public class MonthlyScoreItem {

    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Id
    private Long id;
    private Long memberId;
    private Long evaluationItemId;
    private Long monthlyScoreId;

    @Convert(converter = ScoreRecordYearAndMonthConverter.class)
    private final LocalDate yearMonth = LocalDate.now();

    public MonthlyScoreItem() {
    }

    public MonthlyScoreItem(Long id, Long memberId, Long evaluationItemId, Long monthlyScoreId) {
        this.id = id;
        this.memberId = memberId;
        this.evaluationItemId = evaluationItemId;
        this.monthlyScoreId = monthlyScoreId;
    }

    public MonthlyScoreItem(Long memberId, Long evaluationItemId, Long monthlyScoreId) {
        this(null, memberId, evaluationItemId, monthlyScoreId);
    }
}
