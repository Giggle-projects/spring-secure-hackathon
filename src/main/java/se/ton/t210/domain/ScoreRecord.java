package se.ton.t210.domain;

import lombok.Getter;

import javax.persistence.*;
import java.time.LocalDate;
import se.ton.t210.domain.converter.ScoreRecordYearAndMonthConverter;

@Getter
@Entity
public class ScoreRecord {

    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Id
    private Long id;
    private Long memberId;
    private Long judgingId;
    private int score;

    @Convert(converter = ScoreRecordYearAndMonthConverter.class)
    private LocalDate createdAt = LocalDate.now();

    public ScoreRecord() {
    }

    public ScoreRecord(Long id, Long memberId, Long judgingId, int score) {
        this.id = id;
        this.memberId = memberId;
        this.judgingId = judgingId;
        this.score = score;
    }

    public ScoreRecord(Long memberId, Long judgingId, int score) {
        this(null, memberId, judgingId, score);
    }
}
