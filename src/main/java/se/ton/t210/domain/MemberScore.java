package se.ton.t210.domain;

import java.time.LocalDate;
import javax.persistence.Convert;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import lombok.Getter;
import se.ton.t210.domain.converter.ScoreRecordYearAndMonthConverter;

@Getter
@Entity
public class MemberScore {

    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Id
    private Long id;

    private Long memberId;

    private int score;

    @Convert(converter = ScoreRecordYearAndMonthConverter.class)
    private final LocalDate createdAt = LocalDate.now();

    public MemberScore() {
    }

    public MemberScore(Long id, Long memberId, int score) {
        this.id = id;
        this.memberId = memberId;
        this.score = score;
    }

    public MemberScore(Long memberId, int score) {
        this.memberId = memberId;
        this.score = score;
    }
}
