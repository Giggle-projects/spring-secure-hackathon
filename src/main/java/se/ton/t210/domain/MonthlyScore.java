package se.ton.t210.domain;

import lombok.Getter;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;

@Getter
@Entity
public class MonthlyScore {

    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Id
    private Long id;
    private Long memberId;
    private int score;

    public MonthlyScore() {
    }

    public MonthlyScore(Long id, Long memberId, int score) {
        this.id = id;
        this.memberId = memberId;
        this.score = score;
    }

    public MonthlyScore(Long memberId, int score) {
        this(null, memberId, score);
    }
}
