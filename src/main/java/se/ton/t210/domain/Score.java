package se.ton.t210.domain;

import lombok.Getter;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import java.time.LocalDate;

@Getter
@Entity
public class Score {

    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Id
    private Long id;
    private Long userId;
    private Long judgingId;
    private int score;
    private final LocalDate createdAt = LocalDate.now();

    public Score() {
    }

    public Score(Long id, Long userId, Long judgingId, int score) {
        this.id = id;
        this.userId = userId;
        this.judgingId = judgingId;
        this.score = score;
    }

    public Score(Long userId, Long judgingId, int score) {
        this(null, userId, judgingId, score);
    }
}
