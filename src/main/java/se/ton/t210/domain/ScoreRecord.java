package se.ton.t210.domain;

import lombok.Getter;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import javax.persistence.*;
import java.time.LocalDate;

@Getter
@Entity
public class ScoreRecord {

    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Id
    private Long id;
    private Long memberId;
    private Long judgingId;
    private int score;
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
