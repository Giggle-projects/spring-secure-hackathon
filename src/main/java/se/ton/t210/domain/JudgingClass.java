package se.ton.t210.domain;

import javax.persistence.Entity;
import javax.persistence.EnumType;
import javax.persistence.Enumerated;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import lombok.Getter;

@Getter
@Entity
public class JudgingClass {

    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Id
    private Long id;

    private Long judgingItemId;

    private int targetScore;

    private int takenScore;

    public JudgingClass() {
    }

    public JudgingClass(Long id, Long judgingItemId, int targetScore, int takenScore) {
        this.id = id;
        this.judgingItemId = judgingItemId;
        this.targetScore = targetScore;
        this.takenScore = takenScore;
    }

    public JudgingClass(Long judgingItemId, int targetScore, int takenScore) {
        this(null, judgingItemId, targetScore, takenScore);
    }
}
