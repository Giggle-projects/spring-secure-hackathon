package se.ton.t210.domain;

import lombok.Getter;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.validation.constraints.NotNull;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Getter
@Entity
public class BlackList {

    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Id
    private Long id;
    private Long memberId;

    @NotNull
    private LocalDate createdAt;

    public BlackList() {
    }

    public BlackList(Long id, Long memberId, LocalDate createdAt) {
        this.id = id;
        this.memberId = memberId;
        this.createdAt = createdAt;
    }

    public BlackList(Long memberId, LocalDate createdAt) {
        this(null, memberId, createdAt);
    }
}
