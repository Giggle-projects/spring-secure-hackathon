package se.ton.t210.domain;

import lombok.Getter;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import java.time.LocalDateTime;

@Getter
@Entity
public class AccessDateTime {

    @GeneratedValue(strategy = GenerationType.AUTO)
    @Id
    private Long id;
    private LocalDateTime accessTime;
    private Long memberId;

    public AccessDateTime() {
    }

    public AccessDateTime(Long id, LocalDateTime accessTime, Long memberId) {
        this.id = id;
        this.accessTime = accessTime;
        this.memberId = memberId;
    }

    public AccessDateTime(LocalDateTime accessTime, Long memberId) {
        this(null, accessTime, memberId);
    }
}
