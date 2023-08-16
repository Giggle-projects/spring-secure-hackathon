package se.ton.t210.domain;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import lombok.Getter;

@Getter
@Entity
public class PasswordSalt {

    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Id
    private Long id;

    private Long memberId;

    private String salt;

    public PasswordSalt() {
    }

    public PasswordSalt(Long id, Long memberId, String salt) {
        this.id = id;
        this.memberId = memberId;
        this.salt = salt;
    }

    public PasswordSalt(Long memberId, String salt) {
        this(null, memberId, salt);
    }
}
