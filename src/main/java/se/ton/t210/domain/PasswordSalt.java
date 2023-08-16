package se.ton.t210.domain;

import java.time.LocalDate;
import javax.persistence.Entity;
import javax.persistence.EnumType;
import javax.persistence.Enumerated;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.validation.constraints.Email;
import javax.validation.constraints.NotNull;
import lombok.Getter;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.dto.ResetPersonalInfoRequest;

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
