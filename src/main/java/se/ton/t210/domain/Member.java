package se.ton.t210.domain;

import lombok.Getter;
import org.springframework.web.client.HttpServerErrorException;
import se.ton.t210.domain.converter.SymmetricEncryptionConverter;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.dto.ResetPersonalInfoRequest;

import javax.persistence.*;
import javax.validation.constraints.Email;
import javax.validation.constraints.NotNull;
import java.time.LocalDate;

import se.ton.t210.exception.InnerServiceException;
import se.ton.t210.utils.encript.SHA256Utils;
import se.ton.t210.utils.encript.SupportedAlgorithmType;

@Getter
@Entity
public class Member {

    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Id
    private Long id;

    @Convert(converter = SymmetricEncryptionConverter.class)
    @NotNull
    private String name;

    @Convert(converter = SymmetricEncryptionConverter.class)
    @Email
    @NotNull
    private String email;

    @NotNull
    private String password;

    @Enumerated(EnumType.STRING)
    @NotNull
    private ApplicationType applicationType;

    @NotNull
    private LocalDate createdAt;

    @NotNull
    private LocalDate updatedAt;

    public Member() {
    }

    public Member(Long id, String name, String email, String password,
                  ApplicationType applicationType, LocalDate createdAt, LocalDate updatedAt) {
        this.id = id;
        this.name = name;
        this.email = email;
        this.password = password;
        this.applicationType = applicationType;
        this.createdAt = createdAt;
        this.updatedAt = updatedAt;
    }

    public Member(String name, String email, String password,
                  ApplicationType applicationType, LocalDate createdAt, LocalDate updatedAt) {
        this(null, name, email, password, applicationType, createdAt, updatedAt);
    }

    public Member(String name, String email, String password, ApplicationType applicationType) {
        this(null, name, email, password, applicationType, LocalDate.now(), LocalDate.now());
    }

    public Member(Long id, String name, String email, String password, ApplicationType applicationType) {
        this(id, name, email, password, applicationType, LocalDate.now(), LocalDate.now());
    }

    public void resetApplicationType(ApplicationType applicationType) {
        this.applicationType = applicationType;
    }

    public void validatePassword(String input, PasswordSalt salt) {
        try {
            final String encryptedInput = SHA256Utils.encrypt(input, salt.getSalt());
            if(!this.password.equals(encryptedInput)) {
                throw new IllegalArgumentException();
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new InnerServiceException("Encryption server");
        }
    }

    public Member updatePasswordWith(String password) {
        return new Member(
            this.id,
            this.name,
            this.email,
            password,
            this.applicationType,
            this.createdAt,
            this.updatedAt
        );
    }
}
