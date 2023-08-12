package se.ton.t210.domain;

import lombok.Getter;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.domain.type.Gender;

import javax.persistence.*;
import javax.validation.constraints.Email;
import javax.validation.constraints.NotNull;
import java.time.LocalDate;

@Getter
@Entity
public class Member {

    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Id
    private Long id;

    @NotNull
    private String name;

    @Email
    @NotNull
    String email;

    @NotNull
    private String password;

    @Enumerated(EnumType.STRING)
    @NotNull
    private Gender gender;

    @Enumerated(EnumType.STRING)
    @NotNull
    private ApplicationType applicationType;

    @NotNull
    private LocalDate createdAt;

    @NotNull
    private LocalDate updatedAt;

    public Member() {
    }

    public Member(Long id, String name, String email, String password, Gender gender,
                  ApplicationType applicationType, LocalDate createdAt, LocalDate updatedAt) {
        this.id = id;
        this.name = name;
        this.email = email;
        this.password = password;
        this.gender = gender;
        this.applicationType = applicationType;
        this.createdAt = createdAt;
        this.updatedAt = updatedAt;
    }

    public Member(String name, String email, String password, Gender gender,
                  ApplicationType applicationType, LocalDate createdAt, LocalDate updatedAt) {
        this(null, name, email, password, gender, applicationType, createdAt, updatedAt);
    }

    public Member(Long id, String name, String email, String password, Gender gender, ApplicationType applicationType) {
        this(id, name, email, password, gender, applicationType, LocalDate.now(), LocalDate.now());
    }

    public void reissuePwd(String newPwd) {
        this.password = newPwd;
    }
}
