package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.Member;
import se.ton.t210.domain.type.ApplicationType;

import javax.validation.constraints.Email;
import javax.validation.constraints.NotBlank;
import java.time.LocalDate;

@Getter
public class SignUpRequest {

    @NotBlank
    private String name;

    @NotBlank
    @Email
    private String email;

    @NotBlank
    private String password;

    private ApplicationType applicationType;

    public SignUpRequest() {
    }

    public SignUpRequest(String name, String email, String password, ApplicationType applicationType) {
        this.name = name;
        this.email = email;
        this.password = password;
        this.applicationType = applicationType;
    }

    public Member toEntity() {
        return new Member(name, email, password, applicationType, LocalDate.now(), LocalDate.now());
    }
}
