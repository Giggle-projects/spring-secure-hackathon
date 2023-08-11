package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.ApplicationType;
import se.ton.t210.domain.Member;
import se.ton.t210.domain.type.Gender;

import javax.validation.constraints.Email;
import javax.validation.constraints.NotBlank;
import java.time.LocalDate;

@Getter
public class SignUpRequest {

    @NotBlank
    private String name;

    private Gender gender;

    @NotBlank
    @Email
    private String email;

    @NotBlank
    private String password;

    @NotBlank
    private String rePassword;

    private ApplicationType applicationType;

    private Boolean isAgreePersonalInfo;

    public SignUpRequest() {
    }

    public SignUpRequest(String name, Gender gender, String email, String password,
                         String rePassword, ApplicationType applicationType, Boolean isAgreePersonalInfo) {
        this.name = name;
        this.gender = gender;
        this.email = email;
        this.password = password;
        this.rePassword = rePassword;
        this.applicationType = applicationType;
        this.isAgreePersonalInfo = isAgreePersonalInfo;
    }

    public Member toEntity() {
        return new Member(name, email, password, gender, applicationType, LocalDate.now(), LocalDate.now());
    }

    public void validateRequest() {
        if (!password.equals(rePassword)) {
            throw new IllegalArgumentException("Invalid request arguments");
        }
        if (!isAgreePersonalInfo) {
            throw new IllegalArgumentException("Must agree personal Info");
        }
    }
}
