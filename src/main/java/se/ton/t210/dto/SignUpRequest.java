package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.Member;
import se.ton.t210.domain.type.ApplicationType;

import javax.validation.constraints.Email;
import javax.validation.constraints.NotBlank;
import javax.validation.constraints.Pattern;
import java.time.LocalDate;

@Getter
public class SignUpRequest {

    @NotBlank
    @Pattern(regexp = "^[가-힣]{2,4}$",
            message = "이름은 한국 이름으로 2~4글자 까지 가능합니다.")
    private String name;

    @NotBlank
    @Email
    private String email;

    @NotBlank
    @Pattern(regexp = "(?=.*[0-9])(?=.*[a-zA-Z])(?=.*\\W)(?=\\S+$).{9,16}",
            message = "비밀번호는 영문, 숫자, 특수기호가 적어도 1개 이상씩 포함된 9자 ~ 16자의 비밀번호여야 합니다.")
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
