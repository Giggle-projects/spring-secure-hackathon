package se.ton.t210.dto;

import lombok.Getter;
import org.springframework.http.HttpStatus;
import se.ton.t210.domain.Member;

import javax.validation.constraints.NotBlank;
import se.ton.t210.exception.AuthException;

@Getter
public class SignUpRequest {

    @NotBlank
    private String username;

    @NotBlank
    private String password;

    @NotBlank
    private String rePassword;

    public SignUpRequest() {
    }

    public SignUpRequest(String username, String password, String rePassword) {
        this.username = username;
        this.password = password;
        this.rePassword = rePassword;
        if(password.equals(rePassword)) {
            throw new IllegalArgumentException("Invalid request arguments");
        }
    }

    public boolean isValidSignUp() {
        return password.equals(rePassword);
    }

    public Member toEntity() {
        return new Member(username, password);
    }
}
