package se.ton.t210.dto;

import lombok.Getter;

import javax.validation.constraints.Email;
import javax.validation.constraints.NotBlank;

@Getter
public class SignInRequest {

    @Email
    @NotBlank
    private String email;

    @NotBlank
    private String password;

    public SignInRequest() {
    }

    public SignInRequest(String email, String password) {
        this.email = email;
        this.password = password;
    }
}
