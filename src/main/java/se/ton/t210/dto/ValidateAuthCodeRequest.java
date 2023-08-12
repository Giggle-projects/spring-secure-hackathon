package se.ton.t210.dto;

import lombok.Getter;

import javax.validation.constraints.Email;

@Getter
public class ValidateAuthCodeRequest {

    @Email
    private String email;

    private String authCode;

    public ValidateAuthCodeRequest() {
    }

    public ValidateAuthCodeRequest(String email, String authCode) {
        this.email = email;
        this.authCode = authCode;
    }
}
