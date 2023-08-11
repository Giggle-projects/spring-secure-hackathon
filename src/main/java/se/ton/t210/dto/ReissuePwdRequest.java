package se.ton.t210.dto;

import lombok.Getter;

import javax.validation.constraints.NotBlank;

@Getter
public class ReissuePwdRequest {

    @NotBlank
    private String password;

    @NotBlank
    private String rePassword;

    public ReissuePwdRequest() {
    }

    public ReissuePwdRequest(String password, String rePassword) {
        this.password = password;
        this.rePassword = rePassword;
    }

    public void validRequest() {
        if (!password.equals(rePassword)) {
            throw new IllegalArgumentException("Invalid request arguments");
        }
    }
}
