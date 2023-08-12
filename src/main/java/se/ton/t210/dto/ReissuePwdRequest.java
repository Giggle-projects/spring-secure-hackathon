package se.ton.t210.dto;

import lombok.Getter;

import javax.validation.constraints.NotBlank;

@Getter
public class ReissuePwdRequest {

    @NotBlank
    private String password;

    public ReissuePwdRequest() {
    }

    public ReissuePwdRequest(String password) {
        this.password = password;
    }
}
