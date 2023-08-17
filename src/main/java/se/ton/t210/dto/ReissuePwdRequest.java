package se.ton.t210.dto;

import lombok.Getter;

import javax.validation.constraints.NotBlank;
import javax.validation.constraints.Pattern;

@Getter
public class ReissuePwdRequest {

    @NotBlank
    @Pattern(regexp = "(?=.*[0-9])(?=.*[a-zA-Z])(?=.*\\\\W)(?=\\\\S+$).{9,16}")
    private String password;

    public ReissuePwdRequest() {
    }

    public ReissuePwdRequest(String password) {
        this.password = password;
    }
}
