package se.ton.t210.dto;

import lombok.Getter;

import javax.validation.constraints.NotBlank;

@Getter
public class LogInRequest {

    @NotBlank
    private String username;

    @NotBlank
    private String password;

    public LogInRequest() {
    }

    public LogInRequest(String username, String password) {
        this.username = username;
        this.password = password;
    }
}
