package se.ton.t210.dto;

import lombok.Getter;

import javax.validation.constraints.Email;

@Getter
public class EmailRequest {

    @Email
    private String email;

    public EmailRequest(String email) {
        this.email = email;
    }

    public EmailRequest() {
    }
}
