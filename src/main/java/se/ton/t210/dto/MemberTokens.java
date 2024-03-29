package se.ton.t210.dto;

import lombok.Getter;

@Getter
public class MemberTokens {

    private final String accessToken;
    private final String refreshToken;

    public MemberTokens(String accessToken, String refreshToken) {
        this.accessToken = accessToken;
        this.refreshToken = refreshToken;
    }
}
