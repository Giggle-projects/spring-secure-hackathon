package se.ton.t210.token;

import lombok.Getter;

@Getter
public class TokenData {

    private final String accessToken;
    private final String refreshToken;

    public TokenData(String accessToken, String refreshToken) {
        this.accessToken = accessToken;
        this.refreshToken = refreshToken;
    }
}
