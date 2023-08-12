package se.ton.t210.domain;

import io.jsonwebtoken.security.Keys;
import lombok.Getter;
import se.ton.t210.utils.auth.JwtUtils;

import java.nio.charset.StandardCharsets;
import java.security.Key;
import java.util.Map;

@Getter
public class TokenSecret {

    private final Key secretKey;

    public TokenSecret(Key secretKey) {
        this.secretKey = secretKey;
    }

    public TokenSecret(String secret) {
        this(Keys.hmacShaKeyFor(secret.getBytes(StandardCharsets.UTF_8)));
    }

    public void validateToken(String token) {
        JwtUtils.validate(secretKey, token);
    }

    public String createToken(Map<String, Object> tokenPayload, int expiredTime) {
        return JwtUtils.createToken(secretKey, tokenPayload, expiredTime);
    }

    public boolean isExpired(String token) {
        return JwtUtils.isExpired(secretKey, token);
    }

    public String getPayloadValue(String tokenPayloadEmailKey, String token) {
        return JwtUtils.tokenClaimsFrom(secretKey, token, true)
            .get(tokenPayloadEmailKey, String.class);
    }
}
