package se.ton.t210.service.token;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import se.ton.t210.service.JwtUtils;

import java.util.Map;

@Service
public class AuthTokenService {

    @Value("${auth.jwt.token.payload.email.key:email}")
    private String tokenPayloadEmailKey;

    @Value("${auth.jwt.token.auth.ttl.time:1800}")
    private int authTokenExpireTime;

    private final JwtUtils jwtUtils;

    public AuthTokenService(JwtUtils jwtUtils) {
        this.jwtUtils = jwtUtils;
    }

    public String createAuthTokenByEmail(String email) {
        final Map<String, Object> tokenPayload = Map.of(tokenPayloadEmailKey, email);
        return jwtUtils.createToken(tokenPayload, authTokenExpireTime);
    }
}