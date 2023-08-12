package se.ton.t210.service.token;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import se.ton.t210.cache.TokenCache;
import se.ton.t210.cache.TokenCacheRepository;
import se.ton.t210.dto.MemberTokens;
import se.ton.t210.exception.AuthException;
import se.ton.t210.service.JwtUtils;

import java.time.Duration;
import java.time.LocalTime;
import java.util.Map;

@Service
public class MemberTokenService {

    @Value("${auth.jwt.token.payload.email.key:email}")
    private String tokenPayloadEmailKey;

    @Value("${auth.jwt.token.access.ttl.time:1800}")
    private int accessTokenExpireTime;

    @Value("${auth.jwt.token.refresh.ttl.time:259200}")
    private int refreshTokenExpireTime;

    private final JwtUtils jwtUtils;
    private final TokenCacheRepository tokenCacheRepository;

    public MemberTokenService(JwtUtils jwtUtils, TokenCacheRepository tokenCacheRepository) {
        this.jwtUtils = jwtUtils;
        this.tokenCacheRepository = tokenCacheRepository;
    }

    public MemberTokens createTokensByEmail(String email) {
        final Map<String, Object> tokenPayload = Map.of(tokenPayloadEmailKey, email);
        final String accessToken = jwtUtils.createToken(tokenPayload, accessTokenExpireTime);
        final String refreshToken = jwtUtils.createToken(tokenPayload, refreshTokenExpireTime);
        tokenCacheRepository.save(new TokenCache(email, accessToken, refreshToken));
        return new MemberTokens(accessToken, refreshToken);
    }

    public MemberTokens reissue(String accessToken, String refreshToken) {
        if (!jwtUtils.isExpired(accessToken)) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "Reissue request is invalid");
        }
        if (jwtUtils.isExpired(refreshToken)) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "Reissue request is invalid");
        }
        final String usernameFromAT = jwtUtils.tokenClaimsFrom(accessToken, true)
                .get(tokenPayloadEmailKey, String.class);
        final String usernameFromRT = jwtUtils.tokenClaimsFrom(refreshToken, true)
                .get(tokenPayloadEmailKey, String.class);
        if (!usernameFromAT.equals(usernameFromRT)) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "Reissue request is invalid");
        }

        final TokenCache tokenCache = tokenCacheRepository.findById(usernameFromAT)
                .orElseThrow(() -> new AuthException(HttpStatus.UNAUTHORIZED, "Token is invalid"));
        if (!tokenCache.getRefreshToken().equals(refreshToken) || !tokenCache.getAccessToken().equals(accessToken)) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "Reissue request is invalid");
        }
        final long afterSeconds = Duration.between(tokenCache.getCreatedTime(), LocalTime.now()).getSeconds();
        if (afterSeconds > refreshTokenExpireTime) {
            throw new AuthException(HttpStatus.REQUEST_TIMEOUT, "Email valid time is exceed");
        }
        return createTokensByEmail(tokenCache.getUsername());
    }
}
