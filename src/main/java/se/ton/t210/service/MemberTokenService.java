package se.ton.t210.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import se.ton.t210.dto.MemberTokens;
import se.ton.t210.exception.AuthException;
import se.ton.t210.token.JwtUtils;
import se.ton.t210.token.TokenCache;
import se.ton.t210.token.TokenCacheRepository;

import java.util.Map;

@Service
public class MemberTokenService {

    @Value("${auth.jwt.token.payload.username.key:username}")
    private String tokenPayloadUsernameKey;

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

    public MemberTokens createTokensByUsername(String username) {
        final Map<String, Object> tokenPayload = Map.of(tokenPayloadUsernameKey, username);
        final String accessToken = jwtUtils.createToken(tokenPayload, accessTokenExpireTime);
        final String refreshToken = jwtUtils.createToken(tokenPayload, refreshTokenExpireTime);
        tokenCacheRepository.save(new TokenCache(username, accessToken, refreshToken));
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
                .get(tokenPayloadUsernameKey, String.class);
        final String usernameFromRT = jwtUtils.tokenClaimsFrom(refreshToken, true)
                .get(tokenPayloadUsernameKey, String.class);
        if (!usernameFromAT.equals(usernameFromRT)) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "Reissue request is invalid");
        }

        final TokenCache tokenCache = tokenCacheRepository.findById(usernameFromAT)
                .orElseThrow(() -> new AuthException(HttpStatus.UNAUTHORIZED, "Token is invalid"));
        if (!tokenCache.getRefreshToken().equals(refreshToken) || !tokenCache.getAccessToken().equals(accessToken)) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "Reissue request is invalid");
        }
        return createTokensByUsername(tokenCache.getUsername());
    }
}
