package se.ton.t210.service.token;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import se.ton.t210.cache.EmailAuthMailCache;
import se.ton.t210.cache.EmailAuthMailCacheRepository;
import se.ton.t210.cache.TokenCache;
import se.ton.t210.cache.TokenCacheRepository;
import se.ton.t210.domain.TokenSecret;
import se.ton.t210.dto.MemberTokens;
import se.ton.t210.exception.AuthException;
import se.ton.t210.utils.auth.AuthCodeUtils;

import java.util.Map;

@Service
public class TokenService {

    @Value("${auth.jwt.payload.key:email}")
    private String tokenKey;

    @Value("${auth.jwt.token.access.ttl.time:1800}")
    private int accessTokenExpireTime;

    @Value("${auth.jwt.token.refresh.ttl.time:259200}")
    private int refreshTokenExpireTime;

    @Value("${auth.jwt.token.auth.ttl.time:1800}")
    private int emailAuthTokenExpireTime;

    private final TokenSecret tokenSecret;
    private final TokenCacheRepository tokenCacheRepository;
    private final EmailAuthMailCacheRepository emailAuthMailCacheRepository;

    public TokenService(TokenSecret tokenSecret, TokenCacheRepository tokenCacheRepository, EmailAuthMailCacheRepository emailAuthMailCacheRepository) {
        this.tokenSecret = tokenSecret;
        this.tokenCacheRepository = tokenCacheRepository;
        this.emailAuthMailCacheRepository = emailAuthMailCacheRepository;
    }

    public void saveEmailAuthInfoInCache(String email, String emailAuthCode) {
        emailAuthMailCacheRepository.save(new EmailAuthMailCache(email, emailAuthCode));
    }

    public String issueMailToken(String email) {
        final Map<String, Object> tokenPayload = Map.of(tokenKey, email);
        return tokenSecret.createToken(tokenPayload, emailAuthTokenExpireTime);
    }

    public MemberTokens issue(String email) {
        final Map<String, Object> tokenPayload = Map.of(tokenKey, email);
        final String accessToken = tokenSecret.createToken(tokenPayload, accessTokenExpireTime);
        final String refreshToken = tokenSecret.createToken(tokenPayload, refreshTokenExpireTime);
        tokenCacheRepository.save(new TokenCache(email, accessToken, refreshToken));
        return new MemberTokens(accessToken, refreshToken);
    }

    public MemberTokens reissue(String accessToken, String refreshToken) {
        tokenSecret.validateToken(refreshToken);
        if (tokenSecret.isExpired(accessToken)) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "You can't reissue token with unexpired access token");
        }
        final String userEmail = tokenSecret.getPayloadValue(tokenKey, accessToken);
        if (!userEmail.equals(tokenSecret.getPayloadValue(tokenKey, refreshToken))) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "Reissue request is invalid");
        }
        validateCachedToken(accessToken, refreshToken, userEmail);
        return issue(userEmail);
    }

    private void validateCachedToken(String email, String accessToken, String refreshToken) {
        tokenCacheRepository.findById(email)
            .orElseThrow(() -> new AuthException(HttpStatus.UNAUTHORIZED, "Token is invalid"))
            .validate(accessToken, refreshToken, refreshTokenExpireTime);
    }
}
