package se.ton.t210.cache;

import lombok.Getter;
import org.springframework.data.annotation.Id;
import org.springframework.data.redis.core.RedisHash;
import org.springframework.http.HttpStatus;
import se.ton.t210.exception.AuthException;

import java.time.Duration;
import java.time.LocalTime;

@Getter
@RedisHash(value = "token", timeToLive = 60 * 60 * 24 * 3)
public class TokenCache {

    @Id
    private final String email;
    private final String accessToken;
    private final String refreshToken;
    private final LocalTime createdTime;

    public TokenCache(String email, String accessToken, String refreshToken, LocalTime createdTime) {
        this.email = email;
        this.accessToken = accessToken;
        this.refreshToken = refreshToken;
        this.createdTime =  createdTime;
    }

    public void validate(String accessToken, String refreshToken, int refreshTokenExpireTime) {
        final boolean isValid = this.accessToken.equals(accessToken)
            && this.refreshToken.equals(refreshToken)
            && Duration.between(this.createdTime, LocalTime.now()).getSeconds() < refreshTokenExpireTime;
        if(!isValid) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "Reissue request is invalid");
        }
    }
}
