package se.ton.t210.cache;

import java.time.Duration;
import lombok.Getter;
import org.springframework.data.annotation.Id;
import org.springframework.data.redis.core.RedisHash;

import java.time.LocalTime;
import org.springframework.http.HttpStatus;
import se.ton.t210.exception.AuthException;

@Getter
@RedisHash(value = "EmailAuthCodeCache", timeToLive = 60 * 5)
public class EmailAuthCodeCache {

    @Id
    private final String email;
    private final String authCode;
    private final LocalTime createdTime;

    public EmailAuthCodeCache(String email, String authCode, LocalTime createdTime) {
        this.email = email;
        this.authCode = authCode;
        this.createdTime = createdTime;
    }

    public void checkValidTime(long mailValidTime) {
        long afterSeconds = Duration.between(createdTime, LocalTime.now()).getSeconds();
        if (afterSeconds > mailValidTime) {
            throw new AuthException(HttpStatus.REQUEST_TIMEOUT, "Email valid time is exceed");
        }
    }

    public void checkEmailSame(String email) {
        if (!this.email.equals(email)) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "Email is not correct");
        }
    }

    public void checkAuthCodeSame(String authCode) {
        if (!this.authCode.equals(authCode)) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "AuthCode is not correct");
        }
    }
}
