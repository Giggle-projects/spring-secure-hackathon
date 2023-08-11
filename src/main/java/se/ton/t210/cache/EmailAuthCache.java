package se.ton.t210.cache;

import org.springframework.data.annotation.Id;
import org.springframework.data.redis.core.RedisHash;

import java.time.LocalTime;

@RedisHash(value = "emailAuth", timeToLive = 60 * 5)
public class EmailAuthCache {

    @Id
    private final String email;
    private final String authCode;
    private final LocalTime time;

    public EmailAuthCache(String email, String authCode, LocalTime time) {
        this.email = email;
        this.authCode = authCode;
        this.time = time;
    }
}
