package se.ton.t210.cache;

import lombok.Getter;
import org.springframework.data.annotation.Id;
import org.springframework.data.redis.core.RedisHash;

import java.time.LocalTime;

@Getter
@RedisHash(value = "EmailAuthMailCache", timeToLive = 60 * 5)
public class EmailAuthMailCache {

    @Id
    private final String email;
    private final String authCode;
    private final LocalTime createdTime;

    public EmailAuthMailCache(String email, String authCode, LocalTime createdTime) {
        this.email = email;
        this.authCode = authCode;
        this.createdTime = createdTime;
    }
}
