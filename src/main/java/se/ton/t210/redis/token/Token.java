package se.ton.t210.redis.token;

import lombok.Getter;
import org.springframework.data.annotation.Id;
import org.springframework.data.redis.core.RedisHash;

@Getter
@RedisHash(value = "token", timeToLive = 60 * 3) //3일 60 * 60 * 24 * 3
public class Token {

    @Id
    private final String username;
    private final String accessToken;
    private final String refreshToken;

    public Token(String username, String accessToken, String refreshToken) {
        this.username = username;
        this.accessToken = accessToken;
        this.refreshToken = refreshToken;
    }
}
