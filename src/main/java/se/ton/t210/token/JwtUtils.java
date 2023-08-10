package se.ton.t210.token;

import io.jsonwebtoken.*;
import io.jsonwebtoken.security.Keys;
import org.springframework.http.HttpStatus;
import se.ton.t210.exception.AuthException;

import java.nio.charset.StandardCharsets;
import java.security.Key;
import java.time.Duration;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

public class JwtUtils {

    private final Key key;

    public JwtUtils(String secret) {
        this.key = Keys.hmacShaKeyFor(secret.getBytes(StandardCharsets.UTF_8));
    }

    public static final int ACCESS_TOKEN_EXPIRATION = 30;
    public static final int REFRESH_TOKEN_EXPIRATION = 3 * 60;

    public TokenData createTokenDataByUsername(String username) {
        return new TokenData(
                createAccessTokenByUsername(username),
                createRefreshTokenByUsername(username)
        );
    }

    public String createAccessTokenByUsername(String username) {
        return createToken(username, ACCESS_TOKEN_EXPIRATION);
    }

    public String createRefreshTokenByUsername(String username) {
        return createToken(username, REFRESH_TOKEN_EXPIRATION);
    }

    private String createToken(String username, int expireTime) {
        Map<String, Object> payloads = new HashMap<>();
        payloads.put("username", username);
        Date now = new Date();
        Date expiration = new Date(now.getTime() + Duration.ofSeconds(expireTime).toMillis());
        return Jwts.builder()
                .setHeaderParam(Header.TYPE, Header.JWT_TYPE)
                .setClaims(payloads)
                .setExpiration(expiration)
                .setSubject("user-auto")
                .signWith(key)
                .compact();
    }

    public boolean isExpired(String token) {
        try {
            Jwts.parserBuilder().setSigningKey(this.key).build().parseClaimsJws(token).getBody();
            return false;
        } catch (ExpiredJwtException e) {
            System.out.println("HI");
            return true;
        } catch (Exception e) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "Invalid JWT token");
        }
    }

    public void validateToken(String token) {
        try {
            Jwts.parserBuilder().setSigningKey(this.key).build().parseClaimsJws(token).getBody();
        } catch (io.jsonwebtoken.security.SecurityException | MalformedJwtException e) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "Invalid JWT signature.");
        } catch (ExpiredJwtException e) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "Expired JWT token.");
        } catch (UnsupportedJwtException e) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "Unsupported JWT token.");
        } catch (Exception e) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "Invalid JWT token.");
        }
    }

    public String getUsernameByToken(String token) {
        try {
            return Jwts.parserBuilder()
                    .setSigningKey(key)
                    .build()
                    .parseClaimsJws(token)
                    .getBody()
                    .get("username").toString();

        } catch (ExpiredJwtException e) {
            // 추가 구현 예정
            return e.getClaims().toString();
        }
    }
}
