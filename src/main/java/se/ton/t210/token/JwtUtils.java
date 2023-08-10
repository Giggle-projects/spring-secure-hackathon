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

    public String createToken(Map<String, Object> payloads, int expireTime) {
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
            return true;
        } catch (Exception e) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "Invalid JWT token");
        }
    }

    public void validateToken(String token) {
        try {
            Jwts.parserBuilder().setSigningKey(key).build().parseClaimsJws(token).getBody();
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

    public Claims tokenClaimsFrom(String token) {
        return tokenClaimsFrom(token, false);
    }

    public Claims tokenClaimsFrom(String token, boolean ignoreExpired) {
        try {
            return Jwts.parserBuilder()
                    .setSigningKey(key)
                    .build()
                    .parseClaimsJws(token)
                    .getBody();
        } catch (ExpiredJwtException e) {
            if(ignoreExpired) {
                return e.getClaims();
            }
            throw new IllegalArgumentException("This is not valid JWT token");
        }
    }
}
