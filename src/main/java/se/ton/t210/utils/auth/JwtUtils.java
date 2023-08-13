package se.ton.t210.utils.auth;

import io.jsonwebtoken.*;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import se.ton.t210.exception.AuthException;

import java.security.Key;
import java.time.Duration;
import java.util.Date;
import java.util.Map;

@Component
public class JwtUtils {

    public static String createToken(Key key, Map<String, Object> payloads, int expireTime) {
        final Date now = new Date();
        final Date expiration = new Date(now.getTime() + Duration.ofSeconds(expireTime).toMillis());
        return Jwts.builder()
            .setHeaderParam(Header.TYPE, Header.JWT_TYPE)
            .setClaims(payloads)
            .setExpiration(expiration)
            .setSubject("user-auto")
            .signWith(key)
            .compact();
    }

    public static boolean isExpired(Key key, String token) {
        try {
            Jwts.parserBuilder().setSigningKey(key).build().parseClaimsJws(token).getBody();
            return false;
        } catch (ExpiredJwtException e) {
            return true;
        } catch (Exception e) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "Invalid JWT token");
        }
    }

    public static void validate(Key key, String token) {
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

    public static Claims tokenClaimsFrom(Key key, String token, boolean ignoreExpired) {
        try {
            return Jwts.parserBuilder()
                .setSigningKey(key)
                .build()
                .parseClaimsJws(token)
                .getBody();
        } catch (ExpiredJwtException e) {
            if (ignoreExpired) {
                return e.getClaims();
            }
            throw new IllegalArgumentException("This is not valid JWT token");
        }
    }
}
