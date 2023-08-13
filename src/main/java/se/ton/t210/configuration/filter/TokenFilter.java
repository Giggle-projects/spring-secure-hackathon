package se.ton.t210.configuration.filter;

import java.io.IOException;
import java.util.Arrays;
import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.web.filter.OncePerRequestFilter;
import se.ton.t210.domain.TokenSecret;
import se.ton.t210.exception.AuthException;

public class TokenFilter extends OncePerRequestFilter {

    private final String tokenCookieKey;
    private final TokenSecret secret;

    public TokenFilter(TokenSecret secret, String tokenCookieKey) {
        this.secret = secret;
        this.tokenCookieKey = tokenCookieKey;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain) throws ServletException, IOException {
        try {
            final String refreshToken = getTokenFromCookies(tokenCookieKey, request.getCookies());
            secret.validateToken(refreshToken);

        } catch (AuthException e) {
            response.setStatus(HttpStatus.UNAUTHORIZED.value());
            response.getOutputStream().write(e.getMessage().getBytes());
            return;
        }

        filterChain.doFilter(request, response);
    }

    private String getTokenFromCookies(String keyName, Cookie[] cookies) {
        if (cookies == null) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "Cookie corrupted");
        }
        return Arrays.stream(cookies)
                .filter(cookie -> cookie.getName().equals(keyName))
                .map(Cookie::getValue)
                .findFirst()
                .orElseThrow(() -> new AuthException(HttpStatus.UNAUTHORIZED, "JWT Token is not found"));
    }
}
