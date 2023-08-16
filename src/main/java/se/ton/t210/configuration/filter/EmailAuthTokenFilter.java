package se.ton.t210.configuration.filter;

import org.springframework.http.HttpStatus;
import org.springframework.web.filter.OncePerRequestFilter;
import se.ton.t210.domain.TokenSecret;
import se.ton.t210.exception.AuthException;

import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.Arrays;

import static se.ton.t210.utils.http.CookieUtils.getTokenFromCookies;

public class EmailAuthTokenFilter extends OncePerRequestFilter {

    private final String tokenCookieKey;
    private final TokenSecret secret;

    public EmailAuthTokenFilter(TokenSecret secret, String tokenCookieKey) {
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
}
