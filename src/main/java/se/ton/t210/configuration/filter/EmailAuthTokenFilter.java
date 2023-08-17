package se.ton.t210.configuration.filter;

import org.springframework.http.HttpStatus;
import org.springframework.web.filter.OncePerRequestFilter;
import se.ton.t210.domain.TokenSecret;
import se.ton.t210.exception.AuthException;

import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import se.ton.t210.service.token.TokenService;

import static se.ton.t210.utils.http.CookieUtils.getTokenFromCookies;

public class EmailAuthTokenFilter extends OncePerRequestFilter {

    private final String tokenCookieKey;
    private final TokenService tokenService;

    public EmailAuthTokenFilter(String tokenCookieKey, TokenService tokenService) {
        this.tokenCookieKey = tokenCookieKey;
        this.tokenService = tokenService;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain) throws ServletException, IOException {
        try {
            final String refreshToken = getTokenFromCookies(tokenCookieKey, request.getCookies());
            tokenService.validateToken(refreshToken);
        } catch (AuthException e) {
            response.setStatus(HttpStatus.UNAUTHORIZED.value());
            response.getOutputStream().write(e.getMessage().getBytes());
            return;
        }
        filterChain.doFilter(request, response);
    }
}
