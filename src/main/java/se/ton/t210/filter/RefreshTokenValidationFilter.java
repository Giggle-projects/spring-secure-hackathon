package se.ton.t210.filter;

import org.springframework.http.HttpStatus;
import org.springframework.web.filter.OncePerRequestFilter;
import se.ton.t210.exception.AuthException;
import se.ton.t210.token.JwtUtils;

import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.Arrays;

public class RefreshTokenValidationFilter extends OncePerRequestFilter {

    private final JwtUtils jwtUtils;

    public RefreshTokenValidationFilter(JwtUtils jwtUtils) {
        this.jwtUtils = jwtUtils;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain) throws ServletException, IOException {
        try {
            final String refreshTokenCookieKey = "refreshToken";
            String refreshToken = getTokenFromCookies(refreshTokenCookieKey, request.getCookies());
            jwtUtils.validateToken(refreshToken);

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
