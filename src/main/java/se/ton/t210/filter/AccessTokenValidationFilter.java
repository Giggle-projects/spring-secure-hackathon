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
import java.util.List;

public class AccessTokenValidationFilter extends OncePerRequestFilter {

    private final JwtUtils jwtUtils;

    public AccessTokenValidationFilter(JwtUtils jwtUtils) {
        this.jwtUtils = jwtUtils;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain) throws ServletException, IOException {
        try {
            String accessTokenCookieKey = "accessToken";
            String accessToken = getTokenFromCookies(accessTokenCookieKey, request.getCookies());
            jwtUtils.validateToken(accessToken);

        } catch (AuthException e) {
            filterChain.doFilter(request, response);
        }
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

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        List<String> excludedUrls = List.of("/api/auth/signUp", "/api/auth/signIn");
        return excludedUrls.stream().anyMatch(request.getRequestURI()::contains);
    }
}
