package se.ton.t210.configuration.filter;

import org.springframework.http.HttpStatus;
import org.springframework.web.filter.OncePerRequestFilter;
import se.ton.t210.domain.TokenSecret;
import se.ton.t210.dto.MemberTokens;
import se.ton.t210.exception.AuthException;
import se.ton.t210.service.token.TokenService;
import se.ton.t210.utils.http.CookieUtils;

import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.Arrays;

public class AccessTokenFilter extends OncePerRequestFilter {

    private final String accessTokenCookieKey;
    private final String refreshTokenCookieKey;
    private final TokenSecret secret;
    private final TokenService tokenService;

    public AccessTokenFilter(TokenSecret secret, String accessTokenCookieKey, String refreshTokenCookieKey, TokenService tokenService) {
        this.secret = secret;
        this.accessTokenCookieKey = accessTokenCookieKey;
        this.refreshTokenCookieKey = refreshTokenCookieKey;
        this.tokenService = tokenService;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain) throws ServletException, IOException {
        try {
            final String accessToken = getTokenFromCookies(accessTokenCookieKey, request.getCookies());
            secret.validateToken(accessToken);
            System.out.println("11");
        } catch (AuthException e) {
            try {
                System.out.println("22");
                final String refreshToken = getTokenFromCookies(refreshTokenCookieKey, request.getCookies());
                System.out.println("33");
                secret.validateToken(refreshToken);
                System.out.println("44");

                final MemberTokens tokens = tokenService.reissue(accessTokenCookieKey, refreshToken);
                CookieUtils.loadHttpOnlyCookie(response, accessTokenCookieKey, tokens.getAccessToken());
                CookieUtils.loadHttpOnlyCookie(response, refreshTokenCookieKey, tokens.getRefreshToken());
            } catch (AuthException ae) {
                ae.printStackTrace();
                response.setStatus(HttpStatus.UNAUTHORIZED.value());
                response.sendRedirect("/html/sign-in.html");
                return;
            }
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
