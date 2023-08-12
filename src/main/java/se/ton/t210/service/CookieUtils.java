package se.ton.t210.service;

import org.springframework.stereotype.Component;

import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletResponse;

@Component
public class CookieUtils {

    public static void responseTokens(HttpServletResponse response, String cookieKey, String cookieValue) {
        loadTokenCookie(response, cookieKey, cookieValue);
    }

    public static void loadTokenCookie(HttpServletResponse response, String key, String value) {
        final Cookie cookie = new Cookie(key, value);
        cookie.setHttpOnly(true);
        cookie.setPath("/");
        response.addCookie(cookie);
    }
}
