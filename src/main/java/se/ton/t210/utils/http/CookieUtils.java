package se.ton.t210.utils.http;

import org.springframework.stereotype.Component;

import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletResponse;
import java.util.Arrays;

@Component
public class CookieUtils {

    public static void loadHttpOnlyCookie(HttpServletResponse response, String key, String value) {
        loadHttpOnlyCookie(response, key, value, "/");
    }

    public static void loadHttpOnlyCookie(HttpServletResponse response, String key, String value, String path) {
        final Cookie cookie = new Cookie(key, value);
        cookie.setHttpOnly(true);
        cookie.setPath(path);
        response.addCookie(cookie);
    }

    public static void removeHttpOnlyCookie(HttpServletResponse response, String key) {
        removeHttpOnlyCookie(response, key, "/");
    }

    public static void removeHttpOnlyCookie(HttpServletResponse response, String key, String path) {
        final Cookie cookie = new Cookie(key, null);
        cookie.setMaxAge(0);
        cookie.setHttpOnly(true);
        cookie.setPath(path);
        response.addCookie(cookie);
    }

    public static String getTokenFromCookies(String keyName, Cookie[] cookies) {
        if (cookies == null) {
            return null;
        }
        return Arrays.stream(cookies)
            .filter(cookie -> cookie.getName().equals(keyName))
            .map(Cookie::getValue)
            .findFirst()
            .orElse(null);
    }
}
