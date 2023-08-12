package se.ton.t210.utils.http;

import org.springframework.stereotype.Component;

import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletResponse;

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
}
